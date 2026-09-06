"""
Telegram Bot integration service for Sheprd agents.
Runs asynchronous long-polling bots that map incoming Telegram chats
directly to local llama.cpp agent servers with full persona prompting.
"""

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import aiohttp

from .database import AgentRecord, Database
from .security import mask_token, sanitize_log_text


logger = logging.getLogger("sheprd.telegram")

TELEGRAM_TOKEN_REGEX = re.compile(r"^\d{8,12}:[A-Za-z0-9_-]{30,45}$")


class TelegramBotWorker:
    def __init__(self, agent_name: str, db: Database):
        self.agent_name = agent_name
        self.db = db
        self.running = False
        self.task: Optional[asyncio.Task] = None
        self.chat_contexts: Dict[int, List[Dict[str, str]]] = {}

    def get_agent(self) -> Optional[AgentRecord]:
        return self.db.get_agent_by_name(self.agent_name)

    async def start(self) -> Tuple[bool, str]:
        agent = self.get_agent()
        if not agent:
            return False, f"Agent '{self.agent_name}' not found."

        if not agent.telegram_bot_token:
            return False, f"Agent '{self.agent_name}' has no Telegram bot token configured."

        token = agent.telegram_bot_token.strip()
        if not TELEGRAM_TOKEN_REGEX.match(token):
            return False, "Invalid Telegram bot token format."

        if self.running:
            return True, "Telegram worker already running."

        self.running = True
        self.task = asyncio.create_task(self._poll_loop(token))
        return True, f"Telegram bot started for {self.agent_name} ({mask_token(token)})."

    async def stop(self) -> None:
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.task = None

    async def _poll_loop(self, token: str) -> None:
        offset = 0
        base_url = f"https://api.telegram.org/bot{token}"

        timeout_cfg = aiohttp.ClientTimeout(total=35)
        async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
            # Verify bot credentials
            try:
                async with session.get(f"{base_url}/getMe") as resp:
                    if resp.status != 200:
                        logger.error("Failed to verify Telegram token for %s (status %d)", self.agent_name, resp.status)
                        self.running = False
                        return
                    me_data = await resp.json()
                    bot_username = me_data.get("result", {}).get("username", "Unknown")
                    logger.info("Sheprd Telegram connected: @%s for agent %s", bot_username, self.agent_name)
            except Exception as e:
                clean_err = sanitize_log_text(str(e), [token])
                logger.error("Error connecting to Telegram for %s: %s", self.agent_name, clean_err)
                self.running = False
                return

            while self.running:
                try:
                    url = f"{base_url}/getUpdates?offset={offset}&timeout=20"
                    async with session.get(url) as resp:
                        if resp.status != 200:
                            await asyncio.sleep(5)
                            continue

                        data = await resp.json()
                        updates = data.get("result", [])
                        for update in updates:
                            update_id = update.get("update_id", 0)
                            offset = max(offset, update_id + 1)
                            await self._handle_update(session, base_url, update, token)

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    clean_err = sanitize_log_text(str(e), [token])
                    logger.warning("Telegram polling error for %s: %s", self.agent_name, clean_err)
                    await asyncio.sleep(3)

    async def _handle_update(
        self, session: aiohttp.ClientSession, base_url: str, update: Dict[str, Any], token: str
    ) -> None:
        message = update.get("message", {})
        chat_id = message.get("chat", {}).get("id")
        text = (message.get("text") or "").strip()

        if not chat_id or not text:
            return

        agent = self.get_agent()
        if not agent:
            return

        # S3: Check Telegram chat/user allowlist if configured
        user_info = message.get("from", {})
        user_id = user_info.get("id")
        username = user_info.get("username", "")

        allowed_users_env = os.environ.get("SHEPRD_TELEGRAM_ALLOWED_USERS", "").strip()
        if allowed_users_env:
            allowed_set = {u.strip().lower() for u in allowed_users_env.split(",") if u.strip()}
            is_allowed = (
                str(chat_id) in allowed_set
                or str(user_id) in allowed_set
                or f"@{username}".lower() in allowed_set
                or username.lower() in allowed_set
            )
            if not is_allowed:
                logger.warning(
                    "Blocked unauthorized Telegram message to agent %s from user_id=%s username=@%s",
                    self.agent_name, user_id, username
                )
                await self._send_message(
                    session, base_url, chat_id,
                    "⛔ *Unauthorized*: You are not on the allowlist for this local Sheprd agent.",
                    parse_mode="Markdown"
                )
                return

        # Commands
        if text.startswith("/"):
            cmd = text.split()[0].lower()
            if cmd == "/start":
                welcome = (
                    f"👋 Hello! I am *{agent.name}*.\n\n"
                    f"*Identity:* {agent.identity}\n"
                    f"*Personality:* {agent.personality}\n"
                    f"*Mission:* {agent.job}\n\n"
                    f"Send me any message to begin, or /reset to start fresh."
                )
                await self._send_message(session, base_url, chat_id, welcome, parse_mode="Markdown")
                return
            elif cmd == "/reset":
                self.chat_contexts[chat_id] = []
                await self._send_message(session, base_url, chat_id, "🧹 Conversation memory cleared.")
                return
            elif cmd == "/status":
                status_text = (
                    f"🟢 *Agent:* {agent.name}\n"
                    f"🤖 *Model:* `{os.path.basename(agent.model_path)}`\n"
                    f"⚡ *Context:* {agent.context_size} tokens\n"
                    f"🌐 *Local Server:* Port {agent.port}\n"
                    f"👥 *Groups:* {', '.join(agent.groups)}"
                )
                await self._send_message(session, base_url, chat_id, status_text, parse_mode="Markdown")
                return

        # Send typing indicator
        await self._send_chat_action(session, base_url, chat_id, "typing")

        # Query local agent server
        context = self.chat_contexts.setdefault(chat_id, [])
        system_prompt = (
            f"You are {agent.name}.\n"
            f"IDENTITY: {agent.identity}\n"
            f"PERSONALITY: {agent.personality}\n"
            f"PRIMARY JOB: {agent.job}\n\n"
            f"Answer thoughtfully and consistently according to your assigned character and duties."
        )

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(context[-10:])  # Keep last 10 messages for context
        messages.append({"role": "user", "content": text})

        url = f"http://127.0.0.1:{agent.port}/v1/chat/completions"
        payload = {
            "model": agent.name,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1500,
        }

        try:
            async with session.post(url, json=payload, timeout=90) as resp:
                if resp.status == 200:
                    res_data = await resp.json()
                    content = res_data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    if content:
                        context.append({"role": "user", "content": text})
                        context.append({"role": "assistant", "content": content})
                        self.db.log_chat(agent.name, f"telegram:{chat_id}", text)
                        self.db.log_chat(agent.name, "assistant", content)
                        await self._send_message(session, base_url, chat_id, content)
                else:
                    await self._send_message(
                        session,
                        base_url,
                        chat_id,
                        f"⚠️ Local inference server returned status {resp.status}. Is the model loaded?",
                    )
        except Exception as e:
            await self._send_message(
                session, base_url, chat_id, f"⚠️ Failed to reach local agent server: {e}"
            )

    async def _send_message(
        self,
        session: aiohttp.ClientSession,
        base_url: str,
        chat_id: int,
        text: str,
        parse_mode: Optional[str] = None,
    ) -> None:
        payload: Dict[str, Any] = {"chat_id": chat_id, "text": text}
        if parse_mode:
            payload["parse_mode"] = parse_mode
        try:
            await session.post(f"{base_url}/sendMessage", json=payload, timeout=10)
        except Exception:
            # Fallback without markdown if markdown parsing failed
            if parse_mode:
                payload.pop("parse_mode", None)
                try:
                    await session.post(f"{base_url}/sendMessage", json=payload, timeout=10)
                except Exception:
                    pass

    async def _send_chat_action(
        self, session: aiohttp.ClientSession, base_url: str, chat_id: int, action: str
    ) -> None:
        try:
            await session.post(
                f"{base_url}/sendChatAction",
                json={"chat_id": chat_id, "action": action},
                timeout=5,
            )
        except Exception:
            pass


class TelegramServiceManager:
    """Manages Telegram bot workers for all registered Sheprd agents."""

    def __init__(self, db: Database):
        self.db = db
        self.workers: Dict[str, TelegramBotWorker] = {}

    async def start_agent_bot(self, agent_name: str) -> Tuple[bool, str]:
        if agent_name in self.workers and self.workers[agent_name].running:
            return True, "Bot already active."
        worker = TelegramBotWorker(agent_name, self.db)
        ok, msg = await worker.start()
        if ok:
            self.workers[agent_name] = worker
        return ok, msg

    async def stop_agent_bot(self, agent_name: str) -> None:
        worker = self.workers.pop(agent_name, None)
        if worker:
            await worker.stop()

    async def sync_all(self) -> None:
        """Starts bots for all agents that have telegram_enabled and a valid token."""
        for agent in self.db.list_agents():
            if agent.telegram_enabled and agent.telegram_bot_token:
                await self.start_agent_bot(agent.name)
            else:
                await self.stop_agent_bot(agent.name)

    async def stop_all(self) -> None:
        for name in list(self.workers.keys()):
            await self.stop_agent_bot(name)
