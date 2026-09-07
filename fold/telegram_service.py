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
from .prompts import build_agent_system_prompt
from .security import mask_token, sanitize_log_text
from .tool_hub import ToolHub


logger = logging.getLogger("fold.telegram")

TELEGRAM_TOKEN_REGEX = re.compile(r"^\d{8,12}:[A-Za-z0-9_-]{30,45}$")


LOCK_DIR = Path.home() / ".local/share/fold"


class TelegramBotWorker:
    def __init__(self, agent_name: str, db: Database, server_mgr: Optional[Any] = None):
        self.agent_name = agent_name
        self.db = db
        from .server_manager import ServerManager
        self.server_mgr = server_mgr or ServerManager(self.db)
        self.running = False
        self.task: Optional[asyncio.Task] = None
        self.chat_contexts: Dict[int, List[Dict[str, str]]] = {}
        self.bot_username: str = "Unknown"
        self.tool_hub = ToolHub(self.db)
        self._lock_file = LOCK_DIR / f"tg_{self.agent_name}.lock"

    def get_agent(self) -> Optional[AgentRecord]:
        return self.db.get_agent_by_name(self.agent_name)

    def _acquire_lock(self) -> Tuple[bool, str]:
        """Verify and acquire a PID lockfile for this Telegram worker (N10)."""
        LOCK_DIR.mkdir(parents=True, exist_ok=True)
        if self._lock_file.exists():
            try:
                old_pid = int(self._lock_file.read_text().strip())
                os.kill(old_pid, 0)
                return False, f"Telegram worker already active (PID {old_pid})."
            except (ValueError, OSError):
                try:
                    self._lock_file.unlink(missing_ok=True)
                except OSError:
                    pass
        try:
            fd = os.open(str(self._lock_file), os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as f:
                f.write(str(os.getpid()))
            return True, "Lock acquired."
        except OSError as e:
            return False, f"Could not acquire lock: {e}"

    def _release_lock(self) -> None:
        if self._lock_file.exists():
            try:
                if int(self._lock_file.read_text().strip()) == os.getpid():
                    self._lock_file.unlink(missing_ok=True)
            except Exception:
                pass

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

        locked, lock_msg = self._acquire_lock()
        if not locked:
            return False, lock_msg

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
        self._release_lock()

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
                    self.bot_username = me_data.get("result", {}).get("username", "Unknown")
                    logger.info("Fold Telegram connected: @%s for agent %s", self.bot_username, self.agent_name)
            except Exception as e:
                clean_err = sanitize_log_text(str(e), [token])
                logger.error("Error connecting to Telegram for %s: %s", self.agent_name, clean_err)
                self.running = False
                return

            # Warn once if no Telegram allowlist configured (N2/S3)
            allowed_check = os.environ.get("FOLD_TELEGRAM_ALLOWED_USERS") or os.environ.get("SHEPRD_TELEGRAM_ALLOWED_USERS", "")
            if not allowed_check.strip():
                logger.warning(
                    "Telegram bot for '%s' running with NO user allowlist configured (FOLD_TELEGRAM_ALLOWED_USERS not set) — open to all users.",
                    self.agent_name
                )

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
        message = update.get("message") or update.get("edited_message", {})
        chat_id = message.get("chat", {}).get("id")
        text = (message.get("text") or "").strip()

        if not chat_id or not text:
            return

        agent = self.get_agent()
        if not agent:
            return

        # Strip bot username mention if present in group or direct chat (@username)
        if self.bot_username and self.bot_username != "Unknown":
            text = re.sub(rf"@{re.escape(self.bot_username)}\b", "", text, flags=re.IGNORECASE).strip()

        # S3: Check Telegram chat/user allowlist if configured
        user_info = message.get("from", {})
        user_id = user_info.get("id")
        username = user_info.get("username", "")

        logger.debug(
            "Telegram incoming message for agent '%s' from user_id=%s username=@%s: %s",
            self.agent_name, user_id, username, text[:60]
        )

        allowed_users_env = (os.environ.get("FOLD_TELEGRAM_ALLOWED_USERS") or os.environ.get("SHEPRD_TELEGRAM_ALLOWED_USERS", "")).strip()
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
                    "⛔ *Unauthorized*: You are not on the allowlist for this local Fold agent.",
                    parse_mode="Markdown"
                )
                return

        # Commands
        if text.startswith("/"):
            raw_cmd = text.split()[0].lower()
            cmd = raw_cmd.split("@")[0]
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

        # Ensure local model server is running (LRU hot-swapping)
        ok, srv_err = await asyncio.to_thread(self.server_mgr.ensure_agent_running, agent.name)
        if not ok:
            logger.error("Failed to ensure agent server running for %s: %s", agent.name, srv_err)
            await self._send_message(
                session, base_url, chat_id, f"⚠️ Local agent server could not be started: {srv_err}"
            )
            return
        agent = self.get_agent()
        if not agent:
            return

        # Send typing indicator
        await self._send_chat_action(session, base_url, chat_id, "typing")

        # Query local agent server
        context = self.chat_contexts.setdefault(chat_id, [])
        system_prompt = build_agent_system_prompt(agent)

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(context[-20:])  # Cap at last 20 messages for context (N13)
        messages.append({"role": "user", "content": text})

        async def _on_tool_event(event_type: str, data: Dict[str, Any]) -> None:
            if event_type == "tool_call":
                fn_name = data.get("name", "")
                await self._send_chat_action(session, base_url, chat_id, "typing")
                logger.info("Telegram: Agent %s calling tool '%s' for chat_id %s", agent.name, fn_name, chat_id)

        try:
            content, updated_msgs = await self.tool_hub.chat_with_tools_loop(
                agent=agent,
                messages=messages,
                session=session,
                on_event=_on_tool_event,
                max_iterations=4,
            )
            if content:
                context.append({"role": "user", "content": text})
                context.append({"role": "assistant", "content": content})
                self.chat_contexts[chat_id] = context[-20:]  # Cap memory
                self.db.log_chat(agent.name, f"telegram:{chat_id}", text)
                self.db.log_chat(agent.name, "assistant", content)
                await self._send_message(session, base_url, chat_id, content)
            else:
                await self._send_message(
                    session,
                    base_url,
                    chat_id,
                    "⚠️ Local inference server returned an empty response.",
                )
        except Exception as e:
            logger.error("Error running tool loop for agent %s on Telegram: %s", agent.name, e)
            await self._send_message(
                session, base_url, chat_id, f"⚠️ Failed to reach local agent server: {e}"
            )

    async def _send_single_message(
        self,
        session: aiohttp.ClientSession,
        base_url: str,
        chat_id: int,
        text: str,
        parse_mode: Optional[str] = None,
    ) -> bool:
        payload: Dict[str, Any] = {"chat_id": chat_id, "text": text}
        if parse_mode:
            payload["parse_mode"] = parse_mode
        try:
            async with session.post(f"{base_url}/sendMessage", json=payload, timeout=15) as resp:
                if resp.status == 200:
                    return True
                err_text = await resp.text()
                logger.warning(
                    "Telegram sendMessage failed for %s (status %d): %s",
                    self.agent_name, resp.status, err_text[:200]
                )
                # Fallback to plain text if markdown parsing failed (HTTP 400)
                if parse_mode and resp.status == 400:
                    payload.pop("parse_mode", None)
                    async with session.post(f"{base_url}/sendMessage", json=payload, timeout=15) as fallback_resp:
                        return fallback_resp.status == 200
        except Exception as e:
            logger.error("Error posting to Telegram for %s: %s", self.agent_name, e)
        return False

    async def _send_message(
        self,
        session: aiohttp.ClientSession,
        base_url: str,
        chat_id: int,
        text: str,
        parse_mode: Optional[str] = None,
    ) -> None:
        if not text:
            return
        # Telegram has a 4096-character limit per message
        max_chunk = 4000
        if len(text) <= max_chunk:
            await self._send_single_message(session, base_url, chat_id, text, parse_mode=parse_mode)
        else:
            chunks = [text[i : i + max_chunk] for i in range(0, len(text), max_chunk)]
            for chunk in chunks:
                await self._send_single_message(session, base_url, chat_id, chunk, parse_mode=None)

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

    def __init__(self, db: Database, server_mgr: Optional[Any] = None):
        self.db = db
        self.server_mgr = server_mgr
        self.workers: Dict[str, TelegramBotWorker] = {}

    async def start_agent_bot(self, agent_name: str) -> Tuple[bool, str]:
        if agent_name in self.workers and self.workers[agent_name].running:
            return True, "Bot already active."
        worker = TelegramBotWorker(agent_name, self.db, server_mgr=self.server_mgr)
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
