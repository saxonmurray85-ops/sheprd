"""
Interactive terminal console for Sheprd agents.
Provides a high-contrast digital green terminal interface, real-time token streaming,
Herdr pane status reporting, and multi-agent invocation.
"""

import asyncio
import json
import os
import readline
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp

from .database import AgentRecord, Database
from .herdr_integration import HerdrIntegration
from .prompts import build_agent_system_prompt
from .server_manager import ServerManager
from .tool_hub import ToolHub

# Pastel ANSI Palette (Blue, Sage Green, Slate Grey, Cream)
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_BLUE = "\033[38;5;117m"        # Soft pastel sky blue
C_GREEN = "\033[38;5;151m"       # Soft pastel sage green
C_BRIGHT_GREEN = "\033[38;5;157m"# Radiant pastel mint
C_SLATE = "\033[38;5;248m"       # Refined pastel slate grey
C_CREAM = "\033[38;5;230m"       # Origami paper cream
C_CYAN = "\033[38;5;153m"        # Pastel ice cyan
C_AMBER = "\033[38;5;222m"       # Soft pastel sand
C_RED = "\033[38;5;217m"         # Soft pastel rose
C_BG_ACCENT = "\033[48;5;236m"


BANNER_ART = r"""
   _____   ___   _     ___  
  |  ___| / _ \ | |   |   \ 
  | |_   | | | || |   | |) |
  |_|     \___/ |_|___|___/ 
        Fold of Sheep // Herdr Fleet
"""


class InteractiveChatSession:
    def __init__(self, agent_name: str, db: Optional[Database] = None):
        self.db = db or Database()
        self.agent_name = agent_name
        self.server_mgr = ServerManager(self.db)
        self.pane_id = os.environ.get("HERDR_PANE_ID")
        self.agent = self.db.get_agent_by_name(agent_name)
        self.tool_hub = ToolHub(self.db)
        self.history: List[Dict[str, str]] = []

    def ensure_server_running(self) -> bool:
        if not self.agent:
            print(f"{C_RED}[ERROR]{C_RESET} Agent '{self.agent_name}' not found in Fold registry.")
            return False

        if self.server_mgr.check_health(self.agent.port):
            return True

        print(f"{C_BLUE}[FOLD]{C_RESET} Activating local model for {C_BOLD}{self.agent_name}{C_RESET} on port {self.agent.port}...")
        ok, msg = self.server_mgr.ensure_agent_running(self.agent_name, timeout_sec=35)
        if not ok:
            print(f"{C_RED}[ERROR]{C_RESET} {msg}")
            return False
        print(f"{C_BRIGHT_GREEN}[ONLINE]{C_RESET} {msg}")
        self.agent = self.db.get_agent_by_name(self.agent_name)
        return True

    def build_system_prompt(self) -> str:
        agent = self.agent
        if not agent:
            return "You are an AI assistant."
        return build_agent_system_prompt(agent)

    async def _async_chat(self, user_text: str) -> str:
        messages = [{"role": "system", "content": self.build_system_prompt()}]
        # Cap context history window to last 20 messages (N13)
        messages.extend(self.history[-20:])
        messages.append({"role": "user", "content": user_text})

        async def _on_event(event_type: str, data: Dict[str, Any]) -> None:
            if event_type == "tool_call":
                fn = data.get("name", "")
                args_str = ", ".join(f"{k}={v!r}" for k, v in data.get("arguments", {}).items())
                print(f"\n  {C_CYAN}⚡ [TOOL CALL]{C_RESET} {C_BOLD}{fn}{C_RESET}({args_str})")
            elif event_type == "tool_result":
                res = str(data.get("result", ""))
                first_line = res.strip().split("\n")[0][:100]
                print(f"  {C_DIM}✓ [TOOL RESULT] {first_line}...{C_RESET}")

        async with aiohttp.ClientSession() as session:
            ans, _ = await self.tool_hub.chat_with_tools_loop(
                agent=self.agent,
                messages=messages,
                session=session,
                on_event=_on_event,
                max_iterations=4,
            )
            return ans

    def stream_completion(self, user_text: str) -> str:
        if not self.agent:
            return ""

        # Report Herdr working state
        if self.pane_id:
            HerdrIntegration.report_pane_status(
                self.pane_id, self.agent.name, self.agent.identity, "working"
            )

        try:
            full_reply_text = asyncio.run(self._async_chat(user_text))
            print(f"\n{C_BRIGHT_GREEN}{C_BOLD}{self.agent.name} ›{C_RESET} {C_GREEN}{full_reply_text}{C_RESET}\n")
            if full_reply_text:
                self.history.append({"role": "user", "content": user_text})
                self.history.append({"role": "assistant", "content": full_reply_text})
                # Cap history in memory (N13)
                self.history = self.history[-20:]
                self.db.log_chat(self.agent.name, "user", user_text)
                self.db.log_chat(self.agent.name, "assistant", full_reply_text)
            return full_reply_text
        except Exception as e:
            print(f"\n{C_RED}[Error]{C_RESET} {e}")
            return ""
        finally:
            if self.pane_id:
                HerdrIntegration.report_pane_status(
                    self.pane_id, self.agent.name, self.agent.identity, "idle"
                )

    def print_banner(self):
        agent = self.agent
        print(f"{C_BRIGHT_GREEN}{BANNER_ART}{C_RESET}")
        print(f"{C_DARK_GREEN}{'=' * 68}{C_RESET}")
        print(f"{C_BOLD}{C_BRIGHT_GREEN}  AGENT      :{C_RESET} {agent.name}")
        print(f"{C_BOLD}{C_GREEN}  IDENTITY   :{C_RESET} {agent.identity}")
        print(f"{C_BOLD}{C_GREEN}  PERSONALITY:{C_RESET} {agent.personality}")
        print(f"{C_BOLD}{C_GREEN}  JOB        :{C_RESET} {agent.job}")
        print(f"{C_BOLD}{C_GREEN}  MODEL      :{C_RESET} {os.path.basename(agent.model_path)} ({agent.model_architecture})")
        print(f"{C_BOLD}{C_GREEN}  SERVER     :{C_RESET} http://127.0.0.1:{agent.port} | GPU Layers: {agent.n_gpu_layers}")
        if self.pane_id:
            print(f"{C_BOLD}{C_CYAN}  HERDR PANE :{C_RESET} {self.pane_id}")
        print(f"{C_DARK_GREEN}{'=' * 68}{C_RESET}")
        print(f"{C_DIM}Commands: /help, /info, /clear, /exit | Type '@agent message' to call peers.{C_RESET}\n")

    def run(self):
        if not self.ensure_server_running():
            sys.exit(1)

        self.agent = self.db.get_agent_by_name(self.agent_name)
        if not self.agent:
            sys.exit(1)

        if self.pane_id:
            HerdrIntegration.report_pane_status(
                self.pane_id, self.agent.name, self.agent.identity, "idle"
            )

        self.print_banner()

        history_file = Path.home() / f".local/share/fold/history_{self.agent_name}"
        if history_file.exists():
            try:
                readline.read_history_file(str(history_file))
            except Exception:
                pass

        try:
            while True:
                try:
                    prompt = f"{C_BOLD}{C_BLUE}you ›{C_RESET} "
                    user_input = input(prompt).strip()

                    if not user_input:
                        continue

                    if user_input.startswith("/"):
                        cmd = user_input.lower().split()[0]
                        if cmd in ("/exit", "/quit", "/q"):
                            print(f"{C_GREEN}Disconnecting from {self.agent.name}. Server remains active.{C_RESET}")
                            break
                        elif cmd == "/clear":
                            self.history.clear()
                            print(f"{C_GREEN}[History cleared]{C_RESET}")
                            continue
                        elif cmd == "/info":
                            self.print_banner()
                            continue
                        elif cmd == "/help":
                            print(f"\n{C_BOLD}Available commands:{C_RESET}")
                            print("  /clear - Clear active conversation history")
                            print("  /info  - Show model and agent specifications")
                            print("  /exit  - Leave the chat session")
                            print("  @<agent> <msg> - Query another callable agent\n")
                            continue
                        else:
                            print(f"{C_AMBER}Unknown command: {cmd}. Type /help for assistance.{C_RESET}")
                            continue

                    # Check for @agent delegation
                    if user_input.startswith("@"):
                        parts = user_input.split(" ", 1)
                        target_name = parts[0][1:].strip()
                        query = parts[1].strip() if len(parts) > 1 else ""

                        target_agent = self.db.get_agent_by_name(target_name)
                        if not target_agent:
                            print(f"{C_RED}[Error]{C_RESET} Sibling agent '@{target_name}' not found.")
                            continue

                        if not target_agent.callable_by_agents:
                            print(f"{C_AMBER}[Notice]{C_RESET} Agent '@{target_name}' is private (callable_by_agents is False).")
                            continue

                        print(f"{C_CYAN}[Calling @{target_name}...]{C_RESET}")
                        # Query sibling agent
                        sibling_session = InteractiveChatSession(target_name)
                        sibling_session.ensure_server_running()
                        sibling_session.stream_completion(query)
                        continue

                    self.stream_completion(user_input)

                except KeyboardInterrupt:
                    print(f"\n{C_AMBER}[Prompt cancelled. Type /exit to leave.]{C_RESET}")
                    continue
                except EOFError:
                    print(f"\n{C_GREEN}Session terminated.{C_RESET}")
                    break
        finally:
            try:
                history_file.parent.mkdir(parents=True, exist_ok=True)
                readline.write_history_file(str(history_file))
                os.chmod(history_file, 0o600)
            except Exception:
                pass


def main():
    if len(sys.argv) < 2:
        print(f"{C_RED}Usage:{C_RESET} python3 -m fold.interactive_chat <agent_name>")
        sys.exit(1)
    agent_name = sys.argv[1]
    session = InteractiveChatSession(agent_name)
    session.run()


if __name__ == "__main__":
    main()
