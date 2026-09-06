"""
Interactive terminal console for Sheprd agents.
Provides a high-contrast digital green terminal interface, real-time token streaming,
Herdr pane status reporting, and multi-agent invocation.
"""

import json
import os
import readline
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from .database import AgentRecord, Database
from .herdr_integration import HerdrIntegration
from .server_manager import ServerManager

# Digital Green ANSI Palette
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_GREEN = "\033[38;5;48m"       # Crisp phosphor green
C_BRIGHT_GREEN = "\033[38;5;84m" # Radiant glowing green
C_DARK_GREEN = "\033[38;5;28m"   # Subdued border green
C_CYAN = "\033[38;5;51m"
C_AMBER = "\033[38;5;214m"
C_RED = "\033[38;5;196m"
C_BG_GREEN = "\033[48;5;22m"


BANNER_ART = r"""
  ___ _  _ ___ ___ ___ ___ 
 / __| || | __| _ \ _ \   \ 
 \__ \ __ | _||  _/   / |) |
 |___/_||_|___|_| |_|_\___/ 
"""


class InteractiveChatSession:
    def __init__(self, agent_name: str):
        self.db = Database()
        self.agent_name = agent_name
        self.server_mgr = ServerManager(self.db)
        self.pane_id = os.environ.get("HERDR_PANE_ID")
        self.agent = self.db.get_agent_by_name(agent_name)
        self.history: List[Dict[str, str]] = []

    def ensure_server_running(self) -> bool:
        if not self.agent:
            print(f"{C_RED}[ERROR]{C_RESET} Agent '{self.agent_name}' not found in Sheprd registry.")
            return False

        if self.server_mgr.check_health(self.agent.port):
            return True

        print(f"{C_GREEN}[SHEPRD]{C_RESET} Spawning local server for {C_BOLD}{self.agent_name}{C_RESET} on port {self.agent.port}...")
        ok, msg = self.server_mgr.start_agent_server(self.agent_name, timeout_sec=25)
        if not ok:
            print(f"{C_RED}[ERROR]{C_RESET} {msg}")
            return False
        print(f"{C_BRIGHT_GREEN}[ONLINE]{C_RESET} {msg}")
        return True

    def build_system_prompt(self) -> str:
        agent = self.agent
        if not agent:
            return "You are an AI assistant."
        return (
            f"You are {agent.name}.\n"
            f"IDENTITY: {agent.identity}\n"
            f"PERSONALITY: {agent.personality}\n"
            f"PRIMARY JOB: {agent.job}\n\n"
            f"Instructions:\n"
            f"- Speak consistently in your specified identity and personality.\n"
            f"- Stay dedicated to your assigned job.\n"
            f"- Be clear, insightful, and direct."
        )

    def stream_completion(self, user_text: str) -> str:
        if not self.agent:
            return ""

        url = f"http://127.0.0.1:{self.agent.port}/v1/chat/completions"

        messages = [{"role": "system", "content": self.build_system_prompt()}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_text})

        payload = {
            "model": self.agent.name,
            "messages": messages,
            "stream": True,
            "temperature": 0.7,
            "max_tokens": 2048,
        }

        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data_bytes,
            headers={"Content-Type": "application/json", "User-Agent": "Sheprd-Terminal/1.0"},
        )

        full_reply = []

        # Report Herdr working state
        if self.pane_id:
            HerdrIntegration.report_pane_status(
                self.pane_id, self.agent.name, self.agent.identity, "working"
            )

        print(f"\n{C_BRIGHT_GREEN}{C_BOLD}{self.agent.name} ›{C_RESET} ", end="", flush=True)

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                for line_bytes in resp:
                    line = line_bytes.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                print(f"{C_GREEN}{content}{C_RESET}", end="", flush=True)
                                full_reply.append(content)
                        except json.JSONDecodeError:
                            continue
            print()  # Newline after stream finishes
        except Exception as e:
            print(f"\n{C_RED}[Connection Error]{C_RESET} {e}")
        finally:
            # Report Herdr idle state
            if self.pane_id:
                HerdrIntegration.report_pane_status(
                    self.pane_id, self.agent.name, self.agent.identity, "idle"
                )

        reply_text = "".join(full_reply)
        if reply_text:
            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": reply_text})
            self.db.log_chat(self.agent.name, "user", user_text)
            self.db.log_chat(self.agent.name, "assistant", reply_text)

        return reply_text

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

        history_file = Path.home() / f".local/share/sheprd/history_{self.agent_name}"
        if history_file.exists():
            try:
                readline.read_history_file(str(history_file))
            except Exception:
                pass

        try:
            while True:
                try:
                    prompt = f"{C_BOLD}{C_BRIGHT_GREEN}you ›{C_RESET} "
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
        print(f"{C_RED}Usage:{C_RESET} python3 -m sheprd.interactive_chat <agent_name>")
        sys.exit(1)
    agent_name = sys.argv[1]
    session = InteractiveChatSession(agent_name)
    session.run()


if __name__ == "__main__":
    main()
