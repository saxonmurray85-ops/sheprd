"""
Interactive Fold CLI Shell.
Provides a rich interactive REPL console for managing agents, inspecting models,
initiating chat sessions, and monitoring the Fold fleet directly from the terminal.
"""

import os
import readline
import shlex
import sys
from pathlib import Path
from typing import List, Optional

from .database import Database
from .inspector import analyze_model_and_recommend
from .interactive_chat import InteractiveChatSession
from .mcp_smithery import (
    parse_mcp_install_string,
    search_smithery_registry,
    sync_external_client_configs,
)
from .security import SecurityError, validate_model_path
from .server_manager import ServerManager
from .tool_hub import ToolHub

# Pastel ANSI styling
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_BLUE = "\033[38;5;117m"         # Soft pastel sky blue
C_GREEN = "\033[38;5;151m"        # Soft pastel sage green
C_BRIGHT_GREEN = "\033[38;5;157m" # Radiant mint
C_SLATE = "\033[38;5;248m"        # Pastel slate grey
C_CREAM = "\033[38;5;230m"        # Origami paper cream
C_CYAN = "\033[38;5;153m"
C_AMBER = "\033[38;5;222m"
C_RED = "\033[38;5;217m"


CLI_BANNER = f"""\n{C_BLUE}{C_BOLD}   _____   ___   _     ___  
  |  ___| / _ \\ | |   |   \\ 
  | |_   | | | || |   | |) |
  |_|     \\___/ |_|___|___/ 
{C_RESET}{C_GREEN}  🐑 Fold Interactive CLI Console // Origami Fleet{C_RESET}
{C_DIM}  Type 'help' for commands, 'chat <agent>' to speak with an agent, or 'exit' to quit.{C_RESET}\n"""


class FoldCLI:
    """Interactive command-line shell for Fold."""

    COMMANDS = [
        "list", "ls",
        "status",
        "chat", "run",
        "start",
        "stop",
        "stop-all",
        "activate",
        "inspect",
        "logs",
        "spawn",
        "tools",
        "mcp",
        "web",
        "clear",
        "help",
        "exit", "quit", "q",
    ]

    def __init__(self, db: Optional[Database] = None):
        self.db = db or Database()
        self.server_mgr = ServerManager(self.db)
        self.tool_hub = ToolHub(self.db)
        self._history_file = Path.home() / ".config" / "fold" / "cli_history"
        self._setup_readline()

    def _setup_readline(self):
        try:
            self._history_file.parent.mkdir(parents=True, exist_ok=True)
            if self._history_file.exists():
                readline.read_history_file(str(self._history_file))
        except Exception:
            pass

        readline.set_completer(self._completer)
        readline.parse_and_bind("tab: complete")

    def _save_history(self):
        try:
            readline.write_history_file(str(self._history_file))
        except Exception:
            pass

    def _completer(self, text: str, state: int) -> Optional[str]:
        line = readline.get_line_buffer()
        tokens = line.split()

        if not tokens or (len(tokens) == 1 and not line.endswith(" ")):
            matches = [cmd for cmd in self.COMMANDS if cmd.startswith(text)]
            return matches[state] if state < len(matches) else None

        cmd = tokens[0].lower()
        if cmd in ("chat", "run", "start", "stop", "activate", "logs", "spawn", "remove", "rm"):
            agent_names = [a.name for a in self.db.list_agents()]
            matches = [name for name in agent_names if name.startswith(text)]
            return matches[state] if state < len(matches) else None

        if cmd == "mcp":
            mcp_subs = ["list", "search", "install", "sync", "remove"]
            matches = [sub for sub in mcp_subs if sub.startswith(text)]
            return matches[state] if state < len(matches) else None

        return None

    def print_help(self):
        print(f"\n{C_BOLD}{C_BLUE}FOLD CLI COMMAND REFERENCE{C_RESET}")
        print(f"{C_DIM}{'=' * 65}{C_RESET}")
        commands = [
            ("list, ls", "List all agents in the fleet and status"),
            ("status", "Show fleet overview & running llama-server processes"),
            ("chat <agent>", "Open real-time interactive terminal chat session"),
            ("start <agent>", "Start background llama-server for an agent"),
            ("stop <agent>", "Gracefully stop an agent's llama-server"),
            ("stop-all", "Stop all running agent servers"),
            ("activate <agent>", "Hot-swap agent model into active memory"),
            ("inspect <path>", "Inspect GGUF binary & hardware recommendations"),
            ("logs <agent> [n]", "View recent server process logs (default 40 lines)"),
            ("spawn <agent>", "Spawn agent in Herdr terminal workspace"),
            ("tools", "List available Core Tools & external MCP servers"),
            ("mcp list", "List registered external MCP servers"),
            ("mcp search <query>", "Search the Smithery MCP tool registry"),
            ("mcp install <target>", "1-click install MCP server from Smithery/package"),
            ("mcp sync", "Synchronize MCP tools from Claude Desktop & Cursor"),
            ("mcp remove <name>", "Remove an external MCP server"),
            ("web", "Launch Fold Web UI (http://127.0.0.1:8765)"),
            ("clear", "Clear terminal screen"),
            ("help", "Display this command guide"),
            ("exit, quit", "Exit Fold interactive CLI"),
        ]
        for cmd, desc in commands:
            print(f"  {C_GREEN}{cmd:<20}{C_RESET} {C_SLATE}{desc}{C_RESET}")
        print(f"{C_DIM}{'=' * 65}{C_RESET}\n")

    def run_loop(self):
        print(CLI_BANNER)
        self.cmd_list([])

        while True:
            try:
                prompt = f"{C_BOLD}{C_BLUE}fold{C_RESET}{C_GREEN} 🐑 > {C_RESET}"
                line = input(prompt).strip()
                if not line:
                    continue

                parts = shlex.split(line)
                cmd = parts[0].lower()
                args = parts[1:]

                if cmd in ("exit", "quit", "q"):
                    print(f"{C_BLUE}[FOLD]{C_RESET} Exiting CLI. Have a good one!\n")
                    break

                self.execute_command(cmd, args)

            except KeyboardInterrupt:
                print(f"\n{C_DIM}(Use 'exit' or Ctrl+D to quit){C_RESET}")
            except EOFError:
                print(f"\n{C_BLUE}[FOLD]{C_RESET} Exiting CLI.\n")
                break
            except Exception as e:
                print(f"{C_RED}[Error]{C_RESET} {e}")

        self._save_history()

    def execute_command(self, cmd: str, args: List[str]):
        if cmd in ("help", "?"):
            self.print_help()
        elif cmd in ("list", "ls"):
            self.cmd_list(args)
        elif cmd == "status":
            self.cmd_status()
        elif cmd in ("chat", "run"):
            self.cmd_chat(args)
        elif cmd == "start":
            self.cmd_start(args)
        elif cmd == "stop":
            self.cmd_stop(args)
        elif cmd == "stop-all":
            self.cmd_stop_all()
        elif cmd == "activate":
            self.cmd_activate(args)
        elif cmd == "inspect":
            self.cmd_inspect(args)
        elif cmd == "logs":
            self.cmd_logs(args)
        elif cmd == "spawn":
            self.cmd_spawn(args)
        elif cmd == "tools":
            self.cmd_tools()
        elif cmd == "mcp":
            self.cmd_mcp(args)
        elif cmd == "web":
            self.cmd_web(args)
        elif cmd == "clear":
            os.system("clear")
        else:
            print(f"{C_RED}Unknown command:{C_RESET} '{cmd}'. Type {C_BOLD}'help'{C_RESET} for available commands.")

    def cmd_list(self, args: List[str]):
        agents = self.db.list_agents()
        if not agents:
            print(f"{C_DIM}No agents currently configured. Deploy an agent with 'fold deploy' or launch the web UI ('fold web').{C_RESET}")
            return

        print(f"\n{C_BOLD}{C_BLUE}FOLD AGENT FLEET{C_RESET}")
        print(f"{C_DIM}{'=' * 85}{C_RESET}")
        header = f"{'NAME':<16} {'STATUS':<12} {'PORT':<8} {'GPU-L':<8} {'TG':<6} {'CALLABLE':<10} {'IDENTITY'}"
        print(f"{C_BOLD}{header}{C_RESET}")
        print(f"{C_DIM}{'-' * 85}{C_RESET}")

        for a in agents:
            is_healthy = a.status == "running" and self.server_mgr.check_health(a.port)
            status_col = f"{C_BRIGHT_GREEN}ONLINE{C_RESET}" if is_healthy else f"{C_DIM}STOPPED{C_RESET}"
            tg_col = f"{C_CYAN}YES{C_RESET}" if a.telegram_enabled else f"{C_DIM}NO{C_RESET}"
            call_col = f"{C_GREEN}YES{C_RESET}" if a.callable_by_agents else f"{C_DIM}NO{C_RESET}"

            row = (
                f"{a.name:<16} "
                f"{status_col:<21} "
                f"{a.port:<8} "
                f"{a.n_gpu_layers:<8} "
                f"{tg_col:<15} "
                f"{call_col:<19} "
                f"{a.identity[:25]}"
            )
            print(row)
        print(f"{C_DIM}{'=' * 85}{C_RESET}\n")

    def cmd_status(self):
        agents = self.db.list_agents()
        running = [a for a in agents if a.status == "running" and self.server_mgr.check_health(a.port)]
        stopped = [a for a in agents if a not in running]

        print(f"\n{C_BOLD}{C_BLUE}FOLD FLEET STATUS{C_RESET}")
        print(f"{C_DIM}{'-' * 45}{C_RESET}")
        print(f"  Total Agents Registered : {len(agents)}")
        print(f"  Online / Active         : {C_BRIGHT_GREEN}{len(running)}{C_RESET}")
        print(f"  Stopped                 : {C_DIM}{len(stopped)}{C_RESET}")

        if running:
            print(f"\n{C_BOLD}Active Servers:{C_RESET}")
            for a in running:
                print(f"  • {C_GREEN}{a.name:<16}{C_RESET} Port: {a.port}  PID: {a.pid}  Model: {Path(a.model_path).name}")
        print()

    def cmd_chat(self, args: List[str]):
        if not args:
            print(f"{C_RED}[Error]{C_RESET} Please specify an agent name: {C_BOLD}chat <agent_name>{C_RESET}")
            return
        agent_name = args[0]
        agent = self.db.get_agent_by_name(agent_name)
        if not agent:
            print(f"{C_RED}[Error]{C_RESET} Agent '{agent_name}' not found.")
            return

        session = InteractiveChatSession(agent_name, db=self.db)
        session.start()

    def cmd_start(self, args: List[str]):
        if not args:
            print(f"{C_RED}[Error]{C_RESET} Usage: {C_BOLD}start <agent_name>{C_RESET}")
            return
        agent_name = args[0]
        print(f"{C_BLUE}[FOLD]{C_RESET} Starting agent '{agent_name}' (loading tensors into memory/GPU)...\n")
        success, msg = self.server_mgr.start_agent_server(agent_name)
        if success:
            print(f"{C_BRIGHT_GREEN}[SUCCESS]{C_RESET} {msg}")
        else:
            print(f"{C_RED}[ERROR]{C_RESET} {msg}")

    def cmd_stop(self, args: List[str]):
        if not args:
            print(f"{C_RED}[Error]{C_RESET} Usage: {C_BOLD}stop <agent_name>{C_RESET}")
            return
        agent_name = args[0]
        success, msg = self.server_mgr.stop_agent_server(agent_name)
        if success:
            print(f"{C_BLUE}[FOLD]{C_RESET} {msg}")
        else:
            print(f"{C_RED}[ERROR]{C_RESET} {msg}")

    def cmd_stop_all(self):
        stopped = self.server_mgr.stop_all_servers()
        if stopped:
            print(f"{C_BLUE}[FOLD]{C_RESET} Stopped {len(stopped)} server(s): {', '.join(stopped)}")
        else:
            print(f"{C_DIM}No running agent servers were active.{C_RESET}")

    def cmd_activate(self, args: List[str]):
        if not args:
            print(f"{C_RED}[Error]{C_RESET} Usage: {C_BOLD}activate <agent_name>{C_RESET}")
            return
        agent_name = args[0]
        print(f"{C_BLUE}[FOLD]{C_RESET} Hot-swapping agent '{agent_name}' via LRU engine...")
        success, msg = self.server_mgr.activate_agent(agent_name)
        if success:
            print(f"{C_BRIGHT_GREEN}[SUCCESS]{C_RESET} {msg}")
        else:
            print(f"{C_RED}[ERROR]{C_RESET} {msg}")

    def cmd_inspect(self, args: List[str]):
        if not args:
            print(f"{C_RED}[Error]{C_RESET} Usage: {C_BOLD}inspect <model_path>{C_RESET}")
            return
        try:
            validated = validate_model_path(args[0])
        except SecurityError as e:
            print(f"{C_RED}[Security Error]{C_RESET} {e}")
            return

        print(f"{C_BLUE}[FOLD]{C_RESET} Inspecting GGUF binary: {validated.name}...")
        meta, rec = analyze_model_and_recommend(str(validated))

        print(f"\n{C_BOLD}{C_BLUE}MODEL METADATA{C_RESET}")
        print(f"{C_DIM}{'-' * 45}{C_RESET}")
        print(f"  Architecture       : {meta.architecture}")
        print(f"  Quantization       : {meta.quantization}")
        print(f"  Layer Count        : {meta.layer_count}")
        print(f"  Native Context     : {meta.context_length} tokens")
        print(f"  File Size          : {meta.file_size_gb} GB")
        print(f"  Chat Template Kind : {meta.chat_template_kind}")

        print(f"\n{C_BOLD}{C_GREEN}RECOMMENDED EXECUTION PARAMETERS{C_RESET}")
        print(f"{C_DIM}{'-' * 45}{C_RESET}")
        print(f"  Optimal Context    : {rec.context_size}")
        print(f"  GPU Offload Layers : {rec.n_gpu_layers}")
        print(f"  CPU Threads        : {rec.threads}")
        print(f"  Suggested Port     : {rec.suggested_port}")
        print(f"  Batch Size         : {rec.batch_size}")
        if rec.hardware_notes:
            print(f"\n{C_BOLD}  Hardware Insights:{C_RESET}")
            for note in rec.hardware_notes:
                print(f"   • {C_DIM}{note}{C_RESET}")
        print()

    def cmd_logs(self, args: List[str]):
        if not args:
            print(f"{C_RED}[Error]{C_RESET} Usage: {C_BOLD}logs <agent_name> [num_lines]{C_RESET}")
            return
        agent_name = args[0]
        lines = 40
        if len(args) > 1:
            try:
                lines = max(1, min(1000, int(args[1])))
            except ValueError:
                pass

        log_path = Path.home() / ".local" / "share" / "fold" / "logs" / f"{agent_name}.log"
        if not log_path.exists():
            print(f"{C_DIM}No logs available for '{agent_name}' at {log_path}.{C_RESET}")
            return

        print(f"\n{C_BOLD}Logs for {agent_name} (last {lines} lines):{C_RESET}")
        print(f"{C_DIM}{'=' * 65}{C_RESET}")
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
            for line in all_lines[-lines:]:
                print(line.rstrip())
        print(f"{C_DIM}{'=' * 65}{C_RESET}\n")

    def cmd_spawn(self, args: List[str]):
        if not args:
            print(f"{C_RED}[Error]{C_RESET} Usage: {C_BOLD}spawn <agent_name>{C_RESET}")
            return
        from .herdr_integration import HerdrIntegration
        herdr = HerdrIntegration(self.db)
        success, msg = herdr.spawn_agent_pane(args[0])
        if success:
            print(f"{C_BRIGHT_GREEN}[SUCCESS]{C_RESET} {msg}")
        else:
            print(f"{C_AMBER}[HERDR]{C_RESET} {msg}")

    def cmd_tools(self):
        core_tools = self.tool_hub.get_all_tools()
        mcp_servers = self.db.list_mcp_servers()

        print(f"\n{C_BOLD}{C_BLUE}AVAILABLE TOOLS & MCP SERVERS{C_RESET}")
        print(f"{C_DIM}{'=' * 65}{C_RESET}")
        print(f"{C_BOLD}Core Tool Pack ({len(core_tools)} tools):{C_RESET}")
        for t in core_tools:
            print(f"  • {C_GREEN}{t.name:<24}{C_RESET} {C_SLATE}{t.description}{C_RESET}")

        print(f"\n{C_BOLD}External MCP Servers ({len(mcp_servers)} registered):{C_RESET}")
        if not mcp_servers:
            print(f"  {C_DIM}No external MCP servers configured. Install via 'mcp search' or 'mcp install'.{C_RESET}")
        else:
            for s in mcp_servers:
                status = f"{C_BRIGHT_GREEN}ENABLED{C_RESET}" if s.enabled else f"{C_DIM}DISABLED{C_RESET}"
                print(f"  • {C_CYAN}{s.name:<18}{C_RESET} [{status}] {s.command} {' '.join(s.args)}")
        print(f"{C_DIM}{'=' * 65}{C_RESET}\n")

    def cmd_mcp(self, args: List[str]):
        if not args or args[0] == "list":
            servers = self.db.list_mcp_servers()
            print(f"\n{C_BOLD}{C_BLUE}REGISTERED MCP SERVERS ({len(servers)}){C_RESET}")
            print(f"{C_DIM}{'-' * 60}{C_RESET}")
            for s in servers:
                status = f"{C_BRIGHT_GREEN}ENABLED{C_RESET}" if s.enabled else f"{C_DIM}DISABLED{C_RESET}"
                print(f"  {C_BOLD}{s.name:<16}{C_RESET} [{status}] {s.command} {' '.join(s.args)}")
            print()
        elif args[0] == "search":
            query = " ".join(args[1:]).strip()
            if not query:
                print(f"{C_RED}[Error]{C_RESET} Usage: {C_BOLD}mcp search <query>{C_RESET}")
                return
            print(f"{C_DIM}Searching Smithery registry for '{query}'...{C_RESET}")
            results = search_smithery_registry(query)
            if not results:
                print(f"{C_DIM}No results found for '{query}'.{C_RESET}")
                return
            print(f"\n{C_BOLD}{C_CYAN}SMITHERY REGISTRY RESULTS ({len(results)}){C_RESET}")
            print(f"{C_DIM}{'-' * 65}{C_RESET}")
            for r in results:
                snippet = r.get("installSnippet") or r.get("qualifiedName", "")
                print(f"  {C_BOLD}{r['name']:<20}{C_RESET} {C_SLATE}{snippet:<30}{C_RESET}")
            print()
        elif args[0] == "install":
            if len(args) < 2:
                print(f"{C_RED}[Error]{C_RESET} Usage: {C_BOLD}mcp install <smithery_url_or_package>{C_RESET}")
                return
            raw_target = " ".join(args[1:]).strip()
            try:
                parsed = parse_mcp_install_string(raw_target)
                name = parsed["name"]
                cmd = parsed["command"]
                cmd_args = parsed["args"]
                env = parsed["env"]
                desc = parsed["description"]
            except Exception as e:
                print(f"{C_RED}[Error]{C_RESET} Could not parse install string: {e}")
                return

            existing = self.db.get_mcp_server(name)
            if existing:
                srv = self.db.update_mcp_server(name=name, command=cmd, args=cmd_args, env=env, description=desc, enabled=True)
                action = "Updated"
            else:
                srv = self.db.create_mcp_server(name=name, command=cmd, args=cmd_args, env=env, description=desc, enabled=True)
                action = "Installed"
            print(f"{C_BRIGHT_GREEN}[SUCCESS]{C_RESET} {action} MCP server '{name}': {cmd} {' '.join(cmd_args)}")
        elif args[0] == "sync":
            print(f"{C_DIM}Scanning Claude Desktop and Cursor configs for MCP tools...{C_RESET}")
            new_servers = sync_external_client_configs(self.db)
            if new_servers:
                names = [s["name"] for s in new_servers]
                print(f"{C_BRIGHT_GREEN}[SUCCESS]{C_RESET} Synced {len(new_servers)} server(s): {', '.join(names)}")
            else:
                print(f"{C_BLUE}[FOLD]{C_RESET} All MCP servers already synchronized.")
        elif args[0] in ("remove", "rm"):
            if len(args) < 2:
                print(f"{C_RED}[Error]{C_RESET} Usage: {C_BOLD}mcp remove <name>{C_RESET}")
                return
            name = args[1].strip()
            if self.db.delete_mcp_server(name):
                print(f"{C_BLUE}[FOLD]{C_RESET} Removed MCP server '{name}'.")
            else:
                print(f"{C_RED}[Error]{C_RESET} MCP server '{name}' not found.")
        else:
            print(f"{C_RED}Unknown mcp action:{C_RESET} '{args[0]}'. Choices: list, search, install, sync, remove")

    def cmd_web(self, args: List[str]):
        port = 8765
        host = "127.0.0.1"
        if args:
            try:
                port = int(args[0])
            except ValueError:
                pass
        print(f"\n{C_BLUE}[FOLD]{C_RESET} Launching Fold Web UI at http://{host}:{port}...")
        print(f"{C_DIM}Press Ctrl+C to stop the Web UI and return to Fold CLI.{C_RESET}\n")
        from .web.app import run_web
        try:
            run_web(host=host, port=port)
        except KeyboardInterrupt:
            print(f"\n{C_BLUE}[FOLD]{C_RESET} Web UI stopped. Returning to Fold CLI.\n")
