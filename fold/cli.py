"""
Command-Line Interface for Fold.
Provides subcommands to start the Web UI, inspect models, manage agent lifecycles,
spawn agents into Herdr, and initiate interactive chat sessions.
"""

import argparse
import os
import sys
from pathlib import Path

from .database import Database
from .herdr_integration import HerdrIntegration
from .inspector import analyze_model_and_recommend
from .interactive_chat import InteractiveChatSession
from .mcp_smithery import parse_mcp_install_string, search_smithery_registry, sync_external_client_configs
from .security import SecurityError, mask_token, validate_model_path
from .server_manager import ServerManager
from .tool_hub import ToolHub
from .web.app import run_web

# Pastel terminal styling (Blue, Sage Green, Slate Grey, Cream)
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_BLUE = "\033[38;5;117m"        # Pastel sky blue
C_GREEN = "\033[38;5;151m"       # Pastel sage green
C_BRIGHT_GREEN = "\033[38;5;157m"# Soft mint
C_SLATE = "\033[38;5;248m"       # Pastel slate grey
C_CREAM = "\033[38;5;230m"       # Origami paper cream
C_CYAN = "\033[38;5;153m"
C_AMBER = "\033[38;5;222m"
C_RED = "\033[38;5;217m"


def cmd_web(args):
    print(f"{C_BLUE}[FOLD]{C_RESET} Launching Fold Web UI at {C_BOLD}http://{args.host}:{args.port}{C_RESET}")
    try:
        run_web(host=args.host, port=args.port)
    except OSError as e:
        if getattr(e, "errno", None) == 98 or "address already in use" in str(e).lower():
            print(f"\n{C_RED}[ERROR]{C_RESET} Port {args.port} is already in use by another process.")
            print(f"{C_AMBER}[ADVICE]{C_RESET} Another Fold instance or service is running on port {args.port}.")
            print(f"        To resolve: `fuser -k {args.port}/tcp` or pass `--port <new_port>` (e.g. `fold web --port 8766`).\n")
            sys.exit(1)
        raise


def cmd_list(args):
    db = Database()
    server_mgr = ServerManager(db)
    agents = db.list_agents()

    if not agents:
        print(f"{C_DIM}No agents currently configured. Deploy an agent with 'fold deploy' or launch the web UI ('fold web').{C_RESET}")
        return

    print(f"\n{C_BOLD}{C_BLUE}FOLD AGENT FLEET{C_RESET}")
    print(f"{C_DIM}{'=' * 85}{C_RESET}")
    header = f"{'NAME':<16} {'STATUS':<12} {'PORT':<8} {'GPU-L':<8} {'TG':<6} {'CALLABLE':<10} {'IDENTITY'}"
    print(f"{C_BOLD}{header}{C_RESET}")
    print(f"{C_DIM}{'-' * 85}{C_RESET}")

    for a in agents:
        is_healthy = a.status == "running" and server_mgr.check_health(a.port)
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
    print(f"{C_DIM}{'=' * 85}{C_RESET}")
    print(f"{C_DIM}Spawn in Herdr: Type '{C_BLUE}<agent_name>{C_RESET}{C_DIM}' in any terminal or run 'fold spawn <name>'.{C_RESET}\n")


def cmd_inspect(args):
    try:
        validated = validate_model_path(args.path)
    except SecurityError as e:
        print(f"{C_RED}[Security Error]{C_RESET} {e}")
        sys.exit(1)

    print(f"{C_BLUE}[FOLD]{C_RESET} Inspecting GGUF binary: {validated.name}...")
    meta, rec = analyze_model_and_recommend(str(validated))

    print(f"\n{C_BOLD}{C_BLUE}MODEL METADATA{C_RESET}")
    print(f"{C_DIM}{'-' * 45}{C_RESET}")
    print(f"  Architecture       : {meta.architecture}")
    print(f"  Quantization       : {meta.quantization}")
    print(f"  Layer Count        : {meta.layer_count}")
    print(f"  Native Context     : {meta.context_length} tokens")
    print(f"  File Size          : {meta.file_size_gb} GB")
    print(f"  Chat Template Kind : {meta.detected_template_kind}")

    print(f"\n{C_BOLD}{C_BLUE}RECOMMENDED EXECUTION PARAMETERS{C_RESET}")
    print(f"{C_DIM}{'-' * 45}{C_RESET}")
    print(f"  Optimal Context    : {rec.context_size}")
    print(f"  GPU Offload Layers : {rec.n_gpu_layers}")
    print(f"  CPU Threads        : {rec.threads}")
    print(f"  Suggested Port     : {rec.recommended_port}")
    if rec.summary_notes:
        print("\n  Hardware Insights:")
        for note in rec.summary_notes:
            print(f"   • {note}")
    print()


def cmd_start(args):
    db = Database()
    server_mgr = ServerManager(db)
    agent_name = args.name.strip()
    agent = db.get_agent_by_name(agent_name)
    if not agent:
        print(f"{C_RED}[ERROR]{C_RESET} Agent '{agent_name}' not found.")
        sys.exit(1)

    print(f"{C_BLUE}[FOLD]{C_RESET} Starting server for agent '{agent_name}'...")
    ok, msg = server_mgr.ensure_agent_running(agent_name)
    if ok:
        print(f"{C_BRIGHT_GREEN}[SUCCESS]{C_RESET} {msg}")
        if agent.telegram_enabled and agent.telegram_bot_token:
            print(f"{C_CYAN}[TELEGRAM]{C_RESET} Telegram bot is enabled for this agent.")
            print(f"  • Launch '{C_BOLD}fold web{C_RESET}' (runs Web UI & Telegram worker automatically)")
            print(f"  • Or run '{C_BOLD}fold telegram run{C_RESET}' for a headless worker")
    else:
        print(f"{C_RED}[FAILED]{C_RESET} {msg}")
        sys.exit(1)


def cmd_activate(args):
    db = Database()
    server_mgr = ServerManager(db)
    agent_name = args.name.strip()
    agent = db.get_agent_by_name(agent_name)
    if not agent:
        print(f"{C_RED}[ERROR]{C_RESET} Agent '{agent_name}' not found.")
        sys.exit(1)

    print(f"{C_BLUE}[FOLD]{C_RESET} Activating agent '{agent_name}' (LRU hot-swap)...")
    ok, msg = server_mgr.ensure_agent_running(agent_name)
    if ok:
        print(f"{C_BRIGHT_GREEN}[SUCCESS]{C_RESET} {msg}")
    else:
        print(f"{C_RED}[FAILED]{C_RESET} {msg}")
        sys.exit(1)


def cmd_telegram(args):
    db = Database()
    from .telegram_service import TelegramServiceManager
    sub = getattr(args, "action", "status")

    if sub == "status" or not sub:
        agents = [a for a in db.list_agents() if a.telegram_enabled]
        if not agents:
            print(f"{C_DIM}No agents currently have Telegram bot enabled.{C_RESET}")
            return
        print(f"\n{C_BOLD}{C_BLUE}TELEGRAM BOT CONFIGURATION{C_RESET}")
        print(f"{C_DIM}{'=' * 65}{C_RESET}")
        for a in agents:
            tok = mask_token(a.telegram_bot_token or "")
            print(f"  Agent: {C_BOLD}{a.name:<12}{C_RESET} Token: {tok}  Server Port: {a.port}")
        print(f"{C_DIM}{'=' * 65}{C_RESET}")
        print(f"Run '{C_BOLD}fold telegram run{C_RESET}' to run workers in the foreground,")
        print(f"or '{C_BOLD}fold web{C_RESET}' to run both Web UI and all Telegram workers.\n")
    elif sub == "run":
        import asyncio
        print(f"{C_BLUE}[TELEGRAM]{C_RESET} Starting Telegram bot workers...")
        async def _run_loop():
            mgr = TelegramServiceManager(db)
            await mgr.sync_all()
            print(f"{C_GREEN}[TELEGRAM]{C_RESET} Polling active. Press Ctrl+C to stop.")
            try:
                while True:
                    await asyncio.sleep(3600)
            except (KeyboardInterrupt, asyncio.CancelledError):
                print(f"\n{C_CYAN}[TELEGRAM]{C_RESET} Shutting down Telegram workers...")
                await mgr.stop_all()
        try:
            asyncio.run(_run_loop())
        except KeyboardInterrupt:
            pass


def cmd_stop(args):
    db = Database()
    server_mgr = ServerManager(db)
    agent_name = args.name.strip()
    ok, msg = server_mgr.stop_agent_server(agent_name)
    print(f"{C_BLUE}[FOLD]{C_RESET} {msg}")


def cmd_stop_all(args):
    db = Database()
    server_mgr = ServerManager(db)
    print(f"{C_BLUE}[FOLD]{C_RESET} Stopping all running agent servers...")
    server_mgr.stop_all()
    print(f"{C_BRIGHT_GREEN}[DONE]{C_RESET} All servers halted.")


def cmd_spawn(args):
    agent_name = args.name.strip()
    db = Database()
    agent = db.get_agent_by_name(agent_name)
    if not agent:
        print(f"{C_RED}[ERROR]{C_RESET} Agent '{agent_name}' not found.")
        sys.exit(1)

    print(f"{C_CYAN}[HERDR]{C_RESET} Spawning agent '{agent_name}' in Herdr...")
    ok, msg, _ = HerdrIntegration.spawn_in_herdr(agent_name, prefer_tab=not args.split)
    if ok:
        print(f"{C_BRIGHT_GREEN}[SUCCESS]{C_RESET} {msg}")
    else:
        print(f"{C_RED}[FAILED]{C_RESET} {msg}")
        print(f"{C_DIM}Fallback: Running interactive session directly in current terminal...{C_RESET}")
        session = InteractiveChatSession(agent_name)
        session.run()


def cmd_chat(args):
    agent_name = args.name.strip()
    session = InteractiveChatSession(agent_name)
    session.run()


def cmd_logs(args):
    db = Database()
    server_mgr = ServerManager(db)
    agent_name = args.name.strip()
    logs = server_mgr.get_agent_logs(agent_name, max_lines=args.lines)
    print(logs)


def cmd_download(args):
    from .downloader import download_model, list_starter_models
    preset = args.model.strip()
    print(f"{C_BLUE}[FOLD]{C_RESET} Starting download for: {C_BOLD}{preset}{C_RESET}...")

    def on_progress(downloaded, total, speed):
        mb_down = downloaded / (1024 * 1024)
        mb_tot = total / (1024 * 1024) if total else 0
        pct = (downloaded / total * 100) if total else 0
        print(f"\r  Progress: {mb_down:.1f}MB / {mb_tot:.1f}MB ({pct:.1f}%) @ {speed:.2f} MB/s", end="", flush=True)

    try:
        path = download_model(preset, progress_hook=on_progress)
        print(f"\n{C_BRIGHT_GREEN}[SUCCESS]{C_RESET} Model downloaded and verified at: {path}")
    except Exception as e:
        print(f"\n{C_RED}[ERROR]{C_RESET} Download failed: {e}")
        sys.exit(1)


def cmd_remove(args):
    db = Database()
    server_mgr = ServerManager(db)
    agent_name = args.name.strip()
    server_mgr.stop_agent_server(agent_name)
    HerdrIntegration.remove_launcher(agent_name)
    deleted = db.delete_agent(agent_name)
    if deleted:
        print(f"{C_BLUE}[FOLD]{C_RESET} Agent '{agent_name}' and Herdr launcher removed.")
    else:
        print(f"{C_RED}[ERROR]{C_RESET} Agent '{agent_name}' not found.")


def cmd_tools(args):
    from .tool_hub import ToolHub
    db = Database()
    hub = ToolHub(db)
    tools = hub.get_available_tools_catalog()

    print(f"\n{C_BOLD}{C_BLUE}FOLD TOOLS & SKILLS CATALOG{C_RESET}")
    print(f"{C_DIM}{'=' * 80}{C_RESET}")
    print(f"{'TYPE':<10} {'NAME':<24} {'SERVER':<16} {'DESCRIPTION'}")
    print(f"{C_DIM}{'-' * 80}{C_RESET}")
    for t in tools:
        type_col = f"{C_CYAN}CORE{C_RESET}" if t["type"] == "core" else f"{C_AMBER}MCP{C_RESET}"
        print(f"{type_col:<19} {C_BOLD}{t['name']:<24}{C_RESET} {t.get('server_name', 'built-in'):<16} {t['description'][:35]}")
    print(f"{C_DIM}{'=' * 80}{C_RESET}\n")


def _probe_and_cache_mcp(db: Database, srv: Dict[str, Any]) -> None:
    print(f"{C_DIM}Probing tools via stdio...{C_RESET}")
    try:
        from .mcp_client import MCPServerInfo
        srv_info = MCPServerInfo(
            id=srv["id"],
            name=srv["name"],
            command=srv["command"],
            args=srv["args"],
            env=srv["env"],
            enabled=srv["enabled"],
            description=srv["description"],
        )
        hub = ToolHub(db)
        conn = hub.mcp_mgr.get_or_start_server(srv_info)
        if conn:
            discovered = conn.refresh_tools(15.0)
            db.set_mcp_cached_tools(srv["name"], discovered)
            if discovered:
                tool_names = ", ".join(t["namespaced_name"] for t in discovered)
                print(f"{C_CYAN}[DISCOVERED]{C_RESET} {len(discovered)} tool(s) ready: {C_BOLD}{tool_names}{C_RESET}")
            else:
                print(f"{C_DIM}Server connected. Tool schemas will be probed during agent runtime.{C_RESET}")
        else:
            print(f"{C_AMBER}[NOTE]{C_RESET} Server registered. Could not start stdio process.")
    except Exception as e:
        print(f"{C_AMBER}[NOTE]{C_RESET} Server registered. Tool probe warning: {e}")


def cmd_mcp(args):
    db = Database()
    sub = getattr(args, "action", "list")

    if sub == "list" or not sub:
        servers = db.list_mcp_servers()
        if not servers:
            print(f"\n{C_DIM}No external MCP servers registered yet.{C_RESET}")
            print(f"Add an MCP server using: {C_BOLD}fold mcp add <name> <command> [args...]{C_RESET}")
            print(f"Example: {C_CYAN}fold mcp add duckduckgo npx -y @modelcontextprotocol/server-duckduckgo{C_RESET}\n")
            return

        print(f"\n{C_BOLD}{C_BLUE}REGISTERED MCP TOOL SERVERS{C_RESET}")
        print(f"{C_DIM}{'=' * 75}{C_RESET}")
        print(f"{'NAME':<16} {'STATUS':<10} {'COMMAND':<20} {'ARGS'}")
        print(f"{C_DIM}{'-' * 75}{C_RESET}")
        for s in servers:
            st = f"{C_BRIGHT_GREEN}ENABLED{C_RESET}" if s["enabled"] else f"{C_DIM}DISABLED{C_RESET}"
            args_str = " ".join(s.get("args", []))[:30]
            print(f"{C_BOLD}{s['name']:<16}{C_RESET} {st:<19} {s['command']:<20} {args_str}")
        print(f"{C_DIM}{'=' * 75}{C_RESET}\n")

    elif sub == "add":
        name = args.name.strip()
        cmd = args.mcp_command.strip()
        cmd_args = args.extra_args or []
        try:
            from .security import validate_agent_name
            name = validate_agent_name(name)
            srv = db.create_mcp_server(name=name, command=cmd, args=cmd_args, description="Added via CLI")
            action = "Registered"
        except SecurityError as e:
            print(f"{C_RED}[ERROR]{C_RESET} Invalid MCP server name: {e}")
            return
        except Exception as e:
            # Check if updating existing
            existing = db.get_mcp_server(name)
            if existing:
                srv = db.update_mcp_server(name=name, command=cmd, args=cmd_args, description="Added via CLI", enabled=True)
                action = "Updated"
            else:
                print(f"{C_RED}[ERROR]{C_RESET} Failed to register MCP server: {e}")
                return

        print(f"{C_BRIGHT_GREEN}[SUCCESS]{C_RESET} {action} MCP server '{C_BOLD}{name}{C_RESET}': {cmd} {' '.join(cmd_args)}")
        _probe_and_cache_mcp(db, srv)

    elif sub == "install":
        target = " ".join(args.target).strip()
        if not target:
            print(f"{C_RED}[ERROR]{C_RESET} Please provide a Smithery URL, package identifier, or command.")
            return

        print(f"{C_DIM}Parsing installation target: {target}...{C_RESET}")
        try:
            parsed = parse_mcp_install_string(target)
            name = parsed["name"]
            cmd = parsed["command"]
            cmd_args = parsed["args"]
            env = parsed["env"]
            desc = parsed["description"]
        except Exception as e:
            print(f"{C_RED}[ERROR]{C_RESET} Failed to parse install string: {e}")
            return

        existing = db.get_mcp_server(name)
        if existing:
            srv = db.update_mcp_server(name=name, command=cmd, args=cmd_args, env=env, description=desc, enabled=True)
            action = "Updated"
        else:
            srv = db.create_mcp_server(name=name, command=cmd, args=cmd_args, env=env, description=desc, enabled=True)
            action = "Installed"

        print(f"{C_BRIGHT_GREEN}[SUCCESS]{C_RESET} {action} MCP server '{C_BOLD}{name}{C_RESET}': {cmd} {' '.join(cmd_args)}")
        _probe_and_cache_mcp(db, srv)

    elif sub == "sync":
        print(f"{C_DIM}Scanning Claude Desktop and Cursor configs for Smithery tools...{C_RESET}")
        new_servers = sync_external_client_configs(db)
        if new_servers:
            names = [s["name"] for s in new_servers]
            print(f"{C_BRIGHT_GREEN}[SUCCESS]{C_RESET} Synced {len(new_servers)} server(s): {', '.join(names)}")
        else:
            print(f"{C_BLUE}[FOLD]{C_RESET} All MCP servers already synchronized.")

    elif sub == "search":
        query = " ".join(args.query).strip()
        if not query:
            print(f"{C_RED}[ERROR]{C_RESET} Please specify a search query.")
            return
        print(f"{C_DIM}Searching Smithery registry for '{query}'...{C_RESET}")
        results = search_smithery_registry(query)
        if not results:
            print(f"{C_DIM}No results found on Smithery registry for '{query}'.{C_RESET}")
            return
        print(f"\n{C_BOLD}{C_CYAN}SMITHERY REGISTRY RESULTS ({len(results)}){C_RESET}")
        print(f"{C_DIM}{'=' * 75}{C_RESET}")
        print(f"{'NAME':<24} {'COMMAND':<25} {'DESCRIPTION'}")
        print(f"{C_DIM}{'-' * 75}{C_RESET}")
        for r in results:
            desc_snip = r.get("description", "")[:24]
            snippet = r.get("installSnippet") or r.get("command") or r.get("qualifiedName", "")
            print(f"{C_BOLD}{r['name'][:23]:<24}{C_RESET} {snippet[:24]:<25} {desc_snip}")
        print(f"{C_DIM}{'=' * 75}{C_RESET}\n")
        print(f"Install any tool with: {C_BOLD}fold mcp install <command-or-url>{C_RESET}\n")

    elif sub == "remove":
        name = args.name.strip()
        deleted = db.delete_mcp_server(name)
        if deleted:
            print(f"{C_BLUE}[FOLD]{C_RESET} Removed MCP server '{name}'.")
        else:
            print(f"{C_RED}[ERROR]{C_RESET} MCP server '{name}' not found.")


def main():
    parser = argparse.ArgumentParser(
        prog="fold",
        description="Fold: llama.cpp Agent Orchestrator & Herdr Workspace Integration.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # web
    p_web = subparsers.add_parser("web", aliases=["ui"], help="Launch Fold Web UI")
    p_web.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    p_web.add_argument("--port", type=int, default=8765, help="Port to listen on (default: 8765)")
    p_web.set_defaults(func=cmd_web)

    # list
    p_list = subparsers.add_parser("list", aliases=["ls"], help="List registered agents")
    p_list.set_defaults(func=cmd_list)

    # inspect
    p_inspect = subparsers.add_parser("inspect", help="Inspect GGUF model and hardware")
    p_inspect.add_argument("path", help="Path to GGUF file")
    p_inspect.set_defaults(func=cmd_inspect)

    # start
    p_start = subparsers.add_parser("start", help="Start agent's llama-server")
    p_start.add_argument("name", help="Agent name")
    p_start.set_defaults(func=cmd_start)

    # activate
    p_act = subparsers.add_parser("activate", help="Activate/hot-swap agent model into memory")
    p_act.add_argument("name", help="Agent name")
    p_act.set_defaults(func=cmd_activate)

    # stop
    p_stop = subparsers.add_parser("stop", help="Stop agent's llama-server")
    p_stop.add_argument("name", help="Agent name")
    p_stop.set_defaults(func=cmd_stop)

    # stop-all
    p_stop_all = subparsers.add_parser("stop-all", help="Stop all running agent servers")
    p_stop_all.set_defaults(func=cmd_stop_all)

    # spawn
    p_spawn = subparsers.add_parser("spawn", help="Spawn agent in Herdr terminal workspace")
    p_spawn.add_argument("name", help="Agent name")
    p_spawn.add_argument("--split", action="store_true", help="Split pane instead of new tab")
    p_spawn.set_defaults(func=cmd_spawn)

    # chat
    p_chat = subparsers.add_parser("chat", aliases=["run"], help="Interactive terminal chat session")
    p_chat.add_argument("name", help="Agent name")
    p_chat.set_defaults(func=cmd_chat)

    # logs
    p_logs = subparsers.add_parser("logs", help="View agent server logs")
    p_logs.add_argument("name", help="Agent name")
    p_logs.add_argument("-n", "--lines", type=int, default=100, help="Line count")
    p_logs.set_defaults(func=cmd_logs)

    # download
    p_dl = subparsers.add_parser("download", help="Download starter model or Hugging Face GGUF")
    p_dl.add_argument("model", help="Starter key (qwen2.5-0.5b, smollm2-135m, llama-3.2-1b) or GGUF URL")
    p_dl.set_defaults(func=cmd_download)

    # telegram
    p_tg = subparsers.add_parser("telegram", aliases=["tg"], help="Manage Telegram bot workers")
    p_tg.add_argument("action", nargs="?", default="status", choices=["status", "run"], help="Action (status, run)")
    p_tg.set_defaults(func=cmd_telegram)

    # tools
    p_tools = subparsers.add_parser("tools", help="List available Core and MCP tools")
    p_tools.set_defaults(func=cmd_tools)

    # mcp
    p_mcp = subparsers.add_parser("mcp", help="Manage external MCP servers")
    p_mcp_sub = p_mcp.add_subparsers(dest="action")
    p_mcp_list = p_mcp_sub.add_parser("list", help="List registered MCP servers")
    p_mcp_list.set_defaults(func=cmd_mcp)
    p_mcp_install = p_mcp_sub.add_parser("install", help="1-click install from Smithery URL, package identifier, or command")
    p_mcp_install.add_argument("target", nargs="+", help="Smithery URL, @package, command, or featured tool ID")
    p_mcp_install.set_defaults(func=cmd_mcp)
    p_mcp_sync = p_mcp_sub.add_parser("sync", help="Synchronize MCP servers from Claude Desktop and Cursor configs")
    p_mcp_sync.set_defaults(func=cmd_mcp)
    p_mcp_search = p_mcp_sub.add_parser("search", help="Search the Smithery tool registry")
    p_mcp_search.add_argument("query", nargs="+", help="Keyword to search (e.g. fetch, git, search)")
    p_mcp_search.set_defaults(func=cmd_mcp)
    p_mcp_add = p_mcp_sub.add_parser("add", help="Register an MCP server manually")
    p_mcp_add.add_argument("name", help="Server identifier name")
    p_mcp_add.add_argument("mcp_command", help="Command binary, e.g. npx, python3, uvx")
    p_mcp_add.add_argument("extra_args", nargs=argparse.REMAINDER, help="Arguments passed to MCP command")
    p_mcp_add.set_defaults(func=cmd_mcp)
    p_mcp_rm = p_mcp_sub.add_parser("remove", aliases=["rm"], help="Remove an MCP server")
    p_mcp_rm.add_argument("name", help="Server name to remove")
    p_mcp_rm.set_defaults(func=cmd_mcp)
    p_mcp.set_defaults(func=cmd_mcp)

    # remove
    p_rm = subparsers.add_parser("remove", aliases=["rm", "delete"], help="Delete agent and launcher")
    p_rm.add_argument("name", help="Agent name")
    p_rm.set_defaults(func=cmd_remove)

    args = parser.parse_args()

    if not args.command:
        # Default action when run with no arguments: list or show help
        cmd_list(args)
        print("Run 'fold web' to open the Web UI or 'fold --help' for CLI commands.")
        return

    args.func(args)


if __name__ == "__main__":
    main()
