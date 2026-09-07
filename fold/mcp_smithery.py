"""
Smithery.ai integration and seamless MCP installer for Sheprd.
Provides 1-click Smithery/MCP installation, universal smart command parsing,
automatic synchronization with Claude Desktop & Cursor config files,
and curated 1-click popular tool definitions.
"""

import json
import logging
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .database import Database
from .security import validate_agent_name

logger = logging.getLogger("fold.smithery")

CLAUDE_CONFIG_PATH = Path.home() / ".config/Claude/claude_desktop_config.json"
CURSOR_CONFIG_PATH = Path.home() / ".cursor/mcp.json"
CURSOR_CONFIG_ALT_PATH = Path.home() / ".config/Cursor/mcp.json"
FOLD_MCP_CONFIG_PATH = Path.home() / ".config/fold/mcp.json"
SHEPRD_MCP_CONFIG_PATH = Path.home() / ".config/sheprd/mcp.json"


FEATURED_MCP_CATALOG: List[Dict[str, Any]] = [
    {
        "id": "fetch",
        "name": "web_fetcher",
        "title": "Web Page Fetcher",
        "badge": "Smithery / Fast",
        "icon": "🌐",
        "description": "Scrapes and converts any public webpage or documentation into clean LLM markdown.",
        "command": "uvx",
        "args": ["mcp-server-fetch"],
        "env": {},
        "needs_config": False,
    },
    {
        "id": "sqlite",
        "name": "sqlite_db",
        "title": "SQLite Database",
        "badge": "Official / Zero-Config",
        "icon": "💾",
        "description": "Inspect table schemas and execute read-only queries against local SQLite databases.",
        "command": "uvx",
        "args": ["mcp-server-sqlite"],
        "env": {},
        "needs_config": False,
    },
    {
        "id": "memory",
        "name": "memory_graph",
        "title": "Knowledge Graph Memory",
        "badge": "Anthropic Official",
        "icon": "🧠",
        "description": "Persistent graph memory enabling agents to remember relations, entities, and facts across chats.",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-memory"],
        "env": {},
        "needs_config": False,
    },
    {
        "id": "git",
        "name": "git_inspector",
        "title": "Git Workspace Inspector",
        "badge": "Official Python",
        "icon": "🐙",
        "description": "Inspect git repositories, diffs, branches, staged changes, and commit history.",
        "command": "uvx",
        "args": ["mcp-server-git"],
        "env": {},
        "needs_config": False,
    },
    {
        "id": "time",
        "name": "time_service",
        "title": "Global Timezones",
        "badge": "Official Python",
        "icon": "⏱️",
        "description": "Timezone conversions, local system clocks, and ISO-8601 formatting.",
        "command": "uvx",
        "args": ["mcp-server-time"],
        "env": {},
        "needs_config": False,
    },
    {
        "id": "filesystem",
        "name": "filesystem",
        "title": "Local Filesystem",
        "badge": "Anthropic Official",
        "icon": "📁",
        "description": "Browse, read, and search local files in allowed workspace directories.",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", str(Path.home() / "Projects")],
        "env": {},
        "needs_config": False,
    },
    {
        "id": "brave",
        "name": "brave_search",
        "title": "Brave Web Search",
        "badge": "Official Search",
        "icon": "🦁",
        "description": "Real-time web search powered by the Brave Search API (requires free Brave API Key).",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-brave-search"],
        "env": {"BRAVE_API_KEY": ""},
        "needs_config": True,
        "config_key": "BRAVE_API_KEY",
        "config_prompt": "Enter Brave Search API Key (from brave.com/search/api)",
    },
    {
        "id": "github",
        "name": "github_mcp",
        "title": "GitHub Integration",
        "badge": "Smithery / GitHub",
        "icon": "🐙",
        "description": "Access GitHub repos, pull requests, issues, and code search via Personal Access Token.",
        "command": "npx",
        "args": ["-y", "@smithery/cli", "run", "@smithery-ai/github"],
        "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": ""},
        "needs_config": True,
        "config_key": "GITHUB_PERSONAL_ACCESS_TOKEN",
        "config_prompt": "Enter GitHub Personal Access Token (ghp_...)",
    },
]


def ensure_claude_desktop_config_exists() -> Path:
    """Ensures ~/.config/Claude/claude_desktop_config.json exists so Smithery CLI writes directly into it."""
    claude_dir = Path.home() / ".config/Claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    cfg_file = claude_dir / "claude_desktop_config.json"
    if not cfg_file.exists():
        cfg_file.write_text(json.dumps({"mcpServers": {}}, indent=2), encoding="utf-8")
    return cfg_file


def parse_mcp_install_string(raw: str) -> Dict[str, Any]:
    """
    Intelligently parses ANY input format from Smithery, CLI, URL, or Claude JSON into
    standard MCP registration fields: name, command, args, env, description.
    """
    raw = raw.strip()
    if not raw:
        raise ValueError("Input string cannot be empty.")

    # 0. Check if this matches a featured catalog ID, name, or title
    lowered = raw.lower().strip().replace("-", "_")
    for feat in FEATURED_MCP_CATALOG:
        candidates = {
            feat["id"].lower(),
            feat["id"].lower().replace("-", "_"),
            feat["name"].lower(),
            feat["name"].lower().replace("-", "_"),
            feat["title"].lower(),
        }
        if feat["id"] == "fetch":
            candidates.update({"web_fetch", "webfetch", "web_fetcher", "fetcher", "fetch"})
        elif feat["id"] == "sqlite":
            candidates.update({"sqlite3", "sql", "sqlite_db", "database"})
        elif feat["id"] == "memory":
            candidates.update({"memory_graph", "graph", "knowledge_graph"})
        elif feat["id"] == "git":
            candidates.update({"git_inspector", "vcs"})
        elif feat["id"] == "time":
            candidates.update({"time_service", "clock", "timezone"})

        if lowered in candidates or raw.lower().strip() in candidates:
            return {
                "name": feat["name"],
                "command": feat["command"],
                "args": list(feat["args"]),
                "env": dict(feat.get("env", {})),
                "description": feat["description"],
            }

    # 1. JSON snippet (e.g. from Claude Desktop, Cursor, or Smithery export)
    if (raw.startswith("{") and raw.endswith("}")) or (raw.startswith("[") and raw.endswith("]")):
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                if "mcpServers" in data and isinstance(data["mcpServers"], dict):
                    first_key = list(data["mcpServers"].keys())[0]
                    cfg = data["mcpServers"][first_key]
                    clean_name = re.sub(r"[^a-zA-Z0-9_-]", "_", first_key).strip("_")
                    return {
                        "name": clean_name or "mcp_service",
                        "command": cfg.get("command", "npx"),
                        "args": cfg.get("args", []),
                        "env": cfg.get("env", {}),
                        "description": f"Imported '{first_key}' from JSON snippet",
                    }
                elif "command" in data:
                    raw_name = data.get("name", "mcp_server")
                    clean_name = re.sub(r"[^a-zA-Z0-9_-]", "_", raw_name).strip("_")
                    return {
                        "name": clean_name or "mcp_service",
                        "command": data.get("command", "npx"),
                        "args": data.get("args", []),
                        "env": data.get("env", {}),
                        "description": data.get("description", "Imported from JSON"),
                    }
                elif len(data) == 1:
                    # e.g. {"fetch": {"command": "npx", "args": [...]}}
                    first_key = list(data.keys())[0]
                    cfg = data[first_key]
                    if isinstance(cfg, dict) and "command" in cfg:
                        clean_name = re.sub(r"[^a-zA-Z0-9_-]", "_", first_key).strip("_")
                        return {
                            "name": clean_name or "mcp_service",
                            "command": cfg.get("command", "npx"),
                            "args": cfg.get("args", []),
                            "env": cfg.get("env", {}),
                            "description": f"Imported '{first_key}' from JSON snippet",
                        }
        except Exception:
            pass

    # 2. Smithery URL (e.g. https://smithery.ai/server/@smithery-ai/fetch or smithery.ai/server/sqlite)
    if "smithery.ai/server/" in raw:
        match = re.search(r"smithery\.ai/server/([^?#\s]+)", raw)
        if match:
            pkg = match.group(1).rstrip("/")
            slug = pkg.split("/")[-1].replace("@", "").replace("server-", "")
            clean_name = re.sub(r"[^a-zA-Z0-9_-]", "_", slug).strip("_")
            return {
                "name": clean_name or "smithery_tool",
                "command": "npx",
                "args": ["-y", "@smithery/cli", "run", pkg],
                "env": {},
                "description": f"Smithery server: {pkg}",
            }

    # 3. Raw package name: e.g. @smithery-ai/fetch or @modelcontextprotocol/server-memory
    if raw.startswith("@smithery") or (raw.startswith("@") and "/server-" in raw):
        pkg = raw
        slug = pkg.split("/")[-1].replace("@", "").replace("server-", "")
        clean_name = re.sub(r"[^a-zA-Z0-9_-]", "_", slug).strip("_")
        if raw.startswith("@smithery"):
            args = ["-y", "@smithery/cli", "run", pkg]
        else:
            args = ["-y", pkg]
        return {
            "name": clean_name or "mcp_pkg",
            "command": "npx",
            "args": args,
            "env": {},
            "description": f"Package {pkg}",
        }

    # 4. Command Line parsing
    parts = shlex.split(raw)
    if not parts:
        raise ValueError("Could not parse command input.")

    # Check if this is a smithery CLI command
    if any(p in raw for p in ["@smithery/cli", "smithery"]):
        pkg = None
        for i, p in enumerate(parts):
            if p in ("run", "install", "add") and i + 1 < len(parts):
                cand = parts[i + 1]
                if not cand.startswith("-"):
                    pkg = cand
                    break
        if not pkg:
            for p in parts:
                if "/" in p and not p.startswith("-") and not p.startswith("http"):
                    pkg = p
                    break
        if pkg:
            slug = pkg.split("/")[-1].replace("@", "").replace("server-", "")
            clean_name = re.sub(r"[^a-zA-Z0-9_-]", "_", slug).strip("_")
            return {
                "name": clean_name or "smithery_tool",
                "command": "npx",
                "args": ["-y", "@smithery/cli", "run", pkg],
                "env": {},
                "description": f"Smithery server: {pkg}",
            }

    # Direct command: e.g. "npx -y @modelcontextprotocol/server-memory" or "uvx mcp-server-sqlite"
    cmd = parts[0]
    args = parts[1:]
    # Derive intelligent name
    name_source = parts[-1]
    for p in reversed(parts):
        if not p.startswith("-") and p != cmd:
            name_source = p
            break
    slug = name_source.split("/")[-1].replace("@", "").replace("server-", "").replace("mcp-", "").replace(".py", "")
    clean_name = re.sub(r"[^a-zA-Z0-9_-]", "_", slug).strip("_")
    return {
        "name": clean_name or "custom_mcp",
        "command": cmd,
        "args": args,
        "env": {},
        "description": f"{cmd} {' '.join(args[:3])}".strip(),
    }


def sync_external_client_configs(db: Database) -> List[Dict[str, Any]]:
    """
    Scans Claude Desktop, Cursor, and Sheprd config files for mcpServers.
    Automatically imports any newly added MCP servers into Sheprd.
    """
    candidate_paths = [
        CLAUDE_CONFIG_PATH,
        CURSOR_CONFIG_PATH,
        CURSOR_CONFIG_ALT_PATH,
        FOLD_MCP_CONFIG_PATH,
        SHEPRD_MCP_CONFIG_PATH,
    ]
    imported = []
    existing_servers = {s["name"]: s for s in db.list_mcp_servers()}

    for p in candidate_paths:
        if not p.exists():
            continue
        try:
            content = p.read_text(encoding="utf-8").strip()
            if not content:
                continue
            data = json.loads(content)
            mcp_servers = data.get("mcpServers", {})
            for srv_name, srv_cfg in mcp_servers.items():
                try:
                    clean_name = re.sub(r"[^a-zA-Z0-9_-]", "_", srv_name).strip("_")
                    clean_name = validate_agent_name(clean_name)
                    cmd = srv_cfg.get("command")
                    if not cmd:
                        continue
                    args = srv_cfg.get("args", [])
                    env = srv_cfg.get("env", {})
                    desc = f"Imported from {p.parent.name} ({p.name})"

                    if clean_name not in existing_servers:
                        created = db.create_mcp_server(
                            name=clean_name,
                            command=cmd,
                            args=args,
                            env=env,
                            enabled=True,
                            description=desc,
                        )
                        existing_servers[clean_name] = created
                        imported.append(created)
                        logger.info("Auto-synced MCP server '%s' from %s", clean_name, p)
                except Exception as srv_err:
                    logger.debug("Could not sync server '%s' from %s: %s", srv_name, p, srv_err)
        except Exception as e:
            logger.debug("Could not parse MCP config from %s: %s", p, e)

    return imported


def search_smithery_registry(query: str, limit: int = 8) -> List[Dict[str, Any]]:
    """Searches Smithery's 100K+ MCP registry using npx -y @smithery/cli mcp search."""
    clean_q = query.strip()
    if not clean_q:
        return []

    try:
        proc = subprocess.run(
            ["npx", "-y", "@smithery/cli", "mcp", "search", clean_q],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=25,
            check=False,
        )
        if proc.returncode != 0:
            return []

        results = []
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                item = json.loads(line)
                qname = item.get("qualifiedName") or item.get("name")
                if not qname:
                    continue
                results.append({
                    "name": item.get("name", qname),
                    "qualifiedName": qname,
                    "description": item.get("description", "")[:180],
                    "useCount": item.get("useCount", 0),
                    "connectionUrl": item.get("connectionUrl", ""),
                    "installSnippet": f"npx -y @smithery/cli run {qname}",
                    "command": f"npx -y @smithery/cli run {qname}",
                })
                if len(results) >= limit:
                    break
            except Exception:
                continue
        return results
    except Exception as e:
        logger.debug("Smithery search error: %s", e)
        return []
