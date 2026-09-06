"""
Web application backend for Sheprd.
Provides RESTful APIs, Server-Sent Events (SSE), and static file serving
using Python's native aiohttp async server.
"""

import asyncio
import json
import logging
import os
import secrets
from pathlib import Path
from typing import Any, Dict, Optional

import aiohttp
from aiohttp import web

from ..database import Database
from ..herdr_integration import HerdrIntegration
from ..inspector import HardwareInspector, analyze_model_and_recommend
from ..multi_agent import MultiAgentRouter
from ..security import (
    SecurityError,
    get_or_create_api_token,
    mask_token,
    validate_agent_name,
    validate_context_size,
    validate_gpu_layers,
    validate_groups,
    validate_model_path,
    validate_port,
)
from ..mcp_smithery import (
    FEATURED_MCP_CATALOG,
    parse_mcp_install_string,
    search_smithery_registry,
    sync_external_client_configs,
)
from ..prompts import build_agent_system_prompt
from ..server_manager import ServerManager
from ..telegram_service import TelegramServiceManager
from ..tool_hub import ToolHub


logger = logging.getLogger("sheprd.web")
WEB_DIR = Path(__file__).parent


class SheprdWebApp:
    def __init__(self, db: Optional[Database] = None, host: str = "127.0.0.1", port: int = 8765):
        self.host = host
        self.port = port
        self.db = db or Database()
        self.server_mgr = ServerManager(self.db)
        self.telegram_mgr = TelegramServiceManager(self.db, server_mgr=self.server_mgr)
        self.tool_hub = ToolHub(self.db)
        self.router = MultiAgentRouter(self.db)
        self.csrf_token = get_or_create_api_token()
        self.app = web.Application(middlewares=[self.csrf_middleware])
        self.app.on_startup.append(self._on_startup)
        self.app.on_cleanup.append(self._on_cleanup)
        self._setup_routes()

    async def _on_startup(self, app: web.Application) -> None:
        logger.info("Sheprd web app starting: syncing all active Telegram bots...")
        await self.telegram_mgr.sync_all()
        # Auto-sync any existing external client configs (Claude/Cursor/Smithery)
        await asyncio.to_thread(sync_external_client_configs, self.db)

    async def _on_cleanup(self, app: web.Application) -> None:
        logger.info("Sheprd web app shutting down: stopping Telegram bots and MCP servers...")
        await self.telegram_mgr.stop_all()
        self.tool_hub.mcp_mgr.stop_all()

    @web.middleware
    async def csrf_middleware(self, request: web.Request, handler):
        # S2 & N5: Enforce CSRF & Origin validation on mutating requests
        if request.method in ("POST", "PUT", "DELETE", "PATCH"):
            # 1. Verify Origin/Referer if present
            origin = request.headers.get("Origin") or request.headers.get("Referer") or ""
            if origin:
                allowed_prefixes = (
                    f"http://127.0.0.1:{self.port}",
                    f"http://localhost:{self.port}",
                    "http://127.0.0.1",
                    "http://localhost",
                )
                if not any(origin.startswith(p) for p in allowed_prefixes):
                    return web.json_response({"error": "Forbidden: Cross-origin request rejected (S2)."}, status=403)

            # 2. Check Sec-Fetch-Site
            fetch_site = request.headers.get("Sec-Fetch-Site")
            if fetch_site == "cross-site":
                return web.json_response({"error": "Forbidden: Cross-site request rejected (S2)."}, status=403)

            # 3. Unconditionally enforce X-Sheprd-Token header (N5)
            token = request.headers.get("X-Sheprd-Token")
            if not token or not secrets.compare_digest(token, self.csrf_token):
                return web.json_response({"error": "Forbidden: Missing or invalid X-Sheprd-Token (S2/N5)."}, status=403)

        return await handler(request)

    def _setup_routes(self):
        # UI views
        self.app.router.add_get("/", self.handle_index)
        self.app.router.add_static("/static/", path=str(WEB_DIR / "static"), name="static")

        # System & Inspection APIs
        self.app.router.add_get("/api/status", self.handle_system_status)
        self.app.router.add_post("/api/inspect", self.handle_inspect_model)
        self.app.router.add_get("/api/events", self.handle_sse_events)

        # Agent CRUD APIs
        self.app.router.add_get("/api/agents", self.handle_list_agents)
        self.app.router.add_post("/api/agents", self.handle_create_agent)
        self.app.router.add_get("/api/agents/{name}", self.handle_get_agent)
        self.app.router.add_put("/api/agents/{name}", self.handle_update_agent)
        self.app.router.add_delete("/api/agents/{name}", self.handle_delete_agent)

        # Agent Lifecycle & Herdr APIs
        self.app.router.add_post("/api/agents/{name}/start", self.handle_start_agent)
        self.app.router.add_post("/api/agents/{name}/stop", self.handle_stop_agent)
        self.app.router.add_post("/api/agents/{name}/activate", self.handle_activate_agent)
        self.app.router.add_post("/api/agents/{name}/spawn-herdr", self.handle_spawn_herdr)
        self.app.router.add_get("/api/agents/{name}/logs", self.handle_get_logs)
        self.app.router.add_get("/api/agents/{name}/history", self.handle_get_history)
        self.app.router.add_post("/api/agents/{name}/chat", self.handle_chat_agent)

        # Multi-agent & Group APIs
        self.app.router.add_get("/api/groups", self.handle_list_groups)
        self.app.router.add_post("/api/groups/{group_name}/ask", self.handle_group_ask)

        # Tools & MCP APIs
        self.app.router.add_get("/api/tools", self.handle_list_tools)
        self.app.router.add_get("/api/mcp/servers", self.handle_list_mcp_servers)
        self.app.router.add_post("/api/mcp/servers", self.handle_create_mcp_server)
        self.app.router.add_put("/api/mcp/servers/{name}", self.handle_update_mcp_server)
        self.app.router.add_delete("/api/mcp/servers/{name}", self.handle_delete_mcp_server)
        self.app.router.add_post("/api/mcp/servers/{name}/test", self.handle_test_mcp_server)
        self.app.router.add_get("/api/mcp/featured", self.handle_featured_mcp)
        self.app.router.add_post("/api/mcp/quick-install", self.handle_quick_install_mcp)
        self.app.router.add_post("/api/mcp/sync-external", self.handle_sync_external_mcp)
        self.app.router.add_get("/api/mcp/search", self.handle_search_mcp)

    async def handle_index(self, request: web.Request) -> web.Response:
        index_file = WEB_DIR / "templates/index.html"
        if not index_file.exists():
            return web.Response(text="Sheprd UI template missing.", status=500)
        html = index_file.read_text(encoding="utf-8")
        html = html.replace('<!-- CSRF_TOKEN -->', f'<meta name="csrf-token" content="{self.csrf_token}">')
        return web.Response(text=html, content_type="text/html")

    async def handle_system_status(self, request: web.Request) -> web.Response:
        hw = HardwareInspector.detect()
        herdr_running = HerdrIntegration.is_herdr_running()
        agents = self.db.list_agents()
        running_count = sum(1 for a in agents if a.status == "running")
        tg_active_count = sum(1 for name, w in self.telegram_mgr.workers.items() if w.running)

        return web.json_response({
            "status": "ok",
            "cpu_model": hw.cpu_model,
            "logical_cores": hw.logical_cores,
            "total_ram_gb": round(hw.total_ram_bytes / (1024 ** 3), 1),
            "available_ram_gb": round(hw.available_ram_bytes / (1024 ** 3), 1),
            "gpu_devices": hw.gpu_devices,
            "vulkan_supported": hw.vulkan_supported,
            "herdr_available": herdr_running,
            "total_agents": len(agents),
            "running_agents": running_count,
            "active_telegram_bots": tg_active_count,
        })

    async def handle_inspect_model(self, request: web.Request) -> web.Response:
        try:
            data = await request.json()
            model_path = data.get("model_path", "").strip()
            if not model_path:
                return web.json_response({"error": "model_path is required."}, status=400)

            # Security validation
            validated_path = validate_model_path(model_path)
            target_port = self.server_mgr.find_free_port()

            # Execute inspection & calculate optimal execution params (non-blocking B1)
            meta, rec = await asyncio.to_thread(analyze_model_and_recommend, str(validated_path), target_port)

            return web.json_response({
                "ok": True,
                "model_metadata": {
                    "file_path": meta.file_path,
                    "file_name": meta.file_name,
                    "file_size_gb": meta.file_size_gb,
                    "architecture": meta.architecture,
                    "model_name": meta.model_name,
                    "quantization": meta.quantization,
                    "layer_count": meta.layer_count,
                    "context_length": meta.context_length,
                    "detected_template_kind": meta.detected_template_kind,
                    "estimated_vram_mb": meta.estimated_vram_mb,
                },
                "recommended_config": {
                    "context_size": rec.context_size,
                    "n_gpu_layers": rec.n_gpu_layers,
                    "threads": rec.threads,
                    "port": rec.recommended_port,
                    "template_kind": rec.template_kind,
                    "summary_notes": rec.summary_notes,
                },
            })
        except SecurityError as e:
            return web.json_response({"error": f"Security violation: {e}"}, status=403)
        except Exception as e:
            return web.json_response({"error": f"Model inspection failed: {e}"}, status=400)

    async def handle_list_agents(self, request: web.Request) -> web.Response:
        agents = self.db.list_agents()
        # Verify and sync live health
        for a in agents:
            if a.status == "running" and not self.server_mgr.check_health(a.port):
                self.db.update_agent_status(a.name, "stopped", None)
                a.status = "stopped"

        data = [a.to_dict(mask_secrets=True) for a in self.db.list_agents()]
        return web.json_response(data)

    async def handle_create_agent(self, request: web.Request) -> web.Response:
        try:
            data = await request.json()
            name = validate_agent_name(data.get("name", ""))
            identity = (data.get("identity") or "").strip()
            personality = (data.get("personality") or "").strip()
            job = (data.get("job") or "").strip()
            model_path_str = data.get("model_path", "").strip()

            if not identity or not personality or not job or not model_path_str:
                return web.json_response(
                    {"error": "Name, identity, personality, job, and model_path are required."}, status=400
                )

            # Validate path & security
            model_target = validate_model_path(model_path_str)

            # Auto-inspect to get architecture if not supplied (non-blocking B1/N18)
            meta, rec = await asyncio.to_thread(analyze_model_and_recommend, str(model_target))
            arch = data.get("model_architecture") or meta.architecture

            # Allocation of port
            requested_port = data.get("port")
            port = validate_port(requested_port) if requested_port else self.server_mgr.find_free_port()

            # Config overrides or defaults
            ctx = validate_context_size(data.get("context_size") or rec.context_size)
            n_gpu = validate_gpu_layers(data.get("n_gpu_layers", rec.n_gpu_layers))
            threads = int(data.get("threads") or rec.threads)
            template_kind = data.get("template_kind") or rec.template_kind

            # Telegram settings
            tg_enabled = bool(data.get("telegram_enabled", False))
            tg_token = (data.get("telegram_bot_token") or "").strip() or None

            # Multi-agent settings
            callable_by = bool(data.get("callable_by_agents", True))
            raw_groups = data.get("groups") or ["default"]
            if isinstance(raw_groups, str):
                groups = [g.strip() for g in raw_groups.split(",") if g.strip()]
            else:
                groups = [str(g).strip() for g in raw_groups if str(g).strip()]

            # Tools settings
            raw_tools = data.get("tools")
            tools = None
            if raw_tools is not None:
                if isinstance(raw_tools, str):
                    tools = [t.strip() for t in raw_tools.split(",") if t.strip()]
                else:
                    tools = [str(t).strip() for t in raw_tools if str(t).strip()]

            # Create in Database
            agent = self.db.create_agent(
                name=name,
                identity=identity,
                personality=personality,
                job=job,
                model_path=str(model_target),
                model_architecture=arch,
                port=port,
                context_size=ctx,
                n_gpu_layers=n_gpu,
                threads=threads,
                template_kind=template_kind,
                telegram_enabled=tg_enabled,
                telegram_bot_token=tg_token,
                callable_by_agents=callable_by,
                groups=groups,
                tools=tools,
            )

            # Install Herdr / terminal launcher script in ~/.local/bin/<name>
            await asyncio.to_thread(HerdrIntegration.install_launcher, name)

            # Auto-start Telegram bot if configured
            if tg_enabled and tg_token:
                await self.telegram_mgr.start_agent_bot(name)

            return web.json_response({
                "ok": True,
                "message": f"Agent '{name}' registered successfully. Launcher installed.",
                "agent": agent.to_dict(mask_secrets=True),
            }, status=201)

        except SecurityError as e:
            return web.json_response({"error": f"Security check failed: {e}"}, status=403)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    async def handle_get_agent(self, request: web.Request) -> web.Response:
        name = request.match_info["name"]
        agent = self.db.get_agent_by_name(name)
        if not agent:
            return web.json_response({"error": f"Agent '{name}' not found."}, status=404)
        return web.json_response(agent.to_dict(mask_secrets=True))

    async def handle_update_agent(self, request: web.Request) -> web.Response:
        name = request.match_info["name"]
        data = await request.json()

        tg_token = data.get("telegram_bot_token")
        # If token unchanged or masked, preserve existing
        if tg_token and "••••" in tg_token:
            tg_token = None

        raw_groups = data.get("groups")
        groups = None
        if raw_groups is not None:
            if isinstance(raw_groups, str):
                groups = [g.strip() for g in raw_groups.split(",") if g.strip()]
            else:
                groups = [str(g).strip() for g in raw_groups if str(g).strip()]

        raw_tools = data.get("tools")
        tools = None
        if raw_tools is not None:
            if isinstance(raw_tools, str):
                tools = [t.strip() for t in raw_tools.split(",") if t.strip()]
            else:
                tools = [str(t).strip() for t in raw_tools if str(t).strip()]

        agent = self.db.update_agent(
            name=name,
            identity=data.get("identity"),
            personality=data.get("personality"),
            job=data.get("job"),
            telegram_enabled=data.get("telegram_enabled"),
            telegram_bot_token=tg_token,
            callable_by_agents=data.get("callable_by_agents"),
            groups=groups,
            tools=tools,
        )

        if not agent:
            return web.json_response({"error": f"Agent '{name}' not found."}, status=404)

        # Sync telegram state if bot enabled changed
        if agent.telegram_enabled and agent.telegram_bot_token:
            await self.telegram_mgr.start_agent_bot(agent.name)
        else:
            await self.telegram_mgr.stop_agent_bot(agent.name)

        return web.json_response({
            "ok": True,
            "message": f"Agent '{name}' updated.",
            "agent": agent.to_dict(mask_secrets=True),
        })

    async def handle_delete_agent(self, request: web.Request) -> web.Response:
        name = request.match_info["name"]
        # Stop servers and workers (non-blocking B1)
        await asyncio.to_thread(self.server_mgr.stop_agent_server, name)
        await self.telegram_mgr.stop_agent_bot(name)
        await asyncio.to_thread(HerdrIntegration.remove_launcher, name)
        deleted = await asyncio.to_thread(self.db.delete_agent, name)
        return web.json_response({"ok": deleted, "message": f"Agent '{name}' deleted."})

    async def handle_start_agent(self, request: web.Request) -> web.Response:
        name = request.match_info["name"]
        agent = self.db.get_agent_by_name(name)
        if not agent:
            return web.json_response({"error": f"Agent '{name}' not found."}, status=404)

        ok, msg = await asyncio.to_thread(self.server_mgr.ensure_agent_running, name)
        if not ok:
            return web.json_response({"ok": False, "error": msg}, status=500)

        # Start Telegram worker if enabled
        if agent.telegram_enabled and agent.telegram_bot_token:
            tg_ok, tg_msg = await self.telegram_mgr.start_agent_bot(name)
            msg += f" | Telegram: {tg_msg}"

        # Sync with Agent-Hub
        await asyncio.to_thread(self.router.sync_with_agent_hub)

        return web.json_response({"ok": True, "message": msg})

    async def handle_activate_agent(self, request: web.Request) -> web.Response:
        """Hot-swap / ensure this agent's model is loaded into memory, evicting LRU if needed."""
        name = request.match_info["name"]
        agent = self.db.get_agent_by_name(name)
        if not agent:
            return web.json_response({"error": f"Agent '{name}' not found."}, status=404)

        ok, msg = await asyncio.to_thread(self.server_mgr.ensure_agent_running, name)
        if not ok:
            return web.json_response({"ok": False, "error": msg}, status=500)
        return web.json_response({"ok": True, "message": msg})

    async def handle_stop_agent(self, request: web.Request) -> web.Response:
        name = request.match_info["name"]
        await self.telegram_mgr.stop_agent_bot(name)
        ok, msg = await asyncio.to_thread(self.server_mgr.stop_agent_server, name)
        await asyncio.to_thread(self.router.sync_with_agent_hub)
        return web.json_response({"ok": ok, "message": msg})

    async def handle_spawn_herdr(self, request: web.Request) -> web.Response:
        name = request.match_info["name"]
        agent = self.db.get_agent_by_name(name)
        if not agent:
            return web.json_response({"error": f"Agent '{name}' not found."}, status=404)

        # Ensure server is running (LRU hot-swap)
        ok, msg = await asyncio.to_thread(self.server_mgr.ensure_agent_running, name)
        if not ok:
            return web.json_response({"error": f"Failed to start server before spawning: {msg}"}, status=500)

        ok, msg, details = await asyncio.to_thread(HerdrIntegration.spawn_in_herdr, name, prefer_tab=True)
        return web.json_response({"ok": ok, "message": msg, "details": details})

    async def handle_get_logs(self, request: web.Request) -> web.Response:
        name = request.match_info["name"]
        logs = await asyncio.to_thread(self.server_mgr.get_agent_logs, name, max_lines=150)
        return web.json_response({"agent": name, "logs": logs})

    async def handle_get_history(self, request: web.Request) -> web.Response:
        name = request.match_info["name"]
        agent = self.db.get_agent_by_name(name)
        if not agent:
            return web.json_response({"error": f"Agent '{name}' not found."}, status=404)
        raw_limit = request.query.get("limit", "50")
        try:
            limit = max(1, min(int(raw_limit), 200))
        except (ValueError, TypeError):
            limit = 50
        history = await asyncio.to_thread(self.db.get_chat_history, name, limit=limit)
        return web.json_response({"agent": name, "history": history})

    async def handle_chat_agent(self, request: web.Request) -> web.Response:
        name = request.match_info["name"]
        agent = self.db.get_agent_by_name(name)
        if not agent:
            return web.json_response({"error": f"Agent '{name}' not found."}, status=404)

        # Ensure server is active via LRU hot-swapping
        ok, msg = await asyncio.to_thread(self.server_mgr.ensure_agent_running, name)
        if not ok:
            return web.json_response({"error": f"Agent server is not running: {msg}"}, status=503)

        agent = self.db.get_agent_by_name(name)
        if not agent:
            return web.json_response({"error": f"Agent '{name}' not found."}, status=404)

        data = await request.json()
        prompt = (data.get("prompt") or "").strip()
        if not prompt:
            return web.json_response({"error": "Prompt cannot be empty."}, status=400)

        history = data.get("history", [])
        system_prompt = build_agent_system_prompt(agent)

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history[-20:])  # Cap at last 20 messages (N13)
        messages.append({"role": "user", "content": prompt})

        # Execute chat completion through ToolHub tool loop
        executed_tools = []
        async def _on_tool_event(event_type: str, evt_data: Dict[str, Any]):
            if event_type == "tool_call":
                executed_tools.append({
                    "name": evt_data.get("name"),
                    "arguments": evt_data.get("arguments"),
                })

        timeout = aiohttp.ClientTimeout(total=120)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                reply, _ = await self.tool_hub.chat_with_tools_loop(
                    agent=agent,
                    messages=messages,
                    session=session,
                    on_event=_on_tool_event,
                    max_iterations=4,
                )

                await asyncio.to_thread(self.db.log_chat, agent.name, "web_user", prompt)
                await asyncio.to_thread(self.db.log_chat, agent.name, "assistant", reply)

                return web.json_response({
                    "ok": True,
                    "agent": agent.name,
                    "response": reply,
                    "tool_calls": executed_tools,
                })
        except Exception as e:
            logger.error("Chat agent inference error: %s", e)
            return web.json_response({"error": f"Inference error: {e}"}, status=500)

    # ---------------------------------------------------------------------------
    # Tools & MCP Server Handlers
    # ---------------------------------------------------------------------------

    async def handle_list_tools(self, request: web.Request) -> web.Response:
        catalog = await asyncio.to_thread(self.tool_hub.get_available_tools_catalog)
        return web.json_response({"tools": catalog})

    @staticmethod
    def _mask_mcp_server_secrets(srv: Dict[str, Any]) -> Dict[str, Any]:
        s_copy = dict(srv)
        if "env" in s_copy and isinstance(s_copy["env"], dict):
            s_copy["env"] = {k: mask_token(str(v)) if v else "" for k, v in s_copy["env"].items()}
        return s_copy

    async def handle_list_mcp_servers(self, request: web.Request) -> web.Response:
        # Automatically sync with Claude Desktop / Cursor / Smithery config files
        await asyncio.to_thread(sync_external_client_configs, self.db)
        servers = await asyncio.to_thread(self.db.list_mcp_servers)
        masked_servers = [self._mask_mcp_server_secrets(s) for s in servers]
        return web.json_response({"servers": masked_servers})

    async def handle_create_mcp_server(self, request: web.Request) -> web.Response:
        data = await request.json()
        name = (data.get("name") or "").strip()
        command = (data.get("command") or "").strip()
        args = data.get("args") or []
        env = data.get("env") or {}
        description = (data.get("description") or "").strip()
        enabled = bool(data.get("enabled", True))

        if not name or not command:
            return web.json_response({"error": "Name and command are required."}, status=400)

        srv = await asyncio.to_thread(
            self.db.create_mcp_server,
            name=name, command=command, args=args, env=env, enabled=enabled, description=description
        )
        return web.json_response({"ok": True, "server": self._mask_mcp_server_secrets(srv)}, status=201)

    async def handle_update_mcp_server(self, request: web.Request) -> web.Response:
        name = request.match_info["name"]
        data = await request.json()
        new_env = data.get("env")
        if new_env is not None and isinstance(new_env, dict):
            existing = await asyncio.to_thread(self.db.get_mcp_server, name)
            if existing and "env" in existing and isinstance(existing["env"], dict):
                merged_env = dict(existing["env"])
                for k, v in new_env.items():
                    if "••••" in str(v):
                        continue  # Keep existing unmasked secret
                    merged_env[k] = str(v)
                new_env = merged_env

        srv = await asyncio.to_thread(
            self.db.update_mcp_server,
            name=name,
            command=data.get("command"),
            args=data.get("args"),
            env=new_env,
            enabled=data.get("enabled"),
            description=data.get("description"),
        )
        if not srv:
            return web.json_response({"error": f"MCP server '{name}' not found."}, status=404)
        return web.json_response({"ok": True, "server": self._mask_mcp_server_secrets(srv)})

    async def handle_delete_mcp_server(self, request: web.Request) -> web.Response:
        name = request.match_info["name"]
        self.tool_hub.mcp_mgr.stop_server(name)
        deleted = await asyncio.to_thread(self.db.delete_mcp_server, name)
        return web.json_response({"ok": deleted})

    async def handle_test_mcp_server(self, request: web.Request) -> web.Response:
        name = request.match_info["name"]
        srv = await asyncio.to_thread(self.db.get_mcp_server, name)
        if not srv:
            return web.json_response({"error": f"MCP server '{name}' not found."}, status=404)

        from ..mcp_client import MCPServerInfo
        srv_info = MCPServerInfo(
            id=srv["id"],
            name=srv["name"],
            command=srv["command"],
            args=srv["args"],
            env=srv["env"],
            enabled=srv["enabled"],
            description=srv["description"],
        )
        conn = await asyncio.to_thread(self.tool_hub.mcp_mgr.get_or_start_server, srv_info)
        if not conn:
            return web.json_response({"ok": False, "error": f"Failed to start/connect to MCP server '{name}'."}, status=500)

        discovered = await asyncio.to_thread(conn.refresh_tools, 15.0)
        tools = [t["namespaced_name"] for t in discovered]
        await asyncio.to_thread(self.db.set_mcp_cached_tools, name, discovered)
        return web.json_response({"ok": True, "tools_count": len(tools), "tools": tools})

    async def handle_featured_mcp(self, request: web.Request) -> web.Response:
        """Returns list of curated 1-click popular MCP tools with installed status."""
        installed_names = {s["name"] for s in await asyncio.to_thread(self.db.list_mcp_servers)}
        catalog = []
        for item in FEATURED_MCP_CATALOG:
            item_copy = dict(item)
            item_copy["installed"] = item["name"] in installed_names
            catalog.append(item_copy)
        return web.json_response({"featured": catalog})

    async def handle_quick_install_mcp(self, request: web.Request) -> web.Response:
        """
        Universal 1-click MCP installer.
        Accepts a Smithery URL, CLI command, JSON snippet, package identifier, or featured ID.
        Automatically parses, creates server in DB, verifies connection, and caches tools.
        """
        try:
            data = await request.json()
            featured_id = data.get("featured_id")
            raw_input = (data.get("input") or data.get("raw_input") or "").strip()
            env_override = data.get("env") or {}

            parsed = None
            if featured_id:
                for item in FEATURED_MCP_CATALOG:
                    if item["id"] == featured_id:
                        parsed = {
                            "name": item["name"],
                            "command": item["command"],
                            "args": list(item["args"]),
                            "env": dict(item.get("env") or {}),
                            "description": item.get("description", ""),
                        }
                        break
                if not parsed:
                    return web.json_response({"error": f"Featured tool '{featured_id}' not found."}, status=404)
            elif raw_input:
                parsed = parse_mcp_install_string(raw_input)
            elif data.get("name") and data.get("command"):
                parsed = {
                    "name": data.get("name"),
                    "command": data.get("command"),
                    "args": data.get("args") or [],
                    "env": data.get("env") or {},
                    "description": data.get("description") or "",
                }
            else:
                return web.json_response({
                    "error": "Please provide a Smithery link, command, package name, or select a featured tool."
                }, status=400)

            # Merge any user-provided env keys
            if env_override and isinstance(env_override, dict):
                parsed["env"].update(env_override)

            name = validate_agent_name(parsed["name"])
            existing = await asyncio.to_thread(self.db.get_mcp_server, name)
            if existing:
                srv = await asyncio.to_thread(
                    self.db.update_mcp_server,
                    name=name,
                    command=parsed["command"],
                    args=parsed["args"],
                    env=parsed["env"],
                    enabled=True,
                    description=parsed["description"],
                )
            else:
                srv = await asyncio.to_thread(
                    self.db.create_mcp_server,
                    name=name,
                    command=parsed["command"],
                    args=parsed["args"],
                    env=parsed["env"],
                    enabled=True,
                    description=parsed["description"],
                )

            # Test connection & discover tools in background thread
            from ..mcp_client import MCPServerInfo
            srv_info = MCPServerInfo(
                id=srv["id"],
                name=srv["name"],
                command=srv["command"],
                args=srv["args"],
                env=srv["env"],
                enabled=srv["enabled"],
                description=srv["description"],
            )
            conn = await asyncio.to_thread(self.tool_hub.mcp_mgr.get_or_start_server, srv_info)
            tools = []
            if conn:
                discovered = await asyncio.to_thread(conn.refresh_tools, 15.0)
                tools = [t["namespaced_name"] for t in discovered]
                await asyncio.to_thread(self.db.set_mcp_cached_tools, name, discovered)

            return web.json_response({
                "ok": True,
                "server": self._mask_mcp_server_secrets(srv),
                "tools_count": len(tools),
                "tools": tools,
                "message": f"Successfully connected '{name}'! {len(tools)} tools discovered.",
            })

        except Exception as e:
            logger.error("Quick install error: %s", e)
            return web.json_response({"error": f"Install failed: {e}"}, status=400)

    async def handle_sync_external_mcp(self, request: web.Request) -> web.Response:
        """Scans Claude Desktop and Cursor configs to import newly added servers."""
        imported = await asyncio.to_thread(sync_external_client_configs, self.db)
        return web.json_response({
            "ok": True,
            "imported_count": len(imported),
            "imported": [self._mask_mcp_server_secrets(s) for s in imported],
            "message": f"Synced with Claude & Smithery configs: {len(imported)} new servers imported.",
        })

    async def handle_search_mcp(self, request: web.Request) -> web.Response:
        """Searches Smithery's registry."""
        q = request.query.get("q", "").strip()
        if not q:
            return web.json_response({"results": []})
        results = await asyncio.to_thread(search_smithery_registry, q)
        return web.json_response({"query": q, "results": results})

    async def handle_list_groups(self, request: web.Request) -> web.Response:
        agents = await asyncio.to_thread(self.db.list_agents)
        groups_map: Dict[str, List[Dict[str, Any]]] = {}
        for a in agents:
            for g in a.groups:
                groups_map.setdefault(g, []).append(a.to_dict(mask_secrets=True))
        return web.json_response({"groups": groups_map})

    async def handle_group_ask(self, request: web.Request) -> web.Response:
        group_name = request.match_info["group_name"]
        data = await request.json()
        prompt = data.get("prompt", "").strip()
        caller = data.get("caller", "coordinator").strip()

        if not prompt:
            return web.json_response({"error": "Prompt cannot be empty."}, status=400)

        out = await asyncio.to_thread(self.router.query_group, group_name, caller, prompt)
        results = out.get("results", []) if isinstance(out, dict) else out
        skipped = out.get("skipped", []) if isinstance(out, dict) else []
        return web.json_response({
            "group": group_name,
            "caller": caller,
            "results": results,
            "skipped": skipped,
        })

    async def handle_sse_events(self, request: web.Request) -> web.StreamResponse:
        """Server-Sent Events endpoint to stream live status and heartbeat to the Web UI."""
        response = web.StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        await response.prepare(request)

        try:
            while True:
                agents = [a.to_dict(mask_secrets=True) for a in self.db.list_agents()]
                event_data = json.dumps({"type": "agents_update", "agents": agents})
                await response.write(f"data: {event_data}\n\n".encode("utf-8"))
                await asyncio.sleep(3.0)
        except (asyncio.CancelledError, ConnectionResetError, aiohttp.ClientConnectionResetError):
            pass

        return response

    async def start(self):
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        logger.info("Sheprd Web UI active at http://%s:%s", self.host, self.port)


def run_web(host: str = "127.0.0.1", port: int = 8765):
    """Entry point for running the web server synchronously."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s [%(name)s]: %(message)s",
        datefmt="%H:%M:%S",
    )
    webapp = SheprdWebApp(host=host, port=port)
    web.run_app(webapp.app, host=host, port=port)
