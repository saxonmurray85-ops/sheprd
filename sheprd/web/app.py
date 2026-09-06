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
    mask_token,
    validate_agent_name,
    validate_context_size,
    validate_gpu_layers,
    validate_groups,
    validate_model_path,
    validate_port,
)
from ..server_manager import ServerManager
from ..telegram_service import TelegramServiceManager


logger = logging.getLogger("sheprd.web")
WEB_DIR = Path(__file__).parent


class SheprdWebApp:
    def __init__(self, db: Optional[Database] = None, host: str = "127.0.0.1", port: int = 8765):
        self.host = host
        self.port = port
        self.db = db or Database()
        self.server_mgr = ServerManager(self.db)
        self.telegram_mgr = TelegramServiceManager(self.db)
        self.router = MultiAgentRouter(self.db)
        self.csrf_token = secrets.token_urlsafe(32)
        self.app = web.Application(middlewares=[self.csrf_middleware])
        self._setup_routes()

    @web.middleware
    async def csrf_middleware(self, request: web.Request, handler):
        # S2: Enforce CSRF & Origin validation on mutating requests
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

            # 3. Check X-Sheprd-Token header
            token = request.headers.get("X-Sheprd-Token")
            if token and not secrets.compare_digest(token, self.csrf_token):
                return web.json_response({"error": "Forbidden: Invalid CSRF/auth token (S2)."}, status=403)

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
        self.app.router.add_post("/api/agents/{name}/spawn-herdr", self.handle_spawn_herdr)
        self.app.router.add_get("/api/agents/{name}/logs", self.handle_get_logs)
        self.app.router.add_post("/api/agents/{name}/chat", self.handle_chat_agent)

        # Multi-agent & Group APIs
        self.app.router.add_get("/api/groups", self.handle_list_groups)
        self.app.router.add_post("/api/groups/{group_name}/ask", self.handle_group_ask)

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

            # Auto-inspect to get architecture if not supplied
            meta, rec = analyze_model_and_recommend(str(model_target))
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
            )

            # Install Herdr / terminal launcher script in ~/.local/bin/<name>
            HerdrIntegration.install_launcher(name)

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

        agent = self.db.update_agent(
            name=name,
            identity=data.get("identity"),
            personality=data.get("personality"),
            job=data.get("job"),
            telegram_enabled=data.get("telegram_enabled"),
            telegram_bot_token=tg_token,
            callable_by_agents=data.get("callable_by_agents"),
            groups=groups,
        )

        if not agent:
            return web.json_response({"error": f"Agent '{name}' not found."}, status=404)

        # Sync telegram state if bot enabled changed
        if agent.telegram_enabled and agent.telegram_bot_token and agent.status == "running":
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
        # Stop servers and workers
        self.server_mgr.stop_agent_server(name)
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

        ok, msg = await asyncio.to_thread(self.server_mgr.start_agent_server, name)
        if not ok:
            return web.json_response({"ok": False, "error": msg}, status=500)

        # Start Telegram worker if enabled
        if agent.telegram_enabled and agent.telegram_bot_token:
            tg_ok, tg_msg = await self.telegram_mgr.start_agent_bot(name)
            msg += f" | Telegram: {tg_msg}"

        # Sync with Agent-Hub
        await asyncio.to_thread(self.router.sync_with_agent_hub)

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

        # Ensure server is running (non-blocking B1)
        if agent.status != "running":
            ok, msg = await asyncio.to_thread(self.server_mgr.start_agent_server, name)
            if not ok:
                return web.json_response({"error": f"Failed to start server before spawning: {msg}"}, status=500)

        ok, msg, details = await asyncio.to_thread(HerdrIntegration.spawn_in_herdr, name, prefer_tab=True)
        return web.json_response({"ok": ok, "message": msg, "details": details})

    async def handle_get_logs(self, request: web.Request) -> web.Response:
        name = request.match_info["name"]
        logs = await asyncio.to_thread(self.server_mgr.get_agent_logs, name, max_lines=150)
        return web.json_response({"agent": name, "logs": logs})

    async def handle_chat_agent(self, request: web.Request) -> web.Response:
        name = request.match_info["name"]
        agent = self.db.get_agent_by_name(name)
        if not agent:
            return web.json_response({"error": f"Agent '{name}' not found."}, status=404)

        if agent.status != "running":
            ok, msg = await asyncio.to_thread(self.server_mgr.start_agent_server, name)
            if not ok:
                return web.json_response({"error": f"Agent server is not running: {msg}"}, status=503)

        data = await request.json()
        prompt = (data.get("prompt") or "").strip()
        if not prompt:
            return web.json_response({"error": "Prompt cannot be empty."}, status=400)

        history = data.get("history", [])

        system_prompt = (
            f"You are {agent.name}.\n"
            f"IDENTITY: {agent.identity}\n"
            f"PERSONALITY: {agent.personality}\n"
            f"PRIMARY JOB: {agent.job}\n\n"
            f"Respond clearly and stay faithful to your identity and task."
        )

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history[-10:])
        messages.append({"role": "user", "content": prompt})

        url = f"http://127.0.0.1:{agent.port}/v1/chat/completions"
        payload = {
            "model": agent.name,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2048,
        }

        # B1 fix: Use async non-blocking aiohttp.ClientSession instead of urllib
        timeout = aiohttp.ClientTimeout(total=120)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as resp:
                    if resp.status == 200:
                        res = await resp.json()
                        reply = res.get("choices", [{}])[0].get("message", {}).get("content", "")

                        await asyncio.to_thread(self.db.log_chat, agent.name, "web_user", prompt)
                        await asyncio.to_thread(self.db.log_chat, agent.name, "assistant", reply)

                        return web.json_response({
                            "ok": True,
                            "agent": agent.name,
                            "response": reply,
                            "usage": res.get("usage", {}),
                        })
                    else:
                        err_text = await resp.text()
                        return web.json_response(
                            {"error": f"Local inference server error ({resp.status}): {err_text}"},
                            status=resp.status,
                        )
        except Exception as e:
            return web.json_response({"error": f"Inference error: {e}"}, status=500)

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

        results = await asyncio.to_thread(self.router.query_group, group_name, caller, prompt)
        return web.json_response({
            "group": group_name,
            "caller": caller,
            "results": results,
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
