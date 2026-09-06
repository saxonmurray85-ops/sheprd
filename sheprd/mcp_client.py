"""
Model Context Protocol (MCP) Client for Sheprd.
Enables agents to discover, load, and execute tools from external MCP servers
(via stdio JSON-RPC 2.0). Compatible with Smithery.ai, official MCP servers,
and local agent tools.
"""

import asyncio
import json
import logging
import os
import shutil
import subprocess
import threading
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple


logger = logging.getLogger("sheprd.mcp")


@dataclass
class MCPServerInfo:
    id: Optional[int]
    name: str
    command: str
    args: List[str]
    env: Dict[str, str]
    enabled: bool
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StdioMCPConnection:
    """Manages an active stdio JSON-RPC connection to an MCP server process."""

    def __init__(self, name: str, command: str, args: List[str], env: Optional[Dict[str, str]] = None):
        self.name = name
        self.command = command
        self.args = args
        self.env = env or {}
        self.proc: Optional[subprocess.Popen] = None
        self._msg_id = 0
        self._lock = threading.Lock()
        self.tools: List[Dict[str, Any]] = []

    def _next_id(self) -> int:
        with self._lock:
            self._msg_id += 1
            return self._msg_id

    def start(self, timeout_sec: float = 10.0) -> bool:
        """Starts the MCP server subprocess and executes initialization handshake."""
        # Resolve command path
        exec_path = shutil.which(self.command) or self.command
        full_env = os.environ.copy()
        full_env.update(self.env)
        # Ensure node and mise binaries are in PATH if running npx
        if "PATH" in os.environ:
            full_env["PATH"] = os.environ["PATH"]

        cmd = [exec_path] + self.args
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=full_env,
            )
        except Exception as e:
            logger.error("Failed to spawn MCP server '%s' (%s): %s", self.name, cmd, e)
            return False

        # 1. Initialize
        init_id = self._next_id()
        init_req = {
            "jsonrpc": "2.0",
            "id": init_id,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "sheprd", "version": "1.0.0"},
            },
        }

        resp = self._send_request(init_req, timeout_sec=timeout_sec)
        if not resp or "error" in resp:
            logger.error("MCP initialize failed for '%s': %s", self.name, resp)
            self.stop()
            return False

        # 2. Initialized notification
        self._send_notification({"jsonrpc": "2.0", "method": "notifications/initialized"})

        # 3. Discover tools
        self.refresh_tools(timeout_sec=timeout_sec)
        return True

    def _send_notification(self, notif: Dict[str, Any]) -> None:
        if not self.proc or not self.proc.stdin:
            return
        try:
            line = json.dumps(notif) + "\n"
            self.proc.stdin.write(line)
            self.proc.stdin.flush()
        except Exception as e:
            logger.warning("Error sending notification to MCP '%s': %s", self.name, e)

    def _send_request(self, req: Dict[str, Any], timeout_sec: float = 15.0) -> Optional[Dict[str, Any]]:
        if not self.proc or not self.proc.stdin or not self.proc.stdout:
            return None

        req_id = req.get("id")
        try:
            line = json.dumps(req) + "\n"
            self.proc.stdin.write(line)
            self.proc.stdin.flush()
        except Exception as e:
            logger.error("Failed to write request to MCP '%s': %s", self.name, e)
            return None

        # Read responses until match or EOF
        import time
        start_time = time.time()
        while time.time() - start_time < timeout_sec:
            if self.proc.poll() is not None:
                stderr_out = self.proc.stderr.read() if self.proc.stderr else ""
                logger.error("MCP process '%s' terminated prematurely: %s", self.name, stderr_out)
                return None

            try:
                # Read line
                line = self.proc.stdout.readline()
                if not line:
                    time.sleep(0.05)
                    continue
                line = line.strip()
                if not line or not line.startswith("{"):
                    continue

                msg = json.loads(line)
                if msg.get("id") == req_id:
                    return msg
            except json.JSONDecodeError:
                continue
            except Exception as e:
                logger.error("Error reading from MCP '%s': %s", self.name, e)
                break

        logger.warning("MCP request timed out for '%s' (req_id=%s)", self.name, req_id)
        return None

    def refresh_tools(self, timeout_sec: float = 10.0) -> List[Dict[str, Any]]:
        """Queries tools/list from the MCP server and stores them."""
        req_id = self._next_id()
        req = {"jsonrpc": "2.0", "id": req_id, "method": "tools/list", "params": {}}
        resp = self._send_request(req, timeout_sec=timeout_sec)
        if not resp or "error" in resp:
            logger.warning("Failed to retrieve tools from MCP '%s': %s", self.name, resp)
            self.tools = []
            return []

        raw_tools = resp.get("result", {}).get("tools", [])
        converted = []
        for t in raw_tools:
            tool_name = t.get("name", "")
            if not tool_name:
                continue
            namespaced_name = f"mcp__{self.name}__{tool_name}"
            desc = t.get("description") or f"Tool from MCP server {self.name}"
            params = t.get("inputSchema") or {"type": "object", "properties": {}}

            converted.append({
                "raw_name": tool_name,
                "namespaced_name": namespaced_name,
                "server_name": self.name,
                "description": desc,
                "definition": {
                    "type": "function",
                    "function": {
                        "name": namespaced_name,
                        "description": f"[{self.name}] {desc}",
                        "parameters": params,
                    },
                },
            })

        self.tools = converted
        logger.info("MCP server '%s' registered %d tools: %s", self.name, len(converted), [t['raw_name'] for t in converted])
        return self.tools

    def call_tool(self, raw_tool_name: str, arguments: Dict[str, Any], timeout_sec: float = 30.0) -> str:
        """Executes a tool on the MCP server via tools/call."""
        req_id = self._next_id()
        req = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "tools/call",
            "params": {"name": raw_tool_name, "arguments": arguments},
        }

        resp = self._send_request(req, timeout_sec=timeout_sec)
        if not resp:
            return f"Error: MCP server '{self.name}' did not respond within {timeout_sec}s."

        if "error" in resp:
            err = resp["error"]
            return f"MCP Error ({err.get('code', 'unknown')}): {err.get('message', 'No error message')}"

        result = resp.get("result", {})
        is_error = result.get("isError", False)
        content_items = result.get("content", [])

        parts = []
        for item in content_items:
            if item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif item.get("type") == "image":
                parts.append(f"[Image: {item.get('mimeType', 'unknown')}]")
            else:
                parts.append(json.dumps(item))

        output = "\n".join(parts) if parts else json.dumps(result)
        if is_error:
            return f"Tool returned error: {output}"
        return output

    def stop(self) -> None:
        """Terminates the MCP server process."""
        if self.proc:
            try:
                if self.proc.stdin:
                    try:
                        self.proc.stdin.close()
                    except Exception:
                        pass
                if self.proc.stdout:
                    try:
                        self.proc.stdout.close()
                    except Exception:
                        pass
                if self.proc.stderr:
                    try:
                        self.proc.stderr.close()
                    except Exception:
                        pass
                self.proc.terminate()
                self.proc.wait(timeout=2.0)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.proc = None


class MCPManager:
    """Manages active MCP server connections and server definitions."""

    def __init__(self, db=None):
        self.db = db
        self.connections: Dict[str, StdioMCPConnection] = {}
        self._lock = threading.Lock()

    def get_or_start_server(self, server_info: MCPServerInfo) -> Optional[StdioMCPConnection]:
        with self._lock:
            conn = self.connections.get(server_info.name)
            if conn and conn.proc and conn.proc.poll() is None:
                return conn

            conn = StdioMCPConnection(
                name=server_info.name,
                command=server_info.command,
                args=server_info.args,
                env=server_info.env,
            )
            ok = conn.start()
            if ok:
                self.connections[server_info.name] = conn
                return conn
            return None

    def get_connection(self, name: str) -> Optional[StdioMCPConnection]:
        return self.connections.get(name)

    def stop_server(self, name: str) -> None:
        with self._lock:
            conn = self.connections.pop(name, None)
            if conn:
                conn.stop()

    def stop_all(self) -> None:
        with self._lock:
            for conn in list(self.connections.values()):
                conn.stop()
            self.connections.clear()
