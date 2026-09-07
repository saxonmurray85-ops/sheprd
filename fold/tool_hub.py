"""
Sheprd Tool Hub.
Central orchestrator for agent tool calling. Manages Core Tools and external MCP servers,
formats OpenAI function calling definitions for llama-server, and coordinates
multi-step tool execution loops.
"""

import asyncio
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import aiohttp

from .core_tools import CORE_TOOLS_REGISTRY, execute_core_tool, get_core_tool_definitions
from .database import AgentRecord, Database
from .mcp_client import MCPManager, MCPServerInfo


logger = logging.getLogger("fold.tools.hub")


class ToolHub:
    """Coordinates Core Tools and MCP servers for Sheprd agents."""

    def __init__(self, db: Optional[Database] = None):
        self.db = db or Database()
        self.mcp_mgr = MCPManager(self.db)

    def get_available_tools_catalog(self) -> List[Dict[str, Any]]:
        """Returns metadata for all tools currently available in the system."""
        catalog = []

        # 1. Core tools
        for name, data in CORE_TOOLS_REGISTRY.items():
            defn = data["definition"]["function"]
            catalog.append({
                "type": "core",
                "name": name,
                "description": defn["description"],
                "parameters": defn.get("parameters", {}),
                "server_name": "built-in",
            })

        # 2. MCP server tools (prefer cached tools to avoid blocking event loop or spawning processes, N7)
        mcp_servers = self.db.list_mcp_servers()
        for srv in mcp_servers:
            if not srv["enabled"]:
                continue

            # Check if tools are cached in database
            cached_tools = srv.get("cached_tools") or []
            if cached_tools:
                for tool in cached_tools:
                    catalog.append({
                        "type": "mcp",
                        "name": tool["namespaced_name"],
                        "raw_name": tool["raw_name"],
                        "description": tool["description"],
                        "parameters": tool.get("definition", {}).get("function", {}).get("parameters", {}),
                        "server_name": srv["name"],
                    })
                continue

            # Fallback: only read from live active connection if already running
            conn = self.mcp_mgr.get_connection(srv["name"])
            if conn and conn.tools:
                for tool in conn.tools:
                    catalog.append({
                        "type": "mcp",
                        "name": tool["namespaced_name"],
                        "raw_name": tool["raw_name"],
                        "description": tool["description"],
                        "parameters": tool["definition"]["function"].get("parameters", {}),
                        "server_name": srv["name"],
                    })

        return catalog

    def get_tool_definitions_for_agent(self, agent: AgentRecord) -> List[Dict[str, Any]]:
        """Returns OpenAI-format tool definitions enabled for this agent."""
        definitions = []
        agent_tools = set(agent.tools or [])

        # Check for core tools
        for name, data in CORE_TOOLS_REGISTRY.items():
            if name in agent_tools or "all_core" in agent_tools:
                definitions.append(data["definition"])

        # Check for MCP servers
        mcp_servers = self.db.list_mcp_servers()
        for srv in mcp_servers:
            if not srv["enabled"]:
                continue
            srv_key = f"mcp:{srv['name']}"
            if srv_key in agent_tools or "all_mcp" in agent_tools or any(t.startswith(f"mcp__{srv['name']}__") for t in agent_tools):
                # Use cached tools if available
                cached_tools = srv.get("cached_tools") or []
                if cached_tools:
                    for tool in cached_tools:
                        if srv_key in agent_tools or "all_mcp" in agent_tools or tool["namespaced_name"] in agent_tools:
                            definitions.append(tool["definition"])
                    continue

                # Live connection fallback
                conn = self.mcp_mgr.get_connection(srv["name"])
                if conn and conn.tools:
                    for tool in conn.tools:
                        if srv_key in agent_tools or "all_mcp" in agent_tools or tool["namespaced_name"] in agent_tools:
                            definitions.append(tool["definition"])

        return definitions

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any], agent: Optional[AgentRecord] = None) -> str:
        """Executes a tool call and returns the text result with strict execution permissions (N11)."""
        logger.info("Executing tool '%s' with arguments: %s", tool_name, json.dumps(arguments)[:200])

        # Enforce agent tool execution permission (N11)
        if agent is not None:
            agent_tools = set(agent.tools or [])
            if tool_name in CORE_TOOLS_REGISTRY:
                if tool_name not in agent_tools and "all_core" not in agent_tools:
                    return f"Permission error: Tool '{tool_name}' is not authorized for agent '{agent.name}'."
            elif tool_name.startswith("mcp__"):
                parts = tool_name.split("__", 2)
                server_name = parts[1] if len(parts) >= 2 else ""
                srv_key = f"mcp:{server_name}"
                if srv_key not in agent_tools and "all_mcp" not in agent_tools and tool_name not in agent_tools:
                    return f"Permission error: MCP tool '{tool_name}' is not authorized for agent '{agent.name}'."

        # 1. Check Core Tools
        if tool_name in CORE_TOOLS_REGISTRY:
            return await asyncio.to_thread(execute_core_tool, tool_name, arguments)

        # 2. Check MCP Tools: mcp__<server_name>__<tool_name>
        if tool_name.startswith("mcp__"):
            parts = tool_name.split("__", 2)
            if len(parts) == 3:
                server_name, raw_tool_name = parts[1], parts[2]
                conn = self.mcp_mgr.get_connection(server_name)
                if not conn or not conn.proc or conn.proc.poll() is not None:
                    # Attempt non-blocking start (N7)
                    srv = self.db.get_mcp_server(server_name)
                    if srv:
                        srv_info = MCPServerInfo(
                            id=srv["id"],
                            name=srv["name"],
                            command=srv["command"],
                            args=srv["args"],
                            env=srv["env"],
                            enabled=srv["enabled"],
                            description=srv["description"],
                        )
                        conn = await asyncio.to_thread(self.mcp_mgr.get_or_start_server, srv_info)
                        if conn and conn.tools:
                            # Update cached tools in DB
                            await asyncio.to_thread(self.db.set_mcp_cached_tools, server_name, conn.tools)

                if conn:
                    return await asyncio.to_thread(conn.call_tool, raw_tool_name, arguments)
                return f"Error: MCP server '{server_name}' is not running."

        return f"Error: Tool '{tool_name}' not found."

    async def chat_with_tools_loop(
        self,
        agent: AgentRecord,
        messages: List[Dict[str, Any]],
        session: aiohttp.ClientSession,
        on_event: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
        max_iterations: int = 4,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Executes the OpenAI-compatible tool-calling loop against llama-server.
        Dispatches tool calls, returns results to the server, and returns final answer.
        """
        tool_defs = self.get_tool_definitions_for_agent(agent)
        current_messages = list(messages)
        iteration = 0
        url = f"http://127.0.0.1:{agent.port}/v1/chat/completions"

        while iteration < max_iterations:
            iteration += 1

            payload: Dict[str, Any] = {
                "model": agent.name,
                "messages": current_messages,
                "temperature": 0.7,
                "max_tokens": 1500,
            }

            if tool_defs:
                payload["tools"] = tool_defs
                payload["tool_choice"] = "auto"

            try:
                async with session.post(url, json=payload, timeout=120) as resp:
                    if resp.status != 200:
                        err_msg = f"llama-server returned status {resp.status}"
                        logger.error("%s for agent %s", err_msg, agent.name)
                        return f"⚠️ {err_msg}", current_messages

                    res_data = await resp.json()
                    choice = res_data.get("choices", [{}])[0]
                    message = choice.get("message", {})
                    finish_reason = choice.get("finish_reason")
                    content = message.get("content") or ""
                    tool_calls = message.get("tool_calls", [])

                    # If model didn't call any tools, we have our final response
                    if not tool_calls or finish_reason != "tool_calls":
                        if content:
                            current_messages.append({"role": "assistant", "content": content})
                        return content, current_messages

                    # Model wants to execute one or more tools!
                    logger.info("Agent '%s' issued %d tool call(s)", agent.name, len(tool_calls))
                    current_messages.append({
                        "role": "assistant",
                        "content": content or "",
                        "tool_calls": tool_calls,
                    })

                    # Execute each tool call
                    for tc in tool_calls:
                        tc_id = tc.get("id", f"call_{iteration}")
                        fn = tc.get("function", {})
                        fn_name = fn.get("name", "")
                        fn_args_raw = fn.get("arguments", "{}")
                        try:
                            fn_args = json.loads(fn_args_raw) if isinstance(fn_args_raw, str) else fn_args_raw
                        except Exception:
                            fn_args = {}

                        if on_event:
                            try:
                                res = on_event("tool_call", {"name": fn_name, "arguments": fn_args, "id": tc_id})
                                if asyncio.iscoroutine(res):
                                    await res
                            except Exception:
                                pass

                        tool_output = await self.execute_tool(fn_name, fn_args, agent=agent)

                        # Truncate output to protect context window limit
                        max_out = 3500
                        if len(tool_output) > max_out:
                            tool_output = tool_output[:max_out] + f"\n... [Truncated {len(tool_output) - max_out} bytes]"

                        if on_event:
                            try:
                                res = on_event("tool_result", {"name": fn_name, "result": tool_output, "id": tc_id})
                                if asyncio.iscoroutine(res):
                                    await res
                            except Exception:
                                pass

                        current_messages.append({
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "name": fn_name,
                            "content": tool_output,
                        })

            except Exception as e:
                logger.error("Tool loop exception for agent %s: %s", agent.name, e)
                return f"⚠️ Error in agent processing: {e}", current_messages

        return "⚠️ Agent exceeded maximum tool execution iterations.", current_messages
