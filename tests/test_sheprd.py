"""
Automated unit and integration test suite for Sheprd.
Verifies security gates, GGUF parsing, database isolation,
Herdr launcher creation, and multi-agent coordination.
"""

import asyncio
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from sheprd.core_tools import CORE_TOOLS_REGISTRY, calculate, execute_core_tool, get_current_time
from sheprd.database import Database
from sheprd.herdr_integration import HerdrIntegration
from sheprd.inspector import GGUFInspector, HardwareInspector, analyze_model_and_recommend
from sheprd.mcp_client import MCPManager, MCPServerInfo, StdioMCPConnection
from sheprd.multi_agent import MultiAgentRouter
from sheprd.security import (
    SecurityError,
    mask_token,
    sanitize_log_text,
    validate_agent_name,
    validate_context_size,
    validate_gpu_layers,
    validate_group_name,
    validate_groups,
    validate_model_path,
    validate_port,
)
from sheprd.telegram_service import TELEGRAM_TOKEN_REGEX


SAMPLE_MODEL_PATH = Path.home() / ".local/share/sheprd/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"


class TestSecurityModule(unittest.TestCase):
    def test_agent_name_validation(self):
        self.assertEqual(validate_agent_name("coder-1"), "coder-1")
        self.assertEqual(validate_agent_name("assistant_v2"), "assistant_v2")

        with self.assertRaises(SecurityError):
            validate_agent_name("")
        with self.assertRaises(SecurityError):
            validate_agent_name("bad name with spaces")
        with self.assertRaises(SecurityError):
            validate_agent_name("bad;rm -rf /")
        with self.assertRaises(SecurityError):
            validate_agent_name("sheprd")  # Reserved

    def test_group_name_validation(self):
        self.assertEqual(validate_group_name("dev-team"), "dev-team")
        self.assertEqual(validate_groups(["dev", "research"]), ["dev", "research"])
        with self.assertRaises(SecurityError):
            validate_group_name("bad;group")
        with self.assertRaises(SecurityError):
            validate_group_name("group with spaces")

    def test_model_path_validation(self):
        if SAMPLE_MODEL_PATH.exists():
            p = validate_model_path(str(SAMPLE_MODEL_PATH))
            self.assertEqual(p, SAMPLE_MODEL_PATH.resolve())

        # S1 fix test: Test on an existing readable system file in /etc to verify directory guard
        with self.assertRaises(SecurityError) as ctx:
            validate_model_path("/etc/os-release")
        self.assertIn("Access to system directory '/etc' is forbidden", str(ctx.exception))

        with self.assertRaises(SecurityError) as ctx2:
            validate_model_path("../../../../etc/hosts")
        self.assertIn("forbidden", str(ctx2.exception).lower())

        with self.assertRaises(SecurityError):
            validate_model_path("~/.ssh/id_rsa")

    def test_token_masking(self):
        raw = "123456789:ABCdefGHIjklMNOpqrSTUvwxYZ12345"
        masked = mask_token(raw)
        self.assertTrue(masked.startswith("123456:"))
        self.assertTrue(masked.endswith("345"))
        self.assertIn("••••", masked)
        self.assertNotIn("ABCdef", masked)

    def test_log_sanitization(self):
        raw_log = "Error connecting with token 123456789:ABCdefGHIjklMNOpqrSTUvwxYZ12345 in worker"
        clean = sanitize_log_text(raw_log)
        self.assertNotIn("ABCdefGHIjkl", clean)
        self.assertIn("••••", clean)

    def test_constraint_bounds(self):
        self.assertEqual(validate_port(8081), 8081)
        with self.assertRaises(SecurityError):
            validate_port(22)
        with self.assertRaises(SecurityError):
            validate_port(70000)

        self.assertEqual(validate_context_size(8192), 8192)
        with self.assertRaises(SecurityError):
            validate_context_size(100)

        self.assertEqual(validate_gpu_layers(32), 32)
        with self.assertRaises(SecurityError):
            validate_gpu_layers(-1)


class TestGGUFInspector(unittest.TestCase):
    def test_sample_model_inspection(self):
        if not SAMPLE_MODEL_PATH.exists():
            self.skipTest("Sample model not available.")

        meta, rec = analyze_model_and_recommend(str(SAMPLE_MODEL_PATH), 8085)
        self.assertEqual(meta.architecture, "qwen2")
        self.assertEqual(meta.quantization, "MOSTLY_Q4_K_M")
        self.assertEqual(meta.layer_count, 24)
        self.assertEqual(meta.detected_template_kind, "chatml")
        self.assertEqual(rec.context_size, 8192)
        self.assertGreater(rec.n_gpu_layers, 0)
        self.assertEqual(rec.threads, 8)


class TestHardwareInspector(unittest.TestCase):
    def test_detect_hardware(self):
        hw = HardwareInspector.detect()
        self.assertGreater(hw.logical_cores, 0)
        self.assertGreater(hw.total_ram_bytes, 0)
        self.assertTrue(hw.vulkan_supported)


class TestDatabaseAndMultiAgent(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"
        self.db = Database(self.db_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_agent_crud(self):
        agent = self.db.create_agent(
            name="testbot",
            identity="Unit Test Bot",
            personality="Deterministic and robotic",
            job="Pass all assertions",
            model_path="/dummy/path.gguf" if not SAMPLE_MODEL_PATH.exists() else str(SAMPLE_MODEL_PATH),
            model_architecture="testarch",
            port=8099,
            context_size=4096,
            n_gpu_layers=10,
            callable_by_agents=True,
            groups=["qa", "ci"],
        )
        self.assertEqual(agent.name, "testbot")
        self.assertIn("qa", agent.groups)

        retrieved = self.db.get_agent_by_name("testbot")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.port, 8099)

        # Update
        updated = self.db.update_agent("testbot", personality="Playful and fast")
        self.assertEqual(updated.personality, "Playful and fast")

        # Delete
        self.assertTrue(self.db.delete_agent("testbot"))
        self.assertIsNone(self.db.get_agent_by_name("testbot"))

    def test_callable_permission(self):
        self.db.create_agent(
            name="private_agent",
            identity="Secret Agent",
            personality="Quiet",
            job="Hidden work",
            model_path=str(SAMPLE_MODEL_PATH) if SAMPLE_MODEL_PATH.exists() else "/tmp/dummy.gguf",
            model_architecture="qwen2",
            port=8098,
            callable_by_agents=False,
        )

        router = MultiAgentRouter(self.db)
        with self.assertRaises(PermissionError):
            router.call_agent("private_agent", "peer", "hello")

    def test_recursion_depth_guard(self):
        router = MultiAgentRouter(self.db)
        with self.assertRaises(RecursionError):
            router.call_agent("private_agent", "peer", "loop", depth=3, max_depth=3)

    def test_pid_verification(self):
        from sheprd.server_manager import ServerManager
        # PID 1 is systemd/init, definitely not a llama-server
        self.assertFalse(ServerManager._verify_process_is_llama(1, 8081))
        self.assertFalse(ServerManager._verify_process_is_llama(-999, 8081))


class TestHerdrLauncher(unittest.TestCase):
    def test_launcher_generation_and_cleanup(self):
        agent_name = "test_launcher_bot"
        launcher_path = HerdrIntegration.install_launcher(agent_name)
        self.assertTrue(launcher_path.exists())
        self.assertTrue(os.access(launcher_path, os.X_OK))

        removed = HerdrIntegration.remove_launcher(agent_name)
        self.assertTrue(removed)
        self.assertFalse(launcher_path.exists())


class TestToolsAndMCP(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_sheprd.db"
        self.db = Database(db_path=self.db_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_core_calculator(self):
        self.assertEqual(calculate("2 + 2 * 10"), "22")
        self.assertEqual(calculate("(100 - 20) / 4"), "20.0")
        self.assertIn("error", calculate("__import__('os').system('ls')").lower())
        self.assertIn("error", calculate("invalid ++ syntax").lower())

    def test_core_get_current_time(self):
        t = get_current_time()
        self.assertIn("UTC", t)

    def test_execute_core_tool(self):
        res = execute_core_tool("calculate", {"expression": "5 * 5"})
        self.assertEqual(res, "25")
        bad = execute_core_tool("nonexistent_tool", {})
        self.assertIn("Unknown tool", bad)

    def test_mock_stdio_mcp_connection(self):
        mock_py = (
            "import sys, json\n"
            "for line in sys.stdin:\n"
            "    line = line.strip()\n"
            "    if not line: continue\n"
            "    req = json.loads(line)\n"
            "    m = req.get('method')\n"
            "    rid = req.get('id')\n"
            "    if m == 'initialize':\n"
            "        res = {'jsonrpc': '2.0', 'id': rid, 'result': {'protocolVersion': '2024-11-05', 'capabilities': {'tools': {}}, 'serverInfo': {'name': 'mock-mcp', 'version': '1.0'}}}\n"
            "        sys.stdout.write(json.dumps(res) + '\\n')\n"
            "        sys.stdout.flush()\n"
            "    elif m == 'notifications/initialized':\n"
            "        pass\n"
            "    elif m == 'tools/list':\n"
            "        res = {'jsonrpc': '2.0', 'id': rid, 'result': {'tools': [{'name': 'ping', 'description': 'returns pong', 'inputSchema': {'type': 'object', 'properties': {'msg': {'type': 'string'}}}}]}}\n"
            "        sys.stdout.write(json.dumps(res) + '\\n')\n"
            "        sys.stdout.flush()\n"
            "    elif m == 'tools/call':\n"
            "        args = req.get('params', {}).get('arguments', {})\n"
            "        res = {'jsonrpc': '2.0', 'id': rid, 'result': {'content': [{'type': 'text', 'text': 'pong: ' + args.get('msg', '')}]}}\n"
            "        sys.stdout.write(json.dumps(res) + '\\n')\n"
            "        sys.stdout.flush()\n"
        )
        conn = StdioMCPConnection("test_server", sys.executable, ["-c", mock_py])
        try:
            started = conn.start(timeout_sec=5.0)
            self.assertTrue(started)
            self.assertEqual(len(conn.tools), 1)
            self.assertEqual(conn.tools[0]["namespaced_name"], "mcp__test_server__ping")

            call_res = conn.call_tool("ping", {"msg": "hello"})
            self.assertEqual(call_res, "pong: hello")
        finally:
            conn.stop()

    def test_database_mcp_servers(self):
        srv = self.db.create_mcp_server(
            name="ddg",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-duckduckgo"],
            description="DuckDuckGo Search",
        )
        self.assertEqual(srv["name"], "ddg")
        self.assertTrue(srv["enabled"])

        servers = self.db.list_mcp_servers()
        self.assertEqual(len(servers), 1)
        self.assertEqual(servers[0]["name"], "ddg")

        updated = self.db.update_mcp_server("ddg", description="Updated description")
        self.assertEqual(updated["description"], "Updated description")

        deleted = self.db.delete_mcp_server("ddg")
        self.assertTrue(deleted)
        self.assertEqual(len(self.db.list_mcp_servers()), 0)

    def test_agent_tools_persistence_and_routing(self):
        from sheprd.tool_hub import ToolHub

        agent = self.db.create_agent(
            name="toolbot",
            identity="Tool Specialist",
            personality="Efficient",
            job="Testing tools",
            model_path="/tmp/fake.gguf",
            model_architecture="qwen2",
            port=8097,
            tools=["calculate", "get_current_time"],
        )
        self.assertIn("calculate", agent.tools)
        self.assertIn("get_current_time", agent.tools)

        retrieved = self.db.get_agent_by_name("toolbot")
        self.assertEqual(retrieved.tools, ["calculate", "get_current_time"])

        hub = ToolHub(self.db)
        catalog = hub.get_available_tools_catalog()
        self.assertTrue(any(t["name"] == "calculate" for t in catalog))
        self.assertTrue(any(t["name"] == "get_weather" for t in catalog))

        agent_defns = hub.get_tool_definitions_for_agent(retrieved)
        self.assertEqual(len(agent_defns), 2)
        defn_names = [d["function"]["name"] for d in agent_defns]
        self.assertIn("calculate", defn_names)
        self.assertIn("get_current_time", defn_names)

        # Async tool execution via hub
        res = asyncio.run(hub.execute_tool("calculate", {"expression": "12 * 12"}))
        self.assertEqual(res, "144")


if __name__ == "__main__":
    unittest.main()
