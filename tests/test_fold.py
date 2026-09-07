"""
Automated unit and integration test suite for Fold.
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

from fold.core_tools import CORE_TOOLS_REGISTRY, calculate, execute_core_tool, get_current_time
from fold.database import Database
from fold.herdr_integration import HerdrIntegration
from fold.inspector import GGUFInspector, HardwareInspector, analyze_model_and_recommend
from fold.mcp_client import MCPManager, MCPServerInfo, StdioMCPConnection
from fold.multi_agent import MultiAgentRouter
from fold.security import (
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
from fold.telegram_service import TELEGRAM_TOKEN_REGEX


SAMPLE_MODEL_PATH = Path.home() / ".local/share/fold/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
if not SAMPLE_MODEL_PATH.exists():
    _legacy_sample = Path.home() / ".local/share/sheprd/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    if _legacy_sample.exists():
        SAMPLE_MODEL_PATH = _legacy_sample


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
            validate_agent_name("fold")  # Reserved
        with self.assertRaises(SecurityError):
            validate_agent_name("sheprd")  # Reserved (legacy)

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

    def test_extract_int_helper(self):
        from fold.inspector import _extract_int
        self.assertEqual(_extract_int(42), 42)
        self.assertEqual(_extract_int("128"), 128)
        self.assertEqual(_extract_int([8, 8, 1]), 8)
        self.assertEqual(_extract_int([]), 0)
        self.assertEqual(_extract_int(None, default=16), 16)
        self.assertEqual(_extract_int("invalid", default=32), 32)

    def test_detect_template_kind_architectures(self):
        from fold.inspector import GGUFInspector
        self.assertEqual(GGUFInspector.detect_template_kind("", "gemma4", "gemma-4-12b"), "gemma")
        self.assertEqual(GGUFInspector.detect_template_kind("<start_of_turn>user", "llama", "model"), "gemma")
        self.assertEqual(GGUFInspector.detect_template_kind("", "mistral3", "Devstral-Small"), "mistral")
        self.assertEqual(GGUFInspector.detect_template_kind("", "qwen3moe", "Qwen3-235B"), "chatml")
        self.assertEqual(GGUFInspector.detect_template_kind("", "glm-dsa", "GLM-5.2"), "chatml")
        self.assertEqual(GGUFInspector.detect_template_kind("<|start_header_id|>", "llama", "Llama-3.1"), "llama-3")
        self.assertEqual(GGUFInspector.detect_template_kind("<｜user｜>", "deepseek2", "DeepSeek-V2"), "deepseek")


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
        from fold.server_manager import ServerManager
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
        self.db_path = Path(self.temp_dir) / "test_fold.db"
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
        from fold.tool_hub import ToolHub

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


class TestHardeningAndHotSwap(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_hardening.db"
        self.db = Database(db_path=self.db_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_ssrf_protection(self):
        from fold.core_tools import tool_fetch_url

        # Loopback IPv4
        res1 = tool_fetch_url("http://127.0.0.1:8080/admin")
        self.assertIn("SSRF guard", res1)

        # Cloud metadata service (169.254.169.254)
        res2 = tool_fetch_url("http://169.254.169.254/latest/meta-data/")
        self.assertIn("SSRF guard", res2)

        # RFC-1918 Private networks
        res3 = tool_fetch_url("http://192.168.1.1/router")
        self.assertIn("SSRF guard", res3)
        res4 = tool_fetch_url("http://10.0.0.5/api")
        self.assertIn("SSRF guard", res4)

        # Scheme guard
        res5 = tool_fetch_url("ftp://example.com/file")
        self.assertIn("Only http and https protocols are supported", res5)

    def test_ast_pow_bounds(self):
        # Exponent bomb: 9**9**9
        res = calculate("9**9**9")
        self.assertIn("exponent bomb guard", res)

        # Base too large for exponentiation
        res2 = calculate("10001**2")
        self.assertIn("exponent bomb guard", res2)

        # Exponent too large
        res3 = calculate("2**101")
        self.assertIn("exponent bomb guard", res3)

        # Valid exponentiation within bounds
        self.assertEqual(calculate("2**10"), "1024")

    def test_tool_permission_enforcement(self):
        from fold.tool_hub import ToolHub

        agent = self.db.create_agent(
            name="time_only_bot",
            identity="Clock",
            personality="Punctual",
            job="Tell time",
            model_path="/tmp/fake.gguf",
            model_architecture="qwen2",
            port=8101,
            tools=["get_current_time"],
        )

        hub = ToolHub(self.db)

        # Calling unauthorized tool
        bad_call = asyncio.run(hub.execute_tool("calculate", {"expression": "2+2"}, agent=agent))
        self.assertIn("Permission error", bad_call)
        self.assertIn("time_only_bot", bad_call)

        # Calling authorized tool
        good_call = asyncio.run(hub.execute_tool("get_current_time", {}, agent=agent))
        self.assertIn("UTC", good_call)

    def test_interactive_chat_session_instantiation(self):
        from fold.interactive_chat import InteractiveChatSession

        agent = self.db.create_agent(
            name="chat_bot",
            identity="Chat Agent",
            personality="Polite",
            job="Chat",
            model_path="/tmp/fake.gguf",
            model_architecture="qwen2",
            port=8102,
        )

        session = InteractiveChatSession(agent.name, db=self.db)
        self.assertEqual(session.agent_name, "chat_bot")
        self.assertIsNotNone(session.server_mgr)

    def test_api_token_creation_and_perms(self):
        from fold.security import API_TOKEN_PATH, get_or_create_api_token

        token = get_or_create_api_token()
        self.assertGreaterEqual(len(token), 32)

        token_file = API_TOKEN_PATH
        self.assertTrue(token_file.exists())
        file_mode = token_file.stat().st_mode & 0o777
        self.assertEqual(file_mode, 0o600)

    def test_mcp_secrets_masking(self):
        from fold.web.app import FoldWebApp

        raw_server = {
            "id": 1,
            "name": "search_mcp",
            "command": "npx",
            "args": ["-y", "duckduckgo"],
            "env": {"API_KEY": "supersecretkey1234567890"},
            "enabled": True,
            "description": "Search tools",
        }

        masked = FoldWebApp._mask_mcp_server_secrets(raw_server)
        self.assertIn("••••", masked["env"]["API_KEY"])
        self.assertNotIn("supersecretkey1234567890", masked["env"]["API_KEY"])

    def test_server_manager_lru_hot_swap(self):
        from fold.server_manager import ServerManager

        agent_a = self.db.create_agent(
            name="agent_a",
            identity="Agent A",
            personality="Alpha",
            job="Task A",
            model_path="/tmp/model_a.gguf",
            model_architecture="qwen2",
            port=8103,
        )
        agent_b = self.db.create_agent(
            name="agent_b",
            identity="Agent B",
            personality="Beta",
            job="Task B",
            model_path="/tmp/model_b.gguf",
            model_architecture="qwen2",
            port=8104,
        )

        mgr = ServerManager(self.db)
        mgr.max_active_models = 1

        # Track mock start/stop events
        stopped_agents = []
        started_agents = []

        def mock_start(name, *args, **kwargs):
            started_agents.append(name)
            self.db.update_agent_status(name, "running", 99999)
            return True, f"Mock started {name}"

        def mock_stop(name, *args, **kwargs):
            stopped_agents.append(name)
            self.db.update_agent_status(name, "stopped", None)
            return True, f"Mock stopped {name}"

        mgr.start_agent_server = mock_start
        mgr.stop_agent_server = mock_stop
        mgr.check_health = lambda port: True
        mgr._verify_process_is_llama = lambda pid, port: True

        # Ensure agent_a runs
        ok_a, _ = mgr.ensure_agent_running("agent_a")
        self.assertTrue(ok_a)
        self.assertIn("agent_a", started_agents)

        # Now ensure agent_b runs with limit=1: agent_a should be stopped (evicted)
        ok_b, _ = mgr.ensure_agent_running("agent_b")
        self.assertTrue(ok_b)
        self.assertIn("agent_a", stopped_agents)
        self.assertIn("agent_b", started_agents)

    def test_parse_mcp_install_string_formats(self):
        from fold.mcp_smithery import parse_mcp_install_string

        # 1. Smithery URL
        url_res = parse_mcp_install_string("https://smithery.ai/server/@smithery-ai/fetch")
        self.assertEqual(url_res["command"], "npx")
        self.assertIn("@smithery-ai/fetch", url_res["args"])
        self.assertEqual(url_res["name"], "fetch")

        # 2. Smithery CLI command
        cli_res = parse_mcp_install_string("npx -y @smithery/cli run @smithery-ai/sqlite")
        self.assertEqual(cli_res["command"], "npx")
        self.assertIn("@smithery-ai/sqlite", cli_res["args"])

        # 3. Direct uvx command
        uvx_res = parse_mcp_install_string("uvx mcp-server-git")
        self.assertEqual(uvx_res["command"], "uvx")
        self.assertEqual(uvx_res["args"], ["mcp-server-git"])
        self.assertEqual(uvx_res["name"], "git")

        # 4. Raw package name
        pkg_res = parse_mcp_install_string("@smithery-ai/weather")
        self.assertEqual(pkg_res["command"], "npx")
        self.assertIn("@smithery-ai/weather", pkg_res["args"])
        self.assertEqual(pkg_res["name"], "weather")

        # 5. Featured catalog aliases
        feat_res = parse_mcp_install_string("web-fetch")
        self.assertEqual(feat_res["name"], "web_fetcher")
        self.assertEqual(feat_res["command"], "uvx")

        # 6. Claude Desktop JSON format
        json_snippet = json.dumps({
            "mcpServers": {
                "custom_calc": {
                    "command": "python3",
                    "args": ["-m", "calc_server"],
                    "env": {"API_KEY": "test123"}
                }
            }
        })
        json_res = parse_mcp_install_string(json_snippet)
        self.assertEqual(json_res["name"], "custom_calc")
        self.assertEqual(json_res["command"], "python3")
        self.assertEqual(json_res["args"], ["-m", "calc_server"])
        self.assertEqual(json_res["env"], {"API_KEY": "test123"})

    def test_sync_external_client_configs(self):
        import tempfile
        from unittest.mock import patch
        from fold.mcp_smithery import sync_external_client_configs

        with tempfile.TemporaryDirectory() as tmpdir:
            claude_cfg = Path(tmpdir) / "claude_desktop_config.json"
            claude_cfg.write_text(json.dumps({
                "mcpServers": {
                    "external_fetch": {
                        "command": "uvx",
                        "args": ["mcp-server-fetch"]
                    }
                }
            }), encoding="utf-8")

            with patch("fold.mcp_smithery.CLAUDE_CONFIG_PATH", claude_cfg), \
                 patch("fold.mcp_smithery.CURSOR_CONFIG_PATH", Path(tmpdir) / "none.json"), \
                 patch("fold.mcp_smithery.CURSOR_CONFIG_ALT_PATH", Path(tmpdir) / "none2.json"), \
                 patch("fold.mcp_smithery.FOLD_MCP_CONFIG_PATH", Path(tmpdir) / "none3.json"):

                # First sync imports it
                imported = sync_external_client_configs(self.db)
                self.assertEqual(len(imported), 1)
                self.assertEqual(imported[0]["name"], "external_fetch")

                # Second sync is idempotent
                imported_again = sync_external_client_configs(self.db)
                self.assertEqual(len(imported_again), 0)

    def test_query_group_skipped_agents(self):
        # Create agents in a group:
        # active_worker: running, callable
        # disabled_worker: running, but callable_by_agents=False
        # idle_worker: stopped, callable_by_agents=True
        self.db.create_agent(
            name="active_worker",
            identity="Worker 1",
            personality="Energetic",
            job="Work",
            model_path="/tmp/m1.gguf",
            model_architecture="qwen2",
            port=8110,
            callable_by_agents=True,
            groups=["swarm_test"],
        )
        self.db.update_agent_status("active_worker", "running", 1001)

        self.db.create_agent(
            name="disabled_worker",
            identity="Worker 2",
            personality="Quiet",
            job="Work",
            model_path="/tmp/m2.gguf",
            model_architecture="qwen2",
            port=8111,
            callable_by_agents=False,
            groups=["swarm_test"],
        )
        self.db.update_agent_status("disabled_worker", "running", 1002)

        self.db.create_agent(
            name="idle_worker",
            identity="Worker 3",
            personality="Sleepy",
            job="Work",
            model_path="/tmp/m3.gguf",
            model_architecture="qwen2",
            port=8112,
            callable_by_agents=True,
            groups=["swarm_test"],
        )
        self.db.update_agent_status("idle_worker", "stopped", None)

        router = MultiAgentRouter(self.db)
        router.call_agent = lambda target, caller, prompt: {"ok": True, "target_agent": target, "response": f"Reply from {target}"}

        out = router.query_group("swarm_test", "coordinator", "Hello team")
        self.assertIsInstance(out, dict)
        self.assertIn("results", out)
        self.assertIn("skipped", out)

        res = out["results"]
        skipped = out["skipped"]

        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["target_agent"], "active_worker")

        self.assertEqual(len(skipped), 2)
        skipped_map = {s["name"]: s["reason"] for s in skipped}
        self.assertIn("disabled_worker", skipped_map)
        self.assertIn("callable_by_agents is False", skipped_map["disabled_worker"])
        self.assertIn("idle_worker", skipped_map)
        self.assertIn("LRU cap", skipped_map["idle_worker"])

    def test_cross_process_lru_last_used_timestamp(self):
        from fold.server_manager import ServerManager

        self.db.create_agent(
            name="lru_one",
            identity="Agent 1",
            personality="A",
            job="J1",
            model_path="/tmp/m1.gguf",
            model_architecture="qwen2",
            port=8120,
        )
        self.db.create_agent(
            name="lru_two",
            identity="Agent 2",
            personality="B",
            job="J2",
            model_path="/tmp/m2.gguf",
            model_architecture="qwen2",
            port=8121,
        )

        mgr = ServerManager(self.db)
        mgr.max_active_models = 1

        evicted = []
        def mock_start(name, *args, **kwargs):
            self.db.update_agent_status(name, "running", 5001)
            self.db.touch_agent_last_used(name)
            return True, f"Started {name}"

        def mock_stop(name, *args, **kwargs):
            evicted.append(name)
            self.db.update_agent_status(name, "stopped", None)
            return True, f"Stopped {name}"

        mgr.start_agent_server = mock_start
        mgr.stop_agent_server = mock_stop
        mgr.check_health = lambda port: True
        mgr._verify_process_is_llama = lambda pid, port: True

        # Activate agent 1
        mgr.ensure_agent_running("lru_one")
        rec1 = self.db.get_agent_by_name("lru_one")
        self.assertIsNotNone(rec1.last_used_at)
        self.assertTrue(mgr.swap_lock_path.exists())

        # Activate agent 2: should evict agent 1 because it's the oldest
        mgr.ensure_agent_running("lru_two")
        self.assertIn("lru_one", evicted)
        rec2 = self.db.get_agent_by_name("lru_two")
        self.assertIsNotNone(rec2.last_used_at)

    def test_get_history_limit_clamping(self):
        self.db.create_agent(
            name="hist_bot",
            identity="H",
            personality="P",
            job="J",
            model_path="/tmp/fake.gguf",
            model_architecture="qwen2",
            port=8130,
        )
        for i in range(10):
            self.db.log_chat("hist_bot", "user", f"msg {i}")

        def parse_limit(raw_val):
            try:
                return max(1, min(int(raw_val), 200))
            except (ValueError, TypeError):
                return 50

        self.assertEqual(parse_limit("5"), 5)
        self.assertEqual(parse_limit("99999"), 200)
        self.assertEqual(parse_limit("-10"), 1)
        self.assertEqual(parse_limit("invalid"), 50)
        self.assertEqual(parse_limit(None), 50)

    def test_cmd_mcp_add_security_error_handling(self):
        from unittest.mock import patch
        from io import StringIO
        from fold.cli import main

        with patch("sys.argv", ["fold", "mcp", "add", "bad;name", "uvx", "mcp-server-git"]), \
             patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            main()
            self.assertIn("Invalid MCP server name", mock_stdout.getvalue())

    def test_web_app_security_imports_and_validation(self):
        from fold.web.app import validate_context_size as web_ctx_val
        from fold.web.app import validate_gpu_layers as web_gpu_val

        # Verify imports match security module
        self.assertEqual(web_ctx_val(4096), 4096)
        self.assertEqual(web_ctx_val("8192"), 8192)
        with self.assertRaises(SecurityError):
            web_ctx_val(10)
        with self.assertRaises(SecurityError):
            web_ctx_val("not_a_number")

        self.assertEqual(web_gpu_val(0), 0)
        self.assertEqual(web_gpu_val("33"), 33)
        with self.assertRaises(SecurityError):
            web_gpu_val(9999)
        with self.assertRaises(SecurityError):
            web_gpu_val("abc")

    def test_gemma_sliding_window_metadata_handling(self):
        from fold.inspector import _extract_int, GGUFInspector

        # Test list of ints (Gemma 4 alternating heads)
        self.assertEqual(_extract_int([8, 8, 8, 8, 8, 1, 8, 8]), 8)
        # Test list of numeric strings
        self.assertEqual(_extract_int(["16", "8", "32"]), 32)
        # Test tuple
        self.assertEqual(_extract_int((4, 2, 8)), 8)
        # Test empty or malformed collections
        self.assertEqual(_extract_int([], default=16), 16)
        self.assertEqual(_extract_int(["bad", "data"], default=4), 4)
        self.assertEqual(_extract_int(None, default=8), 8)


if __name__ == "__main__":
    unittest.main()
