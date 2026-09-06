"""
Automated unit and integration test suite for Sheprd.
Verifies security gates, GGUF parsing, database isolation,
Herdr launcher creation, and multi-agent coordination.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from sheprd.database import Database
from sheprd.herdr_integration import HerdrIntegration
from sheprd.inspector import GGUFInspector, HardwareInspector, analyze_model_and_recommend
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


if __name__ == "__main__":
    unittest.main()
