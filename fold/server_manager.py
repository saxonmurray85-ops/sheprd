"""
Process and lifecycle manager for llama-server instances.
Handles port allocation, isolated daemon execution, health probes,
graceful shutdown, and log auditing.
"""

from collections import OrderedDict
import fcntl
import logging
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .database import AgentRecord, Database
from .security import SecurityError, sanitize_log_text, validate_model_path

logger = logging.getLogger("fold.server")

LEGACY_BIN_PATH = Path.home() / ".local/share/sheprd/bin/llama-server"
LOGS_DIR = Path.home() / ".local/share/fold/logs"
DEFAULT_BIN_PATH = Path.home() / ".local/share/fold/bin/llama-server"


class ServerManager:
    def __init__(self, db: Database, bin_path: Optional[Path] = None):
        self.db = db
        self.bin_path = bin_path or (DEFAULT_BIN_PATH if DEFAULT_BIN_PATH.exists() else (LEGACY_BIN_PATH if LEGACY_BIN_PATH.exists() else DEFAULT_BIN_PATH))
        self.logs_dir = LOGS_DIR
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.locks_dir = getattr(self.db, "db_path", Path.home() / ".local/share/fold/fold.db").parent / "locks"
        self.locks_dir.mkdir(parents=True, exist_ok=True)
        self.swap_lock_path = self.locks_dir / "swap.lock"
        self._running_processes: Dict[str, subprocess.Popen] = {}
        # Model Hot-Swapping configuration: maximum concurrent loaded models in RAM/VRAM
        self.max_active_models = int(os.environ.get("FOLD_MAX_ACTIVE_MODELS") or os.environ.get("SHEPRD_MAX_ACTIVE_MODELS", "1"))
        self._active_lru: OrderedDict[str, int] = OrderedDict()

    def resolve_binary(self) -> str:
        """Finds valid llama-server binary."""
        candidates = [
            self.bin_path,
            LEGACY_BIN_PATH,
            Path("/opt/llama.cpp/build/bin/llama-server"),
            Path("/opt/squire-llama/llama-server"),
            Path.home() / ".local/bin/llama-server",
            Path("/usr/local/bin/llama-server"),
            Path("/usr/bin/llama-server"),
        ]
        for cand in candidates:
            if cand and cand.exists() and os.access(cand, os.X_OK):
                return str(cand.resolve())

        # Fallback to PATH
        import shutil
        which_path = shutil.which("llama-server")
        if which_path and os.access(which_path, os.X_OK):
            return which_path

        raise FileNotFoundError(
            f"llama-server executable not found at '{self.bin_path}' or on PATH. "
            "Please ensure llama.cpp is installed in ~/.local/share/fold/bin/ or /opt/llama.cpp/build/bin/."
        )

    def find_free_port(self, start_port: int = 8081, max_port: int = 9000) -> int:
        """Finds an unused, available TCP port on 127.0.0.1 not allocated to any agent (B8)."""
        allocated = set(self.db.get_allocated_ports())
        for port in range(start_port, max_port):
            if port in allocated:
                continue
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("127.0.0.1", port))
                    return port
                except OSError:
                    continue
        raise RuntimeError("No available ports found in range 8081-8999.")

    def is_port_in_use(self, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return False
            except OSError:
                return True

    def check_health(self, port: int, timeout_sec: float = 1.0) -> bool:
        """Checks if llama-server is healthy and ready to serve requests (HTTP 200) (S9 / N14)."""
        url = f"http://127.0.0.1:{port}/health"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Fold-HealthChecker/1.0"})
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                # Strictly HTTP 200 (503 means model is still loading and not ready)
                return resp.status == 200
        except Exception:
            return False

    def _touch_lru(self, agent_name: str) -> None:
        if agent_name in self._active_lru:
            self._active_lru.move_to_end(agent_name)
        else:
            self._active_lru[agent_name] = int(time.time())

    def get_active_agents(self) -> List[str]:
        """Reconciles and returns list of actually running, healthy agent names."""
        active = []
        for agent in self.db.list_agents():
            if agent.pid and self._verify_process_is_llama(agent.pid, agent.port) and self.check_health(agent.port):
                active.append(agent.name)
                self._touch_lru(agent.name)
            elif agent.status == "running":
                self.db.update_agent_status(agent.name, "stopped", None)
        return active

    def ensure_agent_running(self, agent_name: str, timeout_sec: int = 45) -> Tuple[bool, str]:
        """
        Hot-Swapping Engine: Ensures that the requested agent's model is loaded and ready.
        If loading this agent would exceed max_active_models (default: 1), gracefully
        evicts the least recently used running agent model to free GPU VRAM and system memory.
        Uses fcntl file locking for cross-process mutual exclusion and persists last_used_at.
        """
        with open(self.swap_lock_path, "a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                agent = self.db.get_agent_by_name(agent_name)
                if not agent:
                    return False, f"Agent '{agent_name}' not found."

                # 1. Check if already healthy
                if agent.pid and self._verify_process_is_llama(agent.pid, agent.port) and self.check_health(agent.port):
                    self._touch_lru(agent_name)
                    self.db.touch_agent_last_used(agent_name)
                    return True, f"Agent '{agent_name}' is already running and ready."

                # 2. Reconcile currently active models
                running_agents = self.get_active_agents()
                if agent_name in running_agents:
                    running_agents.remove(agent_name)

                # 3. Evict oldest model(s) if at or above capacity limit
                while len(running_agents) >= self.max_active_models:
                    candidates = []
                    for r_name in running_agents:
                        r_agent = self.db.get_agent_by_name(r_name)
                        ts = (r_agent.last_used_at or r_agent.updated_at or r_agent.created_at or "") if r_agent else ""
                        candidates.append((ts, r_name))
                    candidates.sort(key=lambda x: x[0])
                    evict_candidate = candidates[0][1] if candidates else None

                    if evict_candidate:
                        logger.info(
                            "[HOT-SWAP] Evicting model for agent '%s' to free VRAM/RAM for '%s' (max active: %d)...",
                            evict_candidate, agent_name, self.max_active_models
                        )
                        self.stop_agent_server(evict_candidate)
                        self._active_lru.pop(evict_candidate, None)
                        if evict_candidate in running_agents:
                            running_agents.remove(evict_candidate)
                    else:
                        break

                # 4. Start the target agent
                ok, msg = self.start_agent_server(agent_name, timeout_sec=timeout_sec)
                if ok:
                    self._touch_lru(agent_name)
                    self.db.touch_agent_last_used(agent_name)
                    logger.info("[HOT-SWAP] Successfully loaded model for agent '%s' on port %d.", agent_name, agent.port)
                    return True, f"Model swapped: agent '{agent_name}' is now active."
                return False, msg
            finally:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass

    def start_agent_server(self, agent_name: str, timeout_sec: int = 30) -> Tuple[bool, str]:
        """
        Starts the llama-server for the specified agent.
        Runs safely bound strictly to localhost (127.0.0.1).
        """
        agent = self.db.get_agent_by_name(agent_name)
        if not agent:
            return False, f"Agent '{agent_name}' not found."

        # Check if already running
        if agent.pid and self._is_pid_alive(agent.pid):
            if self.check_health(agent.port):
                self.db.update_agent_status(agent_name, "running", agent.pid)
                self.db.touch_agent_last_used(agent_name)
                return True, f"Agent '{agent_name}' is already running on port {agent.port}."

        log_file_path = self.logs_dir / f"{agent_name}.log"
        try:
            # Pre-flight binary and model resolution safely caught (B7a / N21)
            bin_exec = self.resolve_binary()
            model_target = validate_model_path(agent.model_path)

            log_file = open(log_file_path, "a", encoding="utf-8")
            try:
                os.chmod(log_file_path, 0o600)
            except OSError:
                pass

            # Calculate dynamic startup timeout based on model size (larger models take longer to allocate)
            try:
                model_size_gb = model_target.stat().st_size / (1024 ** 3)
            except Exception:
                model_size_gb = 5.0

            effective_timeout = max(timeout_sec, 45)
            if model_size_gb > 15:
                effective_timeout = max(effective_timeout, 90)
            if model_size_gb > 30:
                effective_timeout = max(effective_timeout, 180)
            if model_size_gb > 70:
                effective_timeout = max(effective_timeout, 300)

            # Command arguments list (no shell=True for strict command injection defense)
            cmd = [
                bin_exec,
                "-m", str(model_target),
                "-c", str(agent.context_size),
                "--port", str(agent.port),
                "--host", "127.0.0.1",
                "-t", str(agent.threads),
                "-b", "2048",
                "-ub", "512",
                "--jinja",
            ]

            # Chat templating: modern llama.cpp uses each model's embedded
            # tokenizer.chat_template via minja. Do NOT pass --chat-template:
            # named templates ("gemma", "chatml", ...) are no longer resolved
            # and the literal string becomes the template, corrupting prompts.

            if agent.n_gpu_layers > 0:
                cmd.extend(["-ngl", str(agent.n_gpu_layers)])

            # Support --no-mmap for unified memory architectures (ROCm) or huge models
            if os.environ.get("FOLD_NO_MMAP") == "1" or model_size_gb > 40:
                cmd.append("--no-mmap")

            # Additional optimizations for responsiveness
            cmd.extend(["--parallel", "1", "--cont-batching"])

            # Timestamp log entry
            log_file.write(f"\n--- Fold starting {agent_name} at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            log_file.write(f"Command: {' '.join(cmd)}\n")
            log_file.flush()

            proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,  # Create detached process group for clean kill
                close_fds=True,
            )
            self._running_processes[agent_name] = proc
            self.db.update_agent_status(agent_name, "starting", proc.pid)

            # Wait for server readiness
            start_time = time.time()
            ready = False
            while time.time() - start_time < effective_timeout:
                if proc.poll() is not None:
                    # Process died early
                    self.db.update_agent_status(agent_name, "error", None)
                    return False, f"llama-server exited prematurely with code {proc.returncode}. Check logs at {log_file_path}."

                if self.check_health(agent.port):
                    ready = True
                    break
                time.sleep(0.5)

            if ready:
                self.db.update_agent_status(agent_name, "running", proc.pid)
                self.db.touch_agent_last_used(agent_name)
                return True, f"Agent '{agent_name}' started successfully on port {agent.port} (PID {proc.pid})."
            else:
                self.stop_agent_server(agent_name)
                return False, f"Server startup timed out after {effective_timeout}s. Check logs at {log_file_path}."

        except Exception as e:
            self.db.update_agent_status(agent_name, "error", None)
            return False, f"Failed to start server: {e}"

    def stop_agent_server(self, agent_name: str) -> Tuple[bool, str]:
        """Gracefully terminates the llama-server process for an agent."""
        agent = self.db.get_agent_by_name(agent_name)
        if not agent:
            return False, f"Agent '{agent_name}' not found."

        pid = agent.pid
        proc = self._running_processes.pop(agent_name, None)

        if not pid:
            self.db.update_agent_status(agent_name, "stopped", None)
            return True, f"Agent '{agent_name}' was not running."

        # S5: Verify PID actually belongs to this agent's llama-server before sending signals
        if not self._verify_process_is_llama(pid, agent.port):
            self.db.update_agent_status(agent_name, "stopped", None)
            return True, f"Agent '{agent_name}' process {pid} is not a matching llama-server (stale PID ignored)."

        try:
            # Kill process group
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass

        # Wait up to 5 seconds for termination
        start = time.time()
        while time.time() - start < 5.0:
            if not self._is_pid_alive(pid):
                break
            time.sleep(0.3)

        # Force kill if still running
        if self._is_pid_alive(pid):
            try:
                pgid = os.getpgid(pid)
                os.killpg(pgid, signal.SIGKILL)
            except OSError:
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass

        self.db.update_agent_status(agent_name, "stopped", None)
        return True, f"Agent '{agent_name}' stopped."

    def stop_all(self) -> None:
        """Terminates all running agent servers cleanly."""
        for agent in self.db.list_agents():
            if agent.status == "running" or agent.pid:
                self.stop_agent_server(agent.name)

    def get_agent_logs(self, agent_name: str, max_lines: int = 100) -> str:
        """Reads recent server log lines with secret redaction."""
        log_file = self.logs_dir / f"{agent_name}.log"
        if not log_file.exists():
            return f"No log file found for agent '{agent_name}'."

        try:
            with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
                tail = lines[-max_lines:] if len(lines) > max_lines else lines
                return sanitize_log_text("".join(tail))
        except OSError as e:
            return f"Failed to read logs: {e}"

    @staticmethod
    def _is_pid_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    @classmethod
    def _verify_process_is_llama(cls, pid: int, expected_port: Optional[int] = None) -> bool:
        """Verifies that the given PID is genuinely a llama-server process (S5)."""
        if not cls._is_pid_alive(pid):
            return False
        cmdline_path = Path(f"/proc/{pid}/cmdline")
        if not cmdline_path.exists():
            return False
        try:
            with open(cmdline_path, "rb") as f:
                raw = f.read().decode("utf-8", errors="replace")
                if "llama-server" not in raw and "llama" not in raw:
                    return False
                if expected_port is not None and str(expected_port) not in raw:
                    return False
                return True
        except (OSError, PermissionError):
            return False

