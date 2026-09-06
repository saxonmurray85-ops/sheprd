"""
Process and lifecycle manager for llama-server instances.
Handles port allocation, isolated daemon execution, health probes,
graceful shutdown, and log auditing.
"""

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


LOGS_DIR = Path.home() / ".local/share/sheprd/logs"
DEFAULT_BIN_PATH = Path.home() / ".local/share/sheprd/bin/llama-server"


class ServerManager:
    def __init__(self, db: Database, bin_path: Optional[Path] = None):
        self.db = db
        self.bin_path = bin_path or DEFAULT_BIN_PATH
        self.logs_dir = LOGS_DIR
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._running_processes: Dict[str, subprocess.Popen] = {}

    def resolve_binary(self) -> str:
        """Finds valid llama-server binary."""
        if self.bin_path.exists() and os.access(self.bin_path, os.X_OK):
            return str(self.bin_path.resolve())

        # Fallback to PATH
        import shutil
        which_path = shutil.which("llama-server")
        if which_path and os.access(which_path, os.X_OK):
            return which_path

        raise FileNotFoundError(
            f"llama-server executable not found at '{self.bin_path}' or on PATH. "
            "Please ensure llama.cpp is installed in ~/.local/share/sheprd/bin/."
        )

    def find_free_port(self, start_port: int = 8081, max_port: int = 8999) -> int:
        """Finds an unused, available TCP port on 127.0.0.1 not allocated to any agent."""
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
        """Checks if llama-server is healthy and ready to serve requests."""
        url = f"http://127.0.0.1:{port}/health"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Sheprd-HealthChecker/1.0"})
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                return resp.status in (200, 503)  # 503 indicates loading model; 200 is ready
        except Exception:
            return False

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
                return True, f"Agent '{agent_name}' is already running on port {agent.port}."

        bin_exec = self.resolve_binary()
        model_target = validate_model_path(agent.model_path)

        log_file_path = self.logs_dir / f"{agent_name}.log"
        log_file = open(log_file_path, "a", encoding="utf-8")
        try:
            os.chmod(log_file_path, 0o600)
        except OSError:
            pass

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
        ]

        if agent.n_gpu_layers > 0:
            cmd.extend(["-ngl", str(agent.n_gpu_layers)])

        # Additional optimizations for responsiveness
        cmd.extend(["--parallel", "1", "--cont-batching"])

        # Timestamp log entry
        log_file.write(f"\n--- Sheprd starting {agent_name} at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        log_file.write(f"Command: {' '.join(cmd)}\n")
        log_file.flush()

        try:
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
            while time.time() - start_time < timeout_sec:
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
                return True, f"Agent '{agent_name}' started successfully on port {agent.port} (PID {proc.pid})."
            else:
                self.stop_agent_server(agent_name)
                return False, f"Server startup timed out after {timeout_sec}s. Check logs at {log_file_path}."

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

