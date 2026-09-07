"""
Herdr integration module for Sheprd.
Enables instant agent spawning by typing the agent's name inside Herdr or any shell,
and provides programmatic tab/pane spawning via Herdr's socket API.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .security import validate_agent_name


HERDR_SOCKET = Path.home() / ".config/herdr/herdr.sock"
LOCAL_BIN_DIR = Path.home() / ".local/bin"


class HerdrIntegration:
    @staticmethod
    def is_herdr_installed() -> bool:
        return shutil.which("herdr") is not None

    @staticmethod
    def is_herdr_running() -> bool:
        if not HERDR_SOCKET.exists():
            return False
        try:
            res = subprocess.run(
                ["herdr", "status", "server"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=2,
            )
            return res.returncode == 0
        except Exception:
            return False

    @classmethod
    def install_launcher(cls, agent_name: str) -> Path:
        """
        Creates an executable launcher script in ~/.local/bin/<agent_name>.
        This allows the user to simply type the agent's name in Herdr or any terminal
        to automatically launch and connect to the agent!
        """
        clean_name = validate_agent_name(agent_name)
        LOCAL_BIN_DIR.mkdir(parents=True, exist_ok=True)
        launcher_path = LOCAL_BIN_DIR / clean_name

        repo_root = Path(__file__).resolve().parent.parent
        script_content = f"""#!/usr/bin/env bash
# Auto-generated Fold launcher for '{clean_name}'
# Spawns or connects to the '{clean_name}' llama.cpp agent session.
if command -v fold >/dev/null 2>&1; then
    exec fold chat "{clean_name}" "$@"
elif command -v sheprd >/dev/null 2>&1; then
    exec sheprd chat "{clean_name}" "$@"
else
    export PYTHONPATH="{repo_root}:${{PYTHONPATH:-}}"
    exec /usr/bin/python3 -m fold.interactive_chat "{clean_name}" "$@"
fi
"""

        with open(launcher_path, "w", encoding="utf-8") as f:
            f.write(script_content)

        # Make executable
        launcher_path.chmod(0o755)
        return launcher_path

    @classmethod
    def remove_launcher(cls, agent_name: str) -> bool:
        """Removes the launcher script from ~/.local/bin/."""
        clean_name = validate_agent_name(agent_name)
        launcher_path = LOCAL_BIN_DIR / clean_name
        if launcher_path.exists():
            try:
                launcher_path.unlink()
                return True
            except OSError:
                return False
        return False

    @classmethod
    def spawn_in_herdr(cls, agent_name: str, prefer_tab: bool = True) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Uses Herdr CLI to create a new tab or split a pane and run the agent command.
        """
        clean_name = validate_agent_name(agent_name)
        if not cls.is_herdr_installed():
            return False, "Herdr executable ('herdr') not found on system PATH.", {}

        if not cls.is_herdr_running():
            return False, "Herdr server is not currently running. Launch Herdr first.", {}

        cls.install_launcher(clean_name)

        try:
            # 1. Create a new tab in Herdr for this agent
            if prefer_tab:
                cmd = ["herdr", "tab", "create", "--label", clean_name]
                proc = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                    timeout=5,
                )
                if proc.returncode != 0:
                    return False, f"Failed to create Herdr tab: {proc.stderr.strip()}", {}

                # Parse JSON output from Herdr
                out_data = json.loads(proc.stdout)
                result = out_data.get("result", {})
                root_pane = result.get("root_pane", {})
                pane_id = root_pane.get("pane_id")

                if not pane_id:
                    # Fallback: find active pane
                    pane_id = cls._get_current_or_first_pane()

                if pane_id:
                    # Run the agent in the new tab's root pane
                    run_cmd = ["herdr", "pane", "run", pane_id, clean_name]
                    subprocess.run(run_cmd, check=False, timeout=5)
                    return True, f"Agent '{clean_name}' spawned in Herdr tab.", {"pane_id": pane_id}

                return True, f"Created Herdr tab for '{clean_name}'.", result

            else:
                # Split pane
                split_cmd = ["herdr", "pane", "split", "--direction", "right"]
                proc = subprocess.run(
                    split_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                    timeout=5,
                )
                if proc.returncode != 0:
                    return False, f"Failed to split Herdr pane: {proc.stderr.strip()}", {}

                out_data = json.loads(proc.stdout)
                pane_id = out_data.get("result", {}).get("pane", {}).get("pane_id")
                if pane_id:
                    subprocess.run(["herdr", "pane", "run", pane_id, clean_name], check=False, timeout=5)
                    return True, f"Agent '{clean_name}' spawned in Herdr split pane {pane_id}.", {"pane_id": pane_id}

                return True, f"Agent '{clean_name}' split in Herdr.", {}

        except Exception as e:
            return False, f"Error communicating with Herdr: {e}", {}

    @classmethod
    def report_pane_status(cls, pane_id: str, agent_name: str, role: str, state: str) -> None:
        """
        Reports agent state to Herdr's UI borders, status indicators, and tabs.
        state: 'idle', 'working', 'blocked', 'unknown'
        """
        if not pane_id or not cls.is_herdr_running():
            return

        try:
            # Report display metadata
            subprocess.run(
                [
                    "herdr", "pane", "report-metadata",
                    "--source", "fold",
                    "--display-agent", agent_name,
                    "--title", f"Fold: {agent_name} [{role}]",
                    pane_id,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=1,
            )

            # Report agent lifecycle status
            subprocess.run(
                [
                    "herdr", "pane", "report-agent",
                    "--source", "fold",
                    "--agent", agent_name,
                    "--state", state,
                    pane_id,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=1,
            )
        except Exception:
            pass

    @classmethod
    def _get_current_or_first_pane(cls) -> Optional[str]:
        try:
            proc = subprocess.run(
                ["herdr", "pane", "list"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=3,
            )
            if proc.returncode == 0:
                data = json.loads(proc.stdout)
                panes = data.get("result", {}).get("panes", [])
                if panes:
                    return panes[-1].get("pane_id")
        except Exception:
            pass
        return None
