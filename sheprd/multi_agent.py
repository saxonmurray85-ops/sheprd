"""
Multi-agent routing, swarm grouping, and Agent-Hub synchronization module for Sheprd.
Enforces inter-agent call permissions, routes tasks between peer agents,
and advertises available agents to local Agent-Hub memory.
"""

import json
import logging
import sqlite3
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .database import AgentRecord, Database
from .security import SecurityError


logger = logging.getLogger("sheprd.multi_agent")
AGENT_HUB_DB = Path.home() / ".agent-hub/data/memory.db"


class MultiAgentRouter:
    def __init__(self, db: Database):
        self.db = db

    def call_agent(
        self,
        target_agent_name: str,
        caller_name: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1500,
        depth: int = 0,
        max_depth: int = 3,
    ) -> Dict[str, Any]:
        """
        Enforces 'callable_by_agents' permission, recursion bounds (B8),
        and routes prompt to target agent.
        """
        if depth >= max_depth:
            raise RecursionError(
                f"Multi-agent call recursion depth exceeded ({depth} >= {max_depth}). Mutual recursion blocked."
            )

        target = self.db.get_agent_by_name(target_agent_name)
        if not target:
            raise ValueError(f"Target agent '{target_agent_name}' not found.")

        # Permission Check
        if not target.callable_by_agents:
            raise PermissionError(
                f"Agent '{target_agent_name}' has disabled external agent calls (callable_by_agents is False)."
            )

        if target.status != "running":
            raise RuntimeError(
                f"Agent '{target_agent_name}' is not currently running (status: {target.status}). Start it first."
            )

        system_prompt = (
            f"You are {target.name}.\n"
            f"IDENTITY: {target.identity}\n"
            f"PERSONALITY: {target.personality}\n"
            f"PRIMARY JOB: {target.job}\n\n"
            f"You have received a delegated request from peer agent '{caller_name}'. "
            f"Provide an expert, professional response fulfilling your specific role."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"[From Agent {caller_name}]: {prompt}"},
        ]

        payload = {
            "model": target.name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        url = f"http://127.0.0.1:{target.port}/v1/chat/completions"
        req_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=req_data,
            headers={"Content-Type": "application/json", "User-Agent": f"Sheprd-Agent/{caller_name}"},
        )

        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                reply = result.get("choices", [{}])[0].get("message", {}).get("content", "")

                # Record in chat log
                self.db.log_chat(target.name, f"agent:{caller_name}", prompt)
                self.db.log_chat(target.name, "assistant", reply)

                return {
                    "ok": True,
                    "target_agent": target.name,
                    "caller": caller_name,
                    "response": reply,
                    "usage": result.get("usage", {}),
                }
        except Exception as e:
            raise RuntimeError(f"Error querying agent '{target_agent_name}' on port {target.port}: {e}")

    def query_group(
        self,
        group_name: str,
        caller_name: str,
        prompt: str,
    ) -> List[Dict[str, Any]]:
        """
        Broadcasts a task to all callable agents belonging to a specified group.
        """
        agents = self.db.list_agents()
        group_members = [
            a for a in agents
            if group_name in a.groups and a.callable_by_agents and a.status == "running"
        ]

        if not group_members:
            return []

        results = []
        for member in group_members:
            try:
                res = self.call_agent(member.name, caller_name, prompt)
                results.append(res)
            except Exception as e:
                results.append({
                    "ok": False,
                    "target_agent": member.name,
                    "caller": caller_name,
                    "error": str(e),
                })
        return results

    def sync_with_agent_hub(self) -> bool:
        """
        If the local Agent-Hub database exists at ~/.agent-hub/data/memory.db,
        syncs Sheprd's active callable models into shared memory for discovery.
        """
        if not AGENT_HUB_DB.exists():
            return False

        try:
            active_agents = [
                {
                    "name": a.name,
                    "identity": a.identity,
                    "job": a.job,
                    "port": a.port,
                    "endpoint": f"http://127.0.0.1:{a.port}/v1",
                    "model": Path(a.model_path).name,
                    "groups": a.groups,
                    "status": a.status,
                }
                for a in self.db.list_agents()
                if a.callable_by_agents and a.status == "running"
            ]

            payload_json = json.dumps(active_agents)

            conn = sqlite3.connect(AGENT_HUB_DB, timeout=5.0)
            with conn:
                conn.execute(
                    """
                    INSERT INTO memories (namespace, key, value, tags, source)
                    VALUES ('sheprd', 'active_agents', ?, 'sheprd,models,local', 'sheprd')
                    ON CONFLICT(namespace, key) DO UPDATE SET
                        value = excluded.value,
                        tags = excluded.tags,
                        updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now');
                    """,
                    (payload_json,),
                )
                conn.execute(
                    """
                    INSERT INTO events (kind, source, message, payload)
                    VALUES ('sheprd_sync', 'sheprd', 'Sheprd updated active local agents', ?)
                    """,
                    (payload_json,),
                )
            conn.close()
            return True
        except Exception as e:
            logger.warning("Agent-Hub sync error: %s", e)
            return False
