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
from .prompts import build_agent_system_prompt
from .security import SecurityError


logger = logging.getLogger("fold.multi_agent")
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

        system_prompt = build_agent_system_prompt(target, caller_name=caller_name)

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
            headers={"Content-Type": "application/json", "User-Agent": f"Fold-Agent/{caller_name}"},
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
    ) -> Dict[str, Any]:
        """
        Broadcasts a task to all callable agents belonging to a specified group.
        Returns results from active agents and explicit skipped records for
        inactive agents (due to LRU capacity) or non-callable agents.
        """
        agents = self.db.list_agents()
        group_members = [a for a in agents if group_name in a.groups]

        results = []
        skipped = []

        for member in group_members:
            if not member.callable_by_agents:
                skipped.append({
                    "name": member.name,
                    "reason": "external calls disabled (callable_by_agents is False)",
                })
                continue

            if member.status != "running":
                skipped.append({
                    "name": member.name,
                    "reason": "not active (LRU cap — raise FOLD_MAX_ACTIVE_MODELS to run concurrently)",
                })
                continue

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

        return {
            "results": results,
            "skipped": skipped,
        }

    def sync_with_agent_hub(self) -> bool:
        """
        If the local Agent-Hub database exists at ~/.agent-hub/data/memory.db,
        syncs Fold's active callable models into shared memory for discovery.
        """
        if not AGENT_HUB_DB.exists():
            return False

        try:
            agents = self.db.list_agents()
            active_agents = [
                {
                    "name": a.name,
                    "identity": a.identity,
                    "job": a.job,
                    "port": a.port,
                    "model": a.model_architecture,
                    "groups": a.groups,
                    "tools": a.tools,
                    "endpoint": f"http://127.0.0.1:{a.port}/v1",
                }
                for a in agents
                if a.status == "running" and a.callable_by_agents
            ]
            payload_json = json.dumps(active_agents)

            conn = sqlite3.connect(AGENT_HUB_DB, timeout=5.0)
            with conn:
                conn.execute(
                    """
                    INSERT INTO memories (namespace, key, value, tags, source)
                    VALUES ('fold', 'active_agents', ?, 'fold,models,local', 'fold')
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
                    VALUES ('fold_sync', 'fold', 'Fold updated active local agents', ?)
                    """,
                    (payload_json,),
                )
            conn.close()
            return True
        except Exception as e:
            logger.warning("Agent-Hub sync error: %s", e)
            return False
