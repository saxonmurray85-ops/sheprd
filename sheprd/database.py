"""
Database storage module for Sheprd.
Manages agents, groups, configurations, and chat history in a secure SQLite database.
Enforces file permission 0600 to protect Telegram bot tokens and model paths.
"""

import json
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .security import SecurityError, mask_token, validate_agent_name, validate_groups, validate_port


DEFAULT_DB_PATH = Path.home() / ".local/share/sheprd/sheprd.db"


@dataclass
class AgentRecord:
    id: Optional[int]
    name: str
    identity: str
    personality: str
    job: str
    model_path: str
    model_architecture: str
    port: int
    context_size: int
    n_gpu_layers: int
    threads: int
    template_kind: str
    telegram_enabled: bool
    telegram_bot_token: Optional[str]
    callable_by_agents: bool
    groups: List[str]
    status: str
    pid: Optional[int]
    created_at: str
    updated_at: str

    def to_dict(self, mask_secrets: bool = True) -> Dict[str, Any]:
        d = asdict(self)
        if mask_secrets and d.get("telegram_bot_token"):
            d["telegram_bot_token"] = mask_token(d["telegram_bot_token"])
        return d


class Database:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _enforce_permissions(self):
        """Enforces 0600 permissions on database, WAL, and SHM files (S6)."""
        for ext in ("", "-wal", "-shm"):
            p = self.db_path.with_name(self.db_path.name + ext)
            if p.exists():
                try:
                    os.chmod(p, 0o600)
                except OSError:
                    pass

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=8000;")
        conn.execute("PRAGMA foreign_keys=ON;")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
            self._enforce_permissions()

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS agents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                identity TEXT NOT NULL,
                personality TEXT NOT NULL,
                job TEXT NOT NULL,
                model_path TEXT NOT NULL,
                model_architecture TEXT NOT NULL,
                port INTEGER UNIQUE NOT NULL,
                context_size INTEGER NOT NULL DEFAULT 8192,
                n_gpu_layers INTEGER NOT NULL DEFAULT 0,
                threads INTEGER NOT NULL DEFAULT 8,
                template_kind TEXT NOT NULL DEFAULT 'chatml',
                telegram_enabled INTEGER NOT NULL DEFAULT 0,
                telegram_bot_token TEXT DEFAULT '',
                callable_by_agents INTEGER NOT NULL DEFAULT 1,
                groups TEXT NOT NULL DEFAULT 'default',
                status TEXT NOT NULL DEFAULT 'stopped',
                pid INTEGER DEFAULT NULL,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
            );
            """)

            conn.execute("""
            CREATE TABLE IF NOT EXISTS groups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT DEFAULT '',
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
            );
            """)

            conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT NOT NULL,
                sender TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
            );
            """)

            conn.execute("CREATE INDEX IF NOT EXISTS idx_agents_name ON agents(name);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_logs_agent ON chat_logs(agent_name);")

        self._enforce_permissions()

    def create_agent(
        self,
        name: str,
        identity: str,
        personality: str,
        job: str,
        model_path: str,
        model_architecture: str,
        port: int,
        context_size: int = 8192,
        n_gpu_layers: int = 0,
        threads: int = 8,
        template_kind: str = "chatml",
        telegram_enabled: bool = False,
        telegram_bot_token: Optional[str] = None,
        callable_by_agents: bool = True,
        groups: Optional[List[str]] = None,
    ) -> AgentRecord:
        clean_name = validate_agent_name(name)
        clean_port = validate_port(port)
        clean_groups = validate_groups(groups)
        group_str = ",".join(clean_groups)

        with self._conn() as conn:
            now = datetime.now(timezone.utc).isoformat()
            cursor = conn.execute(
                """
                INSERT INTO agents (
                    name, identity, personality, job, model_path, model_architecture,
                    port, context_size, n_gpu_layers, threads, template_kind,
                    telegram_enabled, telegram_bot_token, callable_by_agents,
                    groups, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'stopped', ?, ?)
                """,
                (
                    clean_name,
                    identity.strip(),
                    personality.strip(),
                    job.strip(),
                    str(model_path),
                    model_architecture,
                    clean_port,
                    context_size,
                    n_gpu_layers,
                    threads,
                    template_kind,
                    1 if telegram_enabled else 0,
                    (telegram_bot_token or "").strip(),
                    1 if callable_by_agents else 0,
                    group_str,
                    now,
                    now,
                ),
            )
            agent_id = cursor.lastrowid

        return self.get_agent_by_id(agent_id)  # type: ignore

    def get_agent_by_id(self, agent_id: int) -> Optional[AgentRecord]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM agents WHERE id = ?", (agent_id,)).fetchone()
            return self._row_to_agent(row) if row else None

    def get_agent_by_name(self, name: str) -> Optional[AgentRecord]:
        clean_name = name.strip()
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM agents WHERE name = ?", (clean_name,)).fetchone()
            return self._row_to_agent(row) if row else None

    def list_agents(self) -> List[AgentRecord]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM agents ORDER BY id ASC").fetchall()
            return [self._row_to_agent(r) for r in rows]

    def update_agent_status(self, name: str, status: str, pid: Optional[int] = None) -> None:
        with self._conn() as conn:
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                "UPDATE agents SET status = ?, pid = ?, updated_at = ? WHERE name = ?",
                (status, pid, now, name),
            )

    def update_agent(
        self,
        name: str,
        identity: Optional[str] = None,
        personality: Optional[str] = None,
        job: Optional[str] = None,
        telegram_enabled: Optional[bool] = None,
        telegram_bot_token: Optional[str] = None,
        callable_by_agents: Optional[bool] = None,
        groups: Optional[List[str]] = None,
    ) -> Optional[AgentRecord]:
        agent = self.get_agent_by_name(name)
        if not agent:
            return None

        new_identity = identity if identity is not None else agent.identity
        new_personality = personality if personality is not None else agent.personality
        new_job = job if job is not None else agent.job
        new_tg_enabled = telegram_enabled if telegram_enabled is not None else agent.telegram_enabled
        new_tg_token = telegram_bot_token if telegram_bot_token is not None else agent.telegram_bot_token
        new_callable = callable_by_agents if callable_by_agents is not None else agent.callable_by_agents
        new_groups = ",".join(validate_groups(groups)) if groups is not None else ",".join(agent.groups)

        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE agents SET
                    identity = ?, personality = ?, job = ?,
                    telegram_enabled = ?, telegram_bot_token = ?,
                    callable_by_agents = ?, groups = ?, updated_at = ?
                WHERE name = ?
                """,
                (
                    new_identity,
                    new_personality,
                    new_job,
                    1 if new_tg_enabled else 0,
                    new_tg_token,
                    1 if new_callable else 0,
                    new_groups,
                    now,
                    name,
                ),
            )
        return self.get_agent_by_name(name)

    def delete_agent(self, name: str) -> bool:
        with self._conn() as conn:
            cursor = conn.execute("DELETE FROM agents WHERE name = ?", (name,))
            return cursor.rowcount > 0

    def get_allocated_ports(self) -> List[int]:
        with self._conn() as conn:
            rows = conn.execute("SELECT port FROM agents").fetchall()
            return [r["port"] for r in rows]

    def log_chat(self, agent_name: str, sender: str, content: str) -> None:
        with self._conn() as conn:
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                "INSERT INTO chat_logs (agent_name, sender, content, created_at) VALUES (?, ?, ?, ?)",
                (agent_name, sender, content, now),
            )

    def get_chat_history(self, agent_name: str, limit: int = 50) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT sender, content, created_at FROM chat_logs WHERE agent_name = ? ORDER BY id DESC LIMIT ?",
                (agent_name, limit),
            ).fetchall()
            return [dict(r) for r in reversed(rows)]

    @staticmethod
    def _row_to_agent(row: sqlite3.Row) -> AgentRecord:
        group_list = [g.strip() for g in row["groups"].split(",") if g.strip()]
        return AgentRecord(
            id=row["id"],
            name=row["name"],
            identity=row["identity"],
            personality=row["personality"],
            job=row["job"],
            model_path=row["model_path"],
            model_architecture=row["model_architecture"],
            port=row["port"],
            context_size=row["context_size"],
            n_gpu_layers=row["n_gpu_layers"],
            threads=row["threads"],
            template_kind=row["template_kind"],
            telegram_enabled=bool(row["telegram_enabled"]),
            telegram_bot_token=row["telegram_bot_token"],
            callable_by_agents=bool(row["callable_by_agents"]),
            groups=group_list,
            status=row["status"],
            pid=row["pid"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
