from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone



class SessionStore:
    def __init__(self, db_path: str = "ouroboros_sessions.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    seed TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    config TEXT NOT NULL,
                    state TEXT NOT NULL,
                    stream TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    ended_at TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS insights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    insight TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            """)

    def save_session(self, session_id: str, data: dict):
        state = {k: v for k, v in data.get("state", {}).items() if k != "messages"}
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO sessions
                   (id, seed, mode, config, state, stream, created_at, ended_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    data["seed"],
                    data.get("mode", "explore"),
                    json.dumps(data.get("config", {})),
                    json.dumps(state, default=str),
                    json.dumps(data.get("stream", [])),
                    data.get("created_at", datetime.now(timezone.utc).isoformat()),
                    data.get("ended_at"),
                ),
            )
            for insight in data.get("state", {}).get("insights", []):
                conn.execute(
                    "INSERT INTO insights (session_id, insight, created_at) VALUES (?, ?, ?)",
                    (session_id, insight, datetime.now(timezone.utc).isoformat()),
                )

    def get_session(self, session_id: str) -> dict | None:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if not row:
                return None
            insights_rows = conn.execute(
                "SELECT insight FROM insights WHERE session_id = ?", (session_id,)
            ).fetchall()
            return {
                "id": row["id"],
                "seed": row["seed"],
                "mode": row["mode"],
                "config": json.loads(row["config"]),
                "state": json.loads(row["state"]),
                "stream": json.loads(row["stream"]),
                "insights": [r["insight"] for r in insights_rows],
                "created_at": row["created_at"],
                "ended_at": row["ended_at"],
            }

    def list_sessions(self) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id, seed, mode, created_at FROM sessions ORDER BY created_at DESC"
            ).fetchall()
            results = []
            for row in rows:
                insight_count = conn.execute(
                    "SELECT COUNT(*) FROM insights WHERE session_id = ?",
                    (row["id"],),
                ).fetchone()[0]
                results.append(
                    {
                        "id": row["id"],
                        "seed": row["seed"],
                        "mode": row["mode"],
                        "created_at": row["created_at"],
                        "insight_count": insight_count,
                    }
                )
            return results

    def delete_session(self, session_id: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM insights WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

    def update_state(self, session_id: str, state: dict):
        clean = {k: v for k, v in state.items() if k != "messages"}
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE sessions SET state = ? WHERE id = ?",
                (json.dumps(clean, default=str), session_id),
            )
