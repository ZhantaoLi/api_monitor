from __future__ import annotations

import sqlite3
import threading
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class Database:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._write_lock = threading.Lock()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        return conn

    @staticmethod
    def _as_target(row: sqlite3.Row) -> Dict[str, Any]:
        item = dict(row)
        item["enabled"] = bool(item.get("enabled", 0))
        item["verify_ssl"] = bool(item.get("verify_ssl", 0))
        return item

    @staticmethod
    def _as_run(row: sqlite3.Row) -> Dict[str, Any]:
        return dict(row)

    @staticmethod
    def _as_log(row: sqlite3.Row) -> Dict[str, Any]:
        item = dict(row)
        item["stream"] = bool(item.get("stream", 0))
        item["success"] = bool(item.get("success", 0))
        item["transport_success"] = bool(item.get("transport_success", 0))
        return item

    def init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS targets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    base_url TEXT NOT NULL,
                    api_key TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    interval_min INTEGER NOT NULL DEFAULT 30,
                    timeout_s REAL NOT NULL DEFAULT 30.0,
                    verify_ssl INTEGER NOT NULL DEFAULT 0,
                    prompt TEXT NOT NULL DEFAULT 'What is the exact model identifier (model string) you are using for this chat/session?',
                    anthropic_version TEXT NOT NULL DEFAULT '2025-09-29',
                    max_models INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    last_run_at REAL,
                    last_status TEXT,
                    last_total INTEGER,
                    last_success INTEGER,
                    last_fail INTEGER,
                    last_log_file TEXT,
                    last_error TEXT,
                    source_url TEXT
                );

                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    target_id INTEGER NOT NULL,
                    started_at REAL NOT NULL,
                    finished_at REAL,
                    status TEXT NOT NULL,
                    total INTEGER NOT NULL DEFAULT 0,
                    success INTEGER NOT NULL DEFAULT 0,
                    fail INTEGER NOT NULL DEFAULT 0,
                    log_file TEXT,
                    error TEXT,
                    FOREIGN KEY(target_id) REFERENCES targets(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS run_models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    target_id INTEGER NOT NULL,
                    protocol TEXT,
                    model TEXT,
                    stream INTEGER NOT NULL DEFAULT 0,
                    duration REAL,
                    success INTEGER NOT NULL DEFAULT 0,
                    transport_success INTEGER NOT NULL DEFAULT 0,
                    tool_calls_count INTEGER NOT NULL DEFAULT 0,
                    tool_calls TEXT,
                    content TEXT,
                    timestamp REAL,
                    error TEXT,
                    status_code INTEGER,
                    route TEXT,
                    endpoint TEXT,
                    FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE,
                    FOREIGN KEY(target_id) REFERENCES targets(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_targets_enabled_last_run
                ON targets(enabled, last_run_at);

                CREATE INDEX IF NOT EXISTS idx_runs_target_started
                ON runs(target_id, started_at DESC);

                CREATE INDEX IF NOT EXISTS idx_run_models_target_time
                ON run_models(target_id, timestamp DESC);

                CREATE INDEX IF NOT EXISTS idx_run_models_run
                ON run_models(run_id);
                """
            )
        self._migrate_db()

    def _migrate_db(self) -> None:
        """Add missing columns to existing tables."""
        with self._connect() as conn:
            # Check if source_url exists in targets
            columns = [info[1] for info in conn.execute("PRAGMA table_info(targets)").fetchall()]
            if "source_url" not in columns:
                try:
                    conn.execute("ALTER TABLE targets ADD COLUMN source_url TEXT")
                except sqlite3.OperationalError:
                    pass

    def list_targets(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM targets ORDER BY id ASC"
            ).fetchall()
        return [self._as_target(row) for row in rows]

    def get_target(self, target_id: int) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM targets WHERE id = ?",
                (target_id,),
            ).fetchone()
        if not row:
            return None
        return self._as_target(row)

    def create_target(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        now = time.time()
        with self._write_lock:
            with self._connect() as conn:
                cur = conn.execute(
                    """
                    INSERT INTO targets (
                        name, base_url, api_key, enabled, interval_min, timeout_s, verify_ssl,
                        prompt, anthropic_version, max_models, source_url, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        payload["name"],
                        payload["base_url"],
                        payload["api_key"],
                        1 if payload.get("enabled", True) else 0,
                        int(payload.get("interval_min", 30)),
                        float(payload.get("timeout_s", 30.0)),
                        1 if payload.get("verify_ssl", False) else 0,
                        payload.get("prompt", "What is the exact model identifier (model string) you are using for this chat/session?"),
                        payload.get("anthropic_version", "2025-09-29"),
                        int(payload.get("max_models", 0)),
                        payload.get("source_url"),
                        now,
                        now,
                    ),
                )
                target_id = int(cur.lastrowid)
        target = self.get_target(target_id)
        if not target:
            raise RuntimeError("create_target failed")
        return target

    def update_target(self, target_id: int, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not updates:
            return self.get_target(target_id)

        allowed = {
            "name",
            "base_url",
            "api_key",
            "enabled",
            "interval_min",
            "timeout_s",
            "verify_ssl",
            "prompt",
            "anthropic_version",
            "anthropic_version",
            "max_models",
            "source_url",
        }
        fields = []
        values: List[Any] = []
        for key, value in updates.items():
            if key not in allowed:
                continue
            if key in {"enabled", "verify_ssl"}:
                value = 1 if bool(value) else 0
            if key in {"interval_min", "max_models"} and value is not None:
                value = int(value)
            if key == "timeout_s" and value is not None:
                value = float(value)
            fields.append(f"{key} = ?")
            values.append(value)

        if not fields:
            return self.get_target(target_id)

        fields.append("updated_at = ?")
        values.append(time.time())
        values.append(target_id)

        with self._write_lock:
            with self._connect() as conn:
                conn.execute(
                    f"UPDATE targets SET {', '.join(fields)} WHERE id = ?",
                    tuple(values),
                )
        return self.get_target(target_id)

        return self.get_target(target_id)

    def delete_target(self, target_id: int) -> bool:
        with self._write_lock:
            with self._connect() as conn:
                cur = conn.execute("DELETE FROM targets WHERE id = ?", (target_id,))
                return cur.rowcount > 0

    def list_due_targets(self, now_ts: float) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM targets
                WHERE enabled = 1
                AND (
                    last_run_at IS NULL
                    OR (? - last_run_at) >= (interval_min * 60)
                )
                ORDER BY COALESCE(last_run_at, 0) ASC, id ASC
                """,
                (now_ts,),
            ).fetchall()
        return [self._as_target(row) for row in rows]

    def get_latest_model_statuses(self, target_id: int) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            # Get latest run_id
            run_row = conn.execute(
                "SELECT id FROM runs WHERE target_id = ? ORDER BY started_at DESC LIMIT 1",
                (target_id,),
            ).fetchone()
            if not run_row:
                return []
            
            run_id = run_row["id"]
            rows = conn.execute(
                """
                SELECT model, success, duration, error, timestamp
                FROM run_models
                WHERE run_id = ?
                ORDER BY model ASC
                """,
                (run_id,),
            ).fetchall()
            
            return [
                {
                    "model": r["model"],
                    "success": bool(r["success"]),
                    "duration": r["duration"],
                    "error": r["error"]
                }
                for r in rows
            ]

    def create_run(self, target_id: int, started_at: float, log_file: str) -> int:
        with self._write_lock:
            with self._connect() as conn:
                cur = conn.execute(
                    """
                    INSERT INTO runs (
                        target_id, started_at, status, log_file
                    ) VALUES (?, ?, 'running', ?)
                    """,
                    (target_id, started_at, log_file),
                )
                run_id = int(cur.lastrowid)
        return run_id

    def finish_run(
        self,
        run_id: int,
        *,
        status: str,
        finished_at: float,
        total: int,
        success: int,
        fail: int,
        error: Optional[str],
    ) -> None:
        with self._write_lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE runs
                    SET status = ?, finished_at = ?, total = ?, success = ?, fail = ?, error = ?
                    WHERE id = ?
                    """,
                    (status, finished_at, total, success, fail, error, run_id),
                )

    def update_target_after_run(
        self,
        target_id: int,
        *,
        last_run_at: float,
        last_status: str,
        last_total: int,
        last_success: int,
        last_fail: int,
        last_log_file: str,
        last_error: Optional[str],
    ) -> None:
        with self._write_lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE targets
                    SET
                        last_run_at = ?,
                        last_status = ?,
                        last_total = ?,
                        last_success = ?,
                        last_fail = ?,
                        last_log_file = ?,
                        last_error = ?,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        last_run_at,
                        last_status,
                        last_total,
                        last_success,
                        last_fail,
                        last_log_file,
                        last_error,
                        time.time(),
                        target_id,
                    ),
                )

    def insert_model_rows(self, run_id: int, target_id: int, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        payload = []
        for row in rows:
            payload.append(
                (
                    run_id,
                    target_id,
                    row.get("protocol"),
                    row.get("model"),
                    1 if row.get("stream") else 0,
                    row.get("duration"),
                    1 if row.get("success") else 0,
                    1 if row.get("transport_success") else 0,
                    int(row.get("tool_calls_count", 0)),
                    row.get("tool_calls_json", "[]"),
                    row.get("content", ""),
                    row.get("timestamp"),
                    row.get("error"),
                    row.get("status_code"),
                    row.get("route"),
                    row.get("endpoint"),
                )
            )
        with self._write_lock:
            with self._connect() as conn:
                conn.executemany(
                    """
                    INSERT INTO run_models (
                        run_id, target_id, protocol, model, stream, duration, success,
                        transport_success, tool_calls_count, tool_calls, content, timestamp,
                        error, status_code, route, endpoint
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    payload,
                )

    def list_runs(self, target_id: int, limit: int = 20) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM runs
                WHERE target_id = ?
                ORDER BY started_at DESC, id DESC
                LIMIT ?
                """,
                (target_id, limit),
            ).fetchall()
        return [self._as_run(row) for row in rows]

    def get_latest_run(self, target_id: int) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM runs
                WHERE target_id = ?
                ORDER BY started_at DESC, id DESC
                LIMIT 1
                """,
                (target_id,),
            ).fetchone()
        if not row:
            return None
        return self._as_run(row)

    def list_logs(self, target_id: int, *, run_id: Optional[int], limit: int) -> List[Dict[str, Any]]:
        sql = """
            SELECT * FROM run_models
            WHERE target_id = ?
        """
        params: List[Any] = [target_id]
        if run_id is not None:
            sql += " AND run_id = ?"
            params.append(run_id)
        sql += " ORDER BY timestamp ASC, id ASC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        logs = [self._as_log(row) for row in rows]
        for item in logs:
            raw = item.pop("tool_calls", "[]")
            try:
                if isinstance(raw, list):
                    item["tool_calls"] = raw
                elif isinstance(raw, str) and raw:
                    item["tool_calls"] = json.loads(raw)
                else:
                    item["tool_calls"] = []
            except Exception:
                item["tool_calls"] = []
        return logs
