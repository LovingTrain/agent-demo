# backend/repo/session_repo.py
from typing import List, Optional, Dict, Any, Tuple
from .db import tx
import sqlite3
import datetime as dt


def upsert_session(session_id: str, user_id: int, name: Optional[str] = None) -> None:
    now = dt.datetime.utcnow().isoformat(sep=" ", timespec="seconds")
    with tx() as conn:
        cur = conn.cursor()
        # try update
        cur.execute(
            "UPDATE sessions SET name = COALESCE(?, name), updated_at = ? WHERE id = ? AND user_id = ?",
            (name, now, session_id, user_id),
        )
        if cur.rowcount == 0:
            cur.execute(
                "INSERT INTO sessions (id, user_id, name, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, user_id, name or session_id, now, now),
            )


def list_sessions(user_id: int) -> List[Dict[str, Any]]:
    with tx() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, name, created_at, updated_at FROM sessions WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        )
        return [dict(r) for r in cur.fetchall()]


def rename_session(user_id: int, session_id: str, new_name: str) -> bool:
    with tx() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE sessions SET name = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ? AND user_id = ?",
            (new_name, session_id, user_id),
        )
        return cur.rowcount > 0


def delete_session(user_id: int, session_id: str) -> bool:
    with tx() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        cur.execute(
            "DELETE FROM sessions WHERE id = ? AND user_id = ?", (session_id, user_id)
        )
        return cur.rowcount > 0


def add_message(
    session_id: str,
    role: str,
    content: str,
    token_count: Optional[int] = None,
    meta: Optional[str] = None,
) -> int:
    with tx() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO messages (session_id, role, content, token_count, meta) VALUES (?, ?, ?, ?, ?)",
            (session_id, role, content, token_count, meta),
        )
        return int(cur.lastrowid)


def list_messages(
    session_id: str, limit: int = 100, before_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    with tx() as conn:
        cur = conn.cursor()
        if before_id:
            cur.execute(
                "SELECT id, role, content, created_at FROM messages WHERE session_id = ? AND id < ? ORDER BY id DESC LIMIT ?",
                (session_id, before_id, limit),
            )
        else:
            cur.execute(
                "SELECT id, role, content, created_at FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            )
        rows = cur.fetchall()
        rows = list(reversed(rows))
        return [dict(r) for r in rows]
