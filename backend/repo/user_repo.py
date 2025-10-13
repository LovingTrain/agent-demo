# backend/repo/user_repo.py
from typing import Optional, Dict, Any
from .db import tx


def create_user(user_id: int, username: str, password_hash: str):
    with tx() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO users (id, username, password_hash) VALUES (?, ?, ?)",
            (user_id, username, password_hash),
        )


def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    with tx() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, username, password_hash FROM users WHERE username = ?",
            (username,),
        )
        row = cur.fetchone()
        return dict(row) if row else None
