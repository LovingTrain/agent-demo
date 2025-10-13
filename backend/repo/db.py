# backend/repo/db.py
import os
import sqlite3
from contextlib import contextmanager
from typing import Iterator, Optional, Any, Dict
import toml

CFG_PATH = os.getenv("DB_TOML_PATH", "config/db.toml")


def _load_cfg() -> Dict[str, Any]:
    if os.path.exists(CFG_PATH):
        return toml.load(CFG_PATH)
    os.makedirs(os.path.dirname(CFG_PATH), exist_ok=True)
    with open(CFG_PATH, "w", encoding="utf-8") as f:
        toml.dump({"sqlite": {"path": "data/app.sqlite"}}, f)
    return {"sqlite": {"path": "data/app.sqlite"}}


cfg = _load_cfg()
DB_PATH = cfg["sqlite"]["path"]
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def tx() -> Iterator[sqlite3.Connection]:
    conn = get_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def migrate():
    with tx() as conn:
        c = conn.cursor()
        # users
        c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)
        # sessions
        c.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id)")
        # messages
        c.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('user','ai','system','tool')),
            content TEXT NOT NULL,
            token_count INTEGER NULL,
            meta TEXT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
        )
        """)
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_msgs_session ON messages(session_id, created_at)"
        )
