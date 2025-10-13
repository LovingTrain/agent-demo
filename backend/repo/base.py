from __future__ import annotations
import os
from typing import Generator
import toml
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase, scoped_session

CFG_PATH = os.getenv("DB_TOML_PATH", "config/db.toml")


def _load_cfg():
    if os.path.exists(CFG_PATH):
        return toml.load(CFG_PATH)
    os.makedirs(os.path.dirname(CFG_PATH), exist_ok=True)
    return {"sqlite": {"path": "data/app.sqlite"}}


cfg = _load_cfg()
DB_PATH = cfg.get("sqlite", {}).get("path", "data/app.sqlite")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

DATABASE_URL = f"sqlite:///{DB_PATH}"


# SQLite 外键约束开启
@event.listens_for(Engine, "connect")
def _set_sqlite_pragma(dbapi_connection, connection_record):
    try:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
    except Exception:
        pass


engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
    if DATABASE_URL.startswith("sqlite")
    else {},
    pool_pre_ping=True,
)

SessionLocal = scoped_session(
    sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
)


class Base(DeclarativeBase):
    pass


def get_db() -> Generator:
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def migrate():
    # 运行 Base.metadata.create_all
    from .models import User, Session, Message  # noqa: F401

    Base.metadata.create_all(bind=engine)
