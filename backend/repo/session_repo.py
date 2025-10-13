from __future__ import annotations
from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import select, update, delete, desc
from .models import Session as ChatSession, Message


def upsert_session(
    db: Session, *, session_id: str, user_id: int, name: Optional[str] = None
) -> ChatSession:
    sess = db.get(ChatSession, session_id)
    now = datetime.utcnow()
    if sess:
        if name:
            sess.name = name
        sess.updated_at = now
        return sess
    sess = ChatSession(
        id=session_id,
        user_id=user_id,
        name=name or session_id,
        created_at=now,
        updated_at=now,
    )
    db.add(sess)
    return sess


def list_sessions(db: Session, *, user_id: int) -> List[Dict[str, Any]]:
    rows = (
        db.execute(
            select(ChatSession)
            .where(ChatSession.user_id == user_id)
            .order_by(desc(ChatSession.updated_at))
        )
        .scalars()
        .all()
    )
    return [
        {
            "id": s.id,
            "name": s.name,
            "created_at": s.created_at,
            "updated_at": s.updated_at,
        }
        for s in rows
    ]


def rename_session(
    db: Session, *, user_id: int, session_id: str, new_name: str
) -> bool:
    q = (
        update(ChatSession)
        .where(ChatSession.id == session_id, ChatSession.user_id == user_id)
        .values(name=new_name, updated_at=datetime.utcnow())
    )
    res = db.execute(q)
    return res.rowcount > 0


def delete_session(db: Session, *, user_id: int, session_id: str) -> bool:
    # messages 设了级联，直接删 session 即可
    res = db.execute(
        delete(ChatSession).where(
            ChatSession.id == session_id, ChatSession.user_id == user_id
        )
    )
    return res.rowcount > 0


def add_message(
    db: Session,
    *,
    session_id: str,
    role: str,
    content: str,
    token_count: Optional[int] = None,
    meta: Optional[str] = None,
) -> Message:
    msg = Message(
        session_id=session_id,
        role=role,
        content=content,
        token_count=token_count,
        meta=meta,
    )
    db.add(msg)
    db.flush()
    return msg


def list_messages(
    db: Session, *, session_id: str, limit: int = 100, before_id: Optional[int] = None
):
    stmt = select(Message).where(Message.session_id == session_id)
    if before_id:
        stmt = stmt.where(Message.id < before_id)
    rows = db.execute(stmt.order_by(desc(Message.id)).limit(limit)).scalars().all()
    rows.reverse()
    return [
        {"id": m.id, "role": m.role, "content": m.content, "created_at": m.created_at}
        for m in rows
    ]
