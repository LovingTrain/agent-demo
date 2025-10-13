from __future__ import annotations
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import select
from .models import User


def get_by_id(db: Session, user_id: int) -> Optional[User]:
    return db.get(User, user_id)


def get_by_username(db: Session, username: str) -> Optional[User]:
    return db.execute(
        select(User).where(User.username == username)
    ).scalar_one_or_none()


def upsert_user(db: Session, user_id: int, username: str, password_hash: str) -> User:
    user = db.get(User, user_id)
    if user:
        user.username = username
        user.password_hash = password_hash
        return user
    user = User(id=user_id, username=username, password_hash=password_hash)
    db.add(user)
    return user


def create_user(db: Session, username: str, password_hash: str) -> User:
    user = User(username=username, password_hash=password_hash)
    db.add(user)
    db.flush()
    return user
