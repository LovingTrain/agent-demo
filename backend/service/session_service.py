from __future__ import annotations
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session as SASession
from repo.base import SessionLocal
from repo import session_repo


class SessionService:
    def _db(self) -> SASession:
        return SessionLocal()

    def upsert_session(
        self, *, user_id: int, session_id: str, name: Optional[str] = None
    ) -> None:
        db = self._db()
        try:
            session_repo.upsert_session(
                db, session_id=session_id, user_id=user_id, name=name
            )
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def list_sessions(self, *, user_id: int) -> List[Dict[str, Any]]:
        db = self._db()
        try:
            return session_repo.list_sessions(db, user_id=user_id)
        finally:
            db.close()

    def rename_session(self, *, user_id: int, session_id: str, new_name: str) -> bool:
        db = self._db()
        try:
            ok = session_repo.rename_session(
                db, user_id=user_id, session_id=session_id, new_name=new_name
            )
            db.commit()
            return ok
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def delete_session(self, *, user_id: int, session_id: str) -> bool:
        db = self._db()
        try:
            ok = session_repo.delete_session(db, user_id=user_id, session_id=session_id)
            db.commit()
            return ok
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def add_user_message(self, *, user_id: int, session_id: str, content: str) -> int:
        db = self._db()
        try:
            # 确保会话存在
            session_repo.upsert_session(
                db, session_id=session_id, user_id=user_id, name=session_id
            )
            m = session_repo.add_message(
                db, session_id=session_id, role="user", content=content
            )
            db.commit()
            return int(m.id)
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def add_ai_message(self, *, session_id: str, content: str) -> int:
        db = self._db()
        try:
            m = session_repo.add_message(
                db, session_id=session_id, role="ai", content=content
            )
            db.commit()
            return int(m.id)
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def list_messages(
        self,
        *,
        user_id: int,
        session_id: str,
        limit: int = 100,
        before_id: Optional[int] = None,
    ):
        db = self._db()
        try:
            return session_repo.list_messages(
                db, session_id=session_id, limit=limit, before_id=before_id
            )
        finally:
            db.close()
