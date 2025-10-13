# backend/service/session_service.py
from typing import List, Optional, Dict, Any
from repo import session_repo


class SessionService:
    def upsert_session(
        self, *, user_id: int, session_id: str, name: Optional[str] = None
    ) -> None:
        session_repo.upsert_session(session_id=session_id, user_id=user_id, name=name)

    def list_sessions(self, *, user_id: int) -> List[Dict[str, Any]]:
        return session_repo.list_sessions(user_id)

    def rename_session(self, *, user_id: int, session_id: str, new_name: str) -> bool:
        return session_repo.rename_session(user_id, session_id, new_name)

    def delete_session(self, *, user_id: int, session_id: str) -> bool:
        return session_repo.delete_session(user_id, session_id)

    def add_user_message(self, *, user_id: int, session_id: str, content: str) -> int:
        # 确保会话存在
        session_repo.upsert_session(
            session_id=session_id, user_id=user_id, name=session_id
        )
        return session_repo.add_message(
            session_id=session_id, role="user", content=content
        )

    def add_ai_message(self, *, session_id: str, content: str) -> int:
        return session_repo.add_message(
            session_id=session_id, role="ai", content=content
        )

    def list_messages(
        self,
        *,
        user_id: int,
        session_id: str,
        limit: int = 100,
        before_id: Optional[int] = None,
    ):
        # 可加入权限校验（确认 session 属于该 user）
        return session_repo.list_messages(
            session_id=session_id, limit=limit, before_id=before_id
        )
