# backend/api/session_api.py
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Body
from pydantic import BaseModel, Field

from service.auth_service import CurrentUser, get_current_user
from service.session_service import SessionService

router = APIRouter(prefix="/sessions", tags=["sessions"])
svc = SessionService()


class SessionCreate(BaseModel):
    id: str = Field(..., description="客户端生成或后端建议的会话ID")
    name: Optional[str] = None


class SessionRename(BaseModel):
    name: str


class MessageCreate(BaseModel):
    content: str


@router.get("")
def list_sessions(current: CurrentUser = Depends(get_current_user)):
    return svc.list_sessions(user_id=current.id)


@router.post("")
def create_or_upsert_session(
    body: SessionCreate, current: CurrentUser = Depends(get_current_user)
):
    svc.upsert_session(
        user_id=current.id, session_id=body.id, name=body.name or body.id
    )
    return {"ok": True}


@router.patch("/{session_id}")
def rename_session(
    session_id: str,
    body: SessionRename,
    current: CurrentUser = Depends(get_current_user),
):
    if not svc.rename_session(
        user_id=current.id, session_id=session_id, new_name=body.name
    ):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"ok": True}


@router.delete("/{session_id}")
def delete_session(session_id: str, current: CurrentUser = Depends(get_current_user)):
    if not svc.delete_session(user_id=current.id, session_id=session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"ok": True}


@router.get("/{session_id}/messages")
def get_messages(
    session_id: str,
    limit: int = Query(100, ge=1, le=500),
    before_id: Optional[int] = Query(None),
    current: CurrentUser = Depends(get_current_user),
):
    return svc.list_messages(
        user_id=current.id, session_id=session_id, limit=limit, before_id=before_id
    )


@router.post("/{session_id}/messages/user")
def add_user_message(
    session_id: str,
    body: MessageCreate,
    current: CurrentUser = Depends(get_current_user),
):
    msg_id = svc.add_user_message(
        user_id=current.id, session_id=session_id, content=body.content
    )
    return {"id": msg_id}
