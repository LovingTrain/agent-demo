# backend/api/chat_api.py
from typing import List
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from service.auth_service import CurrentUser, get_current_user

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    input: str
    session_id: str


# 将在应用启动时注入 run_stream 与 add_history
def init_chat_routes(run_stream_fn, add_history_fn):
    global run_stream, add_history
    run_stream = run_stream_fn
    add_history = add_history_fn


@router.post("/stream")
async def chat_stream(
    req: ChatRequest, current_user: CurrentUser = Depends(get_current_user)
):
    tokens: List[str] = []

    async def sse():
        async for token in run_stream(
            user_id=current_user.id, session_id=req.session_id, query=req.input
        ):
            tokens.append(token)
            yield f"data: {token}\n\n"
        if add_history:
            ai_text = "".join(tokens) if tokens else None
            await add_history(
                user_id=current_user.id,
                session_id=req.session_id,
                user_msg=req.input,
                ai_msg=ai_text,
            )

    return StreamingResponse(sse(), media_type="text/event-stream")
