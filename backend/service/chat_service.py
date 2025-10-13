# backend/service/chat_service.py
import os
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, Any, Optional, List

import toml
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import AsyncCallbackHandler

from api import api_router
from api.auth_api import router as auth_router
from api.chat_api import router as chat_router, init_chat_routes
from api.session_api import router as session_router

from service.history_service import HistoryService
from service.session_service import SessionService

from repo.base import migrate


# --- 配置 ---
config = toml.load("config/chat.toml")
cfg_server = config["server"]
cfg_mcp = config.get("mcp_services", {})  # 可选保留其他 MCP
cfg_llm = config["models"]["llm"]
cfg_prompt = config["prompt"]

# Prompt 模板
agent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", cfg_prompt["system_template"]),
        MessagesPlaceholder(variable_name="history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# 应用状态
app_state: Dict[str, Any] = {
    "agent_executor": None,
    "mcp_tools_map": {},
    "history": None,  # HistoryService 实例
    "llm_model_name": None,
    "llm_base_url": None,
    "llm_api_key": None,
}


# ---------- 回调：把 token 推入队列 ----------
class QueueCallbackHandler(AsyncCallbackHandler):
    def __init__(self, queue: "asyncio.Queue[Optional[str]]"):
        self.queue = queue

    async def on_llm_new_token(self, token: str, *args, **kwargs):
        # 每个新 token 直接送入队列
        # 注意：token 可能是多行，保持原样，不做 trim
        await self.queue.put(token)

    async def on_llm_error(self, error: Exception, *args, **kwargs):
        # 出错也要通知消费端结束
        try:
            await self.queue.put(None)
        except Exception:
            pass

    async def on_llm_end(self, *args, **kwargs):
        # 由上层在推理结束时负责发送 None；这里可不处理
        pass


# ---------- 对内暴露函数 ----------
async def run_stream(
    *, user_id: int, session_id: str, query: str
) -> AsyncGenerator[str, None]:
    """
    使用 AgentExecutor 的 astream 事件流并转为增量字符串输出。
    注意：LangChain 的 astream 返回的是“事件字典”，其中 'output' 事件通常是完整片段，
    但配合 ChatOpenAI(streaming=True + callbacks)，会触发 on_llm_new_token 回调，
    我们通过队列将每个 token 传回本函数并逐个 yield。
    """
    agent_executor: AgentExecutor = app_state["agent_executor"]
    tools_map = app_state["mcp_tools_map"]
    history: HistoryService = app_state["history"]

    # 历史 RAG
    history_context = await history.search_history(
        session_id=session_id, query=query, k=3
    )

    # 其余知识（如仍有 MCP）
    knowledge_context = "(未使用知识库)"
    if "search_multimodal_knowledge" in tools_map:
        knowledge_context = await tools_map["search_multimodal_knowledge"].ainvoke(
            {"user_id": user_id, "query": query}
        )

    agent_input = {
        "input": query,
        "history": [],
        "history_context": history_context,
        "knowledge_context": knowledge_context,
        "user_id": user_id,
        "session_id": session_id,
    }

    # 队列接收 token
    q: "asyncio.Queue[Optional[str]]" = asyncio.Queue(maxsize=1000)
    cb = QueueCallbackHandler(q)

    # 关键：给 AgentExecutor 注入回调，让底层 ChatOpenAI 流式 token 进入回调
    # LangChain 的 Runnable 支持 with_config(callbacks=[...])
    runnable = agent_executor.with_config({"callbacks": [cb]})

    # 启动推理（异步），不要在这里 await 完整结果
    async def produce():
        try:
            # astream 会按事件流产出，但我们不逐个消费事件，
            # 仅仅启动它从而触发底层 llm 的 on_llm_new_token 回调。
            # 为避免阻塞，这里只需 await 完整迭代从而让流程走完。
            async for _ in runnable.astream(agent_input):
                # 我们不直接用 chunk，因为更细粒度的 token 已在回调里送入队列
                pass
        except Exception:
            # 出错时通知消费端结束
            await q.put(None)
            raise
        finally:
            # 结束时发 None 关闭消费端
            await q.put(None)

    producer = asyncio.create_task(produce())

    try:
        while True:
            item = await q.get()
            if item is None:
                break
            # 直接把 token 传给上层（SSE 层），不要改动换行
            yield item
    finally:
        await asyncio.gather(producer, return_exceptions=True)


session_svc = SessionService()


async def add_history(
    *, user_id: int, session_id: str, user_msg: str, ai_msg: Optional[str]
):
    # 结构化存储
    session_svc.upsert_session(user_id=user_id, session_id=session_id, name=session_id)
    session_svc.add_user_message(
        user_id=user_id, session_id=session_id, content=user_msg
    )
    if ai_msg:
        session_svc.add_ai_message(session_id=session_id, content=ai_msg)
    # 向量库（避免重复写入）
    history: HistoryService = app_state["history"]
    await history.add_history(session_id=session_id, role="user", message=user_msg)
    if ai_msg:
        await history.add_history(session_id=session_id, role="ai", message=ai_msg)


# ---------- 初始化 ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Chat service is starting up ...")

    # 0) 迁移数据库
    migrate()

    # 1) 初始化 HistoryService
    history = HistoryService(config_path="config/history.toml")
    await history.ainit()
    app_state["history"] = history

    # 2) 其他 MCP 工具（非历史，可选）
    mcp_tools_map: Dict[str, Any] = {}
    if cfg_mcp:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        client_mcp = MultiServerMCPClient(cfg_mcp)
        try:
            mcp_tools = await client_mcp.get_tools()
            mcp_tools_map = {tool.name: tool for tool in mcp_tools}
            print(f"✅ MCP tools loaded: {list(mcp_tools_map.keys())}")
        except Exception as e:
            raise RuntimeError(f"Failed to load MCP tools: {e}") from e
    app_state["mcp_tools_map"] = mcp_tools_map

    # 3) LLM 与 Agent
    api_key = os.getenv(cfg_llm["api_key_env"])
    if not api_key:
        raise ValueError(f"Environment variable '{cfg_llm['api_key_env']}' is not set")

    # 记录 LLM 基本信息（可用于调试/复用）
    app_state["llm_model_name"] = cfg_llm["model_name"]
    app_state["llm_base_url"] = cfg_llm["base_url"]
    app_state["llm_api_key"] = api_key

    # 注意：这里不在 ChatOpenAI 构造时绑定 callbacks，
    # 因为我们在 run_stream 里通过 runnable = agent_executor.with_config({"callbacks":[cb]}) 注入更灵活
    llm = ChatOpenAI(
        model=cfg_llm["model_name"],
        api_key=api_key,
        base_url=cfg_llm["base_url"],
        temperature=0.7,
        streaming=True,  # 允许流式
    )

    agent = create_tool_calling_agent(
        llm=llm, tools=list(mcp_tools_map.values()), prompt=agent_prompt
    )
    app_state["agent_executor"] = AgentExecutor(
        agent=agent,
        tools=list(mcp_tools_map.values()),
        verbose=True,
        handle_parsing_errors=True,
    )

    # 4) 注入内部分发函数供 API 使用
    init_chat_routes(run_stream, add_history)

    yield
    print("👋 Chat service is closed")


# FastAPI 应用
app = FastAPI(
    title="AI Chat Service (Service Layer + HistoryService)",
    version="10.0 Decoupled History",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg_server["allow_origins"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 路由挂载
api_router.include_router(auth_router)
api_router.include_router(chat_router)
api_router.include_router(session_router)
app.include_router(api_router)


@app.get("/")
def root():
    return {
        "message": "AI Chat Service is running",
        "architecture": "Service layer + HistoryService + optional MCP",
    }


if __name__ == "__main__":
    host = cfg_server.get("host", "0.0.0.0")
    port = cfg_server.get("port", 8000)
    print(f"🌍 Chat service will be launch at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
