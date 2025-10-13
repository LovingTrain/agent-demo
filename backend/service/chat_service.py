# backend/service/chat_service.py
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, Any, Optional

import toml
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from api import api_router
from api.auth_api import router as auth_router
from api.chat_api import router as chat_router, init_chat_routes
from api.session_api import router as session_router

from service.history_service import HistoryService
from service.session_service import SessionService

from repo.db import migrate


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
}


# ---------- 对内暴露函数 ----------
async def run_stream(
    *, user_id: int, session_id: str, query: str
) -> AsyncGenerator[str, None]:
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

    async for chunk in agent_executor.astream(agent_input):
        if "output" in chunk:
            yield chunk["output"]


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
    # 向量库
    history: HistoryService = app_state["history"]
    await history.add_history(session_id=session_id, role="user", message=user_msg)
    if ai_msg:
        await history.add_history(session_id=session_id, role="ai", message=ai_msg)
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

    model = ChatOpenAI(
        model=cfg_llm["model_name"],
        api_key=api_key,
        base_url=cfg_llm["base_url"],
        temperature=0.7,
        streaming=True,
    )
    agent = create_tool_calling_agent(
        llm=model, tools=list(mcp_tools_map.values()), prompt=agent_prompt
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
