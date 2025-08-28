# backend/chat_service.py
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict

import toml
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

# --- 1. 配置加载 ---
config = toml.load("config/chat.toml")
cfg_server = config["server"]
cfg_mcp = config["mcp_services"]
cfg_llm = config["models"]["llm"]
cfg_prompt = config["prompt"]

CORE_DEPENDENCY = "history"

# --- 2. Prompt模板构建 ---
agent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", cfg_prompt["system_template"]),
        MessagesPlaceholder(variable_name="history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# --- 3. FastAPI Lifespan & 资源加载 ---
app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Chat service is starting up ...")
    if CORE_DEPENDENCY not in cfg_mcp:
        raise RuntimeError(
            f"Fatal Error: core dependency '{CORE_DEPENDENCY}' is missing from [mcp_services]"
        )
    client = MultiServerMCPClient(cfg_mcp)
    try:
        mcp_tools = await client.get_tools()
        app_state["mcp_tools_map"] = {tool.name: tool for tool in mcp_tools}
        print(
            f"✅ Successfully load MCP tools: {list(app_state['mcp_tools_map'].keys())}"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load MCP tools: {e}") from e
    api_key = os.getenv(cfg_llm["api_key_env"])
    if not api_key:
        raise ValueError(f"Environment variable '{cfg_llm['api_key_env']}' is not set")
    app_state["model"] = ChatOpenAI(
        model=cfg_llm["model_name"],
        api_key=api_key,
        base_url=cfg_llm["base_url"],
        temperature=0.7,
        streaming=True,
    )
    agent = create_tool_calling_agent(
        llm=app_state["model"],
        tools=list(app_state["mcp_tools_map"].values()),
        prompt=agent_prompt,
    )
    app_state["agent_executor"] = AgentExecutor(
        agent=agent,
        tools=list(app_state["mcp_tools_map"].values()),
        verbose=True,
        handle_parsing_errors=True,
    )
    yield
    print("👋 Chat service is closed")


app = FastAPI(
    title="AI Chat Service (MCP Architecture)",
    version="4.2 CORS Fixed",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg_server["allow_origins"],  # 允许访问的源
    allow_credentials=True,  # 是否支持cookie
    allow_methods=["*"],  # 允许所有方法 (GET, POST, OPTIONS 等)
    allow_headers=["*"],  # 允许所有请求头
)


# --- 4. 核心 Agent 逻辑生成器 (无变化) ---
async def agent_logic_generator(inputs: Dict) -> AsyncGenerator[str, None]:
    query = inputs["input"]
    session_id = inputs.get("session_id")
    tools_map = app_state["mcp_tools_map"]
    history_context = "(未使用历史记录)"
    if session_id and "search_history" in tools_map:
        history_context = await tools_map["search_history"].ainvoke(
            {"session_id": session_id, "query": query}
        )
    print(f"Find history context: {history_context}")
    knowledge_context = "(未使用知识库)"
    if "search_multimodal_knowledge" in tools_map:
        knowledge_context = await tools_map["search_multimodal_knowledge"].ainvoke(
            {"query": query}
        )
    print(f"Find knowledge context: {knowledge_context}")
    agent_input = {
        "input": query,
        "history": [],
        "history_context": history_context,
        "knowledge_context": knowledge_context,
    }
    async for chunk in app_state["agent_executor"].astream(agent_input):
        if "output" in chunk:
            yield chunk["output"]


# --- 5. API 路由 ---
class ChatRequest(BaseModel):
    input: str
    session_id: str


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    full_response = []

    async def stream_wrapper():
        async for token in agent_logic_generator(req.model_dump()):
            full_response.append(token)
            yield f"data: {token}\n\n"

        final_output = "".join(full_response)
        tools_map = app_state["mcp_tools_map"]
        add_history_tool = tools_map.get("add_history")
        if add_history_tool:
            await add_history_tool.ainvoke(
                {"session_id": req.session_id, "role": "user", "message": req.input}
            )
            await add_history_tool.ainvoke(
                {"session_id": req.session_id, "role": "ai", "message": final_output}
            )

    return StreamingResponse(stream_wrapper(), media_type="text/event-stream")


@app.get("/")
def root():
    return {
        "message": "AI Chat Service is running",
        "architecture": "MCP-based History",
    }


# --- 6. 入口点 (无变化) ---
if __name__ == "__main__":
    host = cfg_server.get("host", "0.0.0.0")
    port = cfg_server.get("port", 8000)
    print(f"🌍 Chat service will be launch at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
