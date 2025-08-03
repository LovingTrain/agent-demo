import os
import re
import asyncio
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langserve import add_routes

# --- 1. MCP 工具远程加载（包括 history 工具） ---
MCP_SERVERS = {
    "knowledge": {
        "url": "http://localhost:9000/mcp/",
        "transport": "streamable_http",
    },
    "history": {
        "url": "http://localhost:9100/mcp/",
        "transport": "streamable_http",
    },
}


async def load_tools():
    client = MultiServerMCPClient(MCP_SERVERS)
    return await client.get_tools()


mcp_tools = asyncio.run(load_tools())


# 单独获取历史相关工具
def find_tool(name):
    for t in mcp_tools:
        if t.name == name:
            return t
    raise RuntimeError(f"Tool {name} not found!")


search_history_tool = find_tool("search_history")
add_history_tool = find_tool("add_history")
search_knowledge_tool = find_tool("search_multimodal_knowledge")  # 可选

# --- 2. LLM & Agent ---
model = ChatOpenAI(
    model="qwen-plus",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.7,
)

agent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是原神AI助手。"
            "\n和用户历史对话中与本问题最相关的内容如下：\n{history_context}\n"
            "游戏知识库的相关内容如下：\n{knowledge_context}\n"
            "请根据这些信息回答用户。",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


async def agent_chain_fn(inputs: dict) -> str:
    session_id = inputs.get("session_id", "default")
    query = inputs["input"]

    # 1. 用history MCP工具召回相关历史
    history_context = await search_history_tool.ainvoke(
        {"session_id": session_id, "query": query, "k": 3}
    )
    # 2. 用knowledge MCP工具召回知识（可选，也可以让agent自主调工具）
    knowledge_context = await search_knowledge_tool.ainvoke({"query": query})
    # 3. 组装prompt
    payload = {
        "input": query,
        "session_id": session_id,
        "history_context": history_context,
        "knowledge_context": knowledge_context,
        "history": inputs.get("history", []),
    }
    # 4. Agent function_calling
    agent = create_tool_calling_agent(llm=model, tools=mcp_tools, prompt=agent_prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=mcp_tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3,
    )
    result = await executor.ainvoke(payload)
    # 5. 保存本轮历史（用户问题、AI回答都存到历史RAG工具里）
    await add_history_tool.ainvoke(
        {"session_id": session_id, "role": "user", "message": query}
    )
    await add_history_tool.ainvoke(
        {"session_id": session_id, "role": "ai", "message": result["output"]}
    )
    return result["output"]


class InputChat(BaseModel):
    input: str
    session_id: str = "default"


agent_chain = RunnableLambda(agent_chain_fn).with_types(input_type=InputChat)


# --- 3. session history（传统消息历史, 可选, 支持playground） ---
def _valid_id(s: str) -> bool:
    return bool(re.fullmatch(r"[a-zA-Z0-9\-_]+", s))


def make_history_factory(base: str):
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)

    def _factory(session_id: str) -> FileChatMessageHistory:
        if not _valid_id(session_id):
            raise HTTPException(400, "Session ID 只能包含字母、数字、'-'、'_'")
        return FileChatMessageHistory(str(p / f"{session_id}.json"))

    return _factory


# with_history_chain = RunnableWithMessageHistory(
#     agent_chain,
#     make_history_factory("chat_histories"),
#     input_messages_key="input",
#     history_messages_key="history",
# ).with_types(input_type=InputChat)

# --- 4. FastAPI/Serve ---
app = FastAPI(
    title="Genshin Assistant (历史RAG MCP)",
    version="v0.3-mcp-history",
    description="原神 AI 助手，RAG知识库+历史向量记忆(MCP工具)",
)
add_routes(app, agent_chain, path="/chat")


@app.get("/")
async def root():
    return {
        "how_to": "POST /chat?session_id=your_id  JSON: {'input': '你的问题'}",
        "tip": "本服务支持RAG知识库与历史RAG（历史通过MCP工具远程管理）",
    }


if __name__ == "__main__":
    import uvicorn

    print("🚀 http://localhost:8000")
    uvicorn.run(app, host="localhost", port=8000)
