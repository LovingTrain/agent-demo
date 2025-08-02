from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from mcp.server.fastmcp import FastMCP
from chromadb import Settings

HIST_VEC_PATH = "./history_chroma"
embeddings = DashScopeEmbeddings(model="text-embedding-v1")


def setup_history_vectorstore():
    import os

    os.makedirs(HIST_VEC_PATH, exist_ok=True)
    return Chroma(
        persist_directory=HIST_VEC_PATH,
        embedding_function=embeddings,
        client_settings=Settings(anonymized_telemetry=False),
    )


history_vectorstore = setup_history_vectorstore()

m = FastMCP("history", port=9100)


@m.tool()
def add_history(session_id: str, role: str, message: str) -> str:
    """将对话历史写入历史向量库"""
    doc = Document(
        page_content=message, metadata={"session_id": session_id, "role": role}
    )
    history_vectorstore.add_documents([doc])
    return "ok"


@m.tool()
def search_history(session_id: str, query: str, k: int = 3) -> str:
    """在session历史库用query召回最相关历史消息，返回文本片段（最多k条）"""
    all_hits = history_vectorstore.similarity_search(query, k=10)
    hits = [
        d.page_content for d in all_hits if d.metadata.get("session_id") == session_id
    ][:k]
    return "\n".join(hits) if hits else "(无相关历史)"


if __name__ == "__main__":
    m.run(transport="streamable-http")
