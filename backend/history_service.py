# knowledge_service.py
import logging
import os

import toml
from llama_index.core import Settings, StorageContext
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.vector_stores.qdrant import QdrantVectorStore
from mcp.server.fastmcp import FastMCP
from qdrant_client import QdrantClient

# --- 配置与日志 ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
config = toml.load("config.toml")
cfg_service = config["service"]
cfg_models = config["models"]
cfg_db = config["vector_db"]


def load_retriever_from_qdrant():
    """从已存在的Qdrant索引加载检索器。"""

    # 1. 连接到已存在的Qdrant数据库
    client = QdrantClient(path=cfg_db["path"])
    text_store = QdrantVectorStore(
        client=client, collection_name=cfg_db["text_collection"]
    )
    image_store = QdrantVectorStore(
        client=client, collection_name=cfg_db["image_collection"]
    )
    storage_context = StorageContext.from_defaults(
        vector_store=text_store, image_store=image_store
    )

    # 2. 配置与构建时一致的模型
    Settings.embed_model = HuggingFaceEmbedding(model_name=cfg_models["text_embedding"])
    Settings.multi_modal_llm = OpenAIMultiModal(
        embed_model=cfg_models["image_embedding"], device=cfg_models["device"]
    )

    # 3. 从存储上下文加载索引，不传入documents
    logging.info("正在从Qdrant加载已存在的索引...")
    index = MultiModalVectorStoreIndex(
        nodes=[],  # 传入空列表，表示不添加新文档
        storage_context=storage_context,
    )

    # 4. 创建检索器
    return index.as_retriever(similarity_top_k=3, image_similarity_top_k=1)


def format_docs(docs) -> str:
    """格式化检索结果用于返回"""
    if not docs:
        return "(知识库中无相关内容)"
    lines = []
    for d in docs:
        path = d.metadata.get("file_path", "N/A")
        content = d.get_content(metadata={})
        if d.metadata.get("file_type") == "image/png":
            lines.append(f"[图片]: {path}")
        else:
            lines.append(f"[文本]: {os.path.basename(path)} - {content[:150]}...")
    return "\n".join(lines)


# --- MCP服务定义 ---
m = FastMCP("knowledge", port=cfg_service["port"], host=cfg_service["host"])
retriever = load_retriever_from_qdrant()
logging.info("✅ 检索器加载完成，服务准备就绪。")


@m.tool()
def search_multimodal_knowledge(query: str) -> str:
    """在多模态知识库中检索与查询最相关的信息（文本和图片）"""
    results = retriever.retrieve(query)
    return format_docs(results)


if __name__ == "__main__":
    logging.info(
        f"🚀 Knowledge MCP服务正在启动，监听于 http://{cfg_service['host']}:{cfg_service['port']}"
    )
    m.run(transport="streamable-http")
