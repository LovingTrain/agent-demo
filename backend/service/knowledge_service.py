import logging
import os

import clip
from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from mcp.server.fastmcp import FastMCP  # 保留你的原有服务
from qdrant_client import QdrantClient

# ---- 配置 ----
DATA_DIR = "./knowledge/local"  # 你的知识目录
VECTOR_DB_PATH = "./db/knowledge"  # Qdrant 本地存储
TEXT_COLLECTION = "text_collection"
IMAGE_COLLECTION = "image_collection"


# ---- 多模态索引准备 ----
def build_multimodal_index():
    # 1. 初始化 Qdrant
    client = QdrantClient(path=VECTOR_DB_PATH)
    text_store = QdrantVectorStore(client=client, collection_name=TEXT_COLLECTION)
    image_store = QdrantVectorStore(client=client, collection_name=IMAGE_COLLECTION)
    storage_context = StorageContext.from_defaults(
        vector_store=text_store, image_store=image_store
    )

    # 2. 配置 embeddings
    clip_embedding, preprocess = clip.load(
        name="ViT-B/32",
        device="cuda",
        jit=False,
        download_root="./models/clip-vit-base-patch32",
    )
    text_embedding = HuggingFaceEmbedding(model_name="./models/bge-small-zh-v1.5")

    # 3. 自动读取所有文本和图片
    documents = SimpleDirectoryReader(DATA_DIR, recursive=True).load_data()
    logging.info(f"已加载 {len(documents)} 个文档（含文本和图片）")

    # 4. 构建索引
    index = MultiModalVectorStoreIndex.from_documents(
        documents,
        embed_model=text_embedding,
        storage_context=storage_context,
        show_progress=True,
    )
    return index


# ---- 检索器和格式化 ----
def get_multimodal_retriever(index):
    return index.as_retriever(similarity_top_k=3, image_similarity_top_k=1)


def format_docs(docs):
    if not docs:
        return "没有找到相关信息"
    lines = []
    for d in docs:
        meta = d.metadata
        path = meta.get("file_path", "")
        file_type = os.path.splitext(path)[-1].lower()
        content = d.get_content()
        if file_type in {".jpg", ".jpeg", ".png"}:
            lines.append(f"[图片] {path}")
        else:
            lines.append(f"[文本] {path}: {content[:100]}...")
    return "\n".join(lines)


# ---- MCP服务集成 ----
m = FastMCP("multimodal_knowledge", port=9200)
index = build_multimodal_index()
retriever = get_multimodal_retriever(index)


@m.tool()
def search_multimodal_knowledge(query: str) -> str:
    """支持文本和图片描述的多模态知识检索"""
    results = retriever.retrieve(query)
    return format_docs(results)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    m.run(transport="streamable-http")
