# history_mcp_server.py
import toml
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from mcp.server.fastmcp import FastMCP
from qdrant_client import QdrantClient
from qdrant_client.http import models

# 加载配置
config = toml.load("config/history.toml")
cfg_service = config["service"]
cfg_models = config["models"]
cfg_db = config["vector_db"]

# 初始化模型
embeddings = HuggingFaceEmbeddings(model_name=cfg_models["embedding"])
reranker_model = HuggingFaceCrossEncoder(model_name=cfg_models["reranker"])
compressor = CrossEncoderReranker(model=reranker_model, top_n=3)

# 设置Qdrant向量库
client = QdrantClient(path=cfg_db["path"])
collection_name = cfg_db["collection_name"]
if not client.collection_exists(collection_name=collection_name):
    print(f"Collection '{collection_name}' does not exist, it will be created now")
    # 动态获取嵌入向量的维度，避免硬编码
    vector_size = len(embeddings.embed_query("test"))
    print(f"Vector size detected: {vector_size}")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size, distance=models.Distance.COSINE
        ),
    )
    print(f"✅ Collection '{collection_name}' is created successfully")
else:
    print(f"✅ Collection '{collection_name}' already exists")

history_vectorstore = QdrantVectorStore(
    client=client,
    collection_name=cfg_db["collection_name"],
    embedding=embeddings,
)

# 定义MCP服务
m = FastMCP("history", port=cfg_service["port"], host=cfg_service["host"])


@m.tool()
def add_history(session_id: str, role: str, message: str) -> str:
    """将对话历史写入向量库"""
    try:
        enriched_content = f"[{role}]: {message}"
        doc = Document(
            page_content=enriched_content,
            metadata={"session_id": session_id, "role": role},
        )
        history_vectorstore.add_documents([doc])
        return "ok"
    except Exception as e:
        msg = f"❌ Error: Could not add history to the database. Reason: {e}"
        print(msg)
        return msg


@m.tool()
def search_history(session_id: str, query: str, k: int = 3) -> str:
    """向量检索并重排历史消息"""
    try:
        retriever_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.session_id", match=models.MatchValue(value=session_id)
                )
            ]
        )
        base_retriever = history_vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "filter": retriever_filter,
                "k": 20,  # 召回更多候选给 reranker
            },
        )

        compressor.top_n = k
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=base_retriever
        )

        reranked_docs = compression_retriever.invoke(query)
        hits = [d.page_content for d in reranked_docs]

        return "\n".join(hits) if hits else "(无相关历史)"
    except Exception as e:
        msg = f"❌ Error: Could not search history. Reason: {e}"
        print(msg)
        return msg


if __name__ == "__main__":
    print("🚀 History MCP service is starting up ...")
    m.run(transport="stdio")
