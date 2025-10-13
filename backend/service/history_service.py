# backend/service/history_service.py
from __future__ import annotations

from typing import Optional, List

import toml
from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker

from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_qdrant import QdrantVectorStore


class HistoryService:
    def __init__(self, config_path: str = "config/history.toml"):
        cfg = toml.load(config_path)
        self.cfg_models = cfg["models"]
        self.cfg_db = cfg["vector_db"]

        # 占位属性，延迟初始化
        self.embeddings: Optional[HuggingFaceEmbeddings] = None
        self.cross_encoder: Optional[HuggingFaceCrossEncoder] = None
        self.compressor: Optional[CrossEncoderReranker] = None
        self.qdrant_client: Optional[QdrantClient] = None
        self.vectorstore: Optional[QdrantVectorStore] = None

    async def ainit(self):
        # 初始化模型
        self.embeddings = HuggingFaceEmbeddings(model_name=self.cfg_models["embedding"])
        self.cross_encoder = HuggingFaceCrossEncoder(
            model_name=self.cfg_models["reranker"]
        )
        self.compressor = CrossEncoderReranker(model=self.cross_encoder, top_n=3)

        # 初始化向量库
        self.qdrant_client = QdrantClient(path=self.cfg_db["path"])
        collection_name = self.cfg_db["collection_name"]

        if not self.qdrant_client.collection_exists(collection_name=collection_name):
            print(
                f"Collection '{collection_name}' does not exist, it will be created now"
            )
            # 动态探测向量维度
            vector_size = len(self.embeddings.embed_query("test"))
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size, distance=models.Distance.COSINE
                ),
            )
            print(f"✅ Collection '{collection_name}' is created successfully")
        else:
            print(f"✅ Collection '{collection_name}' already exists")

        self.vectorstore = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=collection_name,
            embedding=self.embeddings,
        )

    async def add_history(self, *, session_id: str, role: str, message: str) -> str:
        try:
            enriched_content = f"[{role}]: {message}"
            doc = Document(
                page_content=enriched_content,
                metadata={"session_id": session_id, "role": role},
            )
            # 使用异步接口写入
            await self.vectorstore.aadd_documents([doc])
            return "ok"
        except Exception as e:
            msg = f"❌ Error: Could not add history to the database. Reason: {e}"
            print(msg)
            return msg

    async def search_history(self, *, session_id: str, query: str, k: int = 3) -> str:
        try:
            # 构建会话过滤器
            retriever_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.session_id",
                        match=models.MatchValue(value=session_id),
                    )
                ]
            )
            # 基础检索召回更多候选
            base_retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"filter": retriever_filter, "k": 20},
            )
            # 重排裁剪
            self.compressor.top_n = k
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=self.compressor, base_retriever=base_retriever
            )
            reranked_docs: List[Document] = await compression_retriever.ainvoke(query)
            hits = [d.page_content for d in reranked_docs]
            return "\n".join(hits) if hits else "(无相关历史)"
        except Exception as e:
            msg = f"❌ Error: Could not search history. Reason: {e}"
            print(msg)
            return msg
