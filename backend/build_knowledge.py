# build_knowledge_index.py
import logging

import clip
import toml
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# --- 配置与日志 ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
config = toml.load("config/knowledge.toml")
cfg_models = config["models"]
cfg_source = config["knowledge_source"]
cfg_db = config["vector_db"]


def build_index():
    """读取源文件，构建多模态索引并存入Qdrant。"""

    # 1. 初始化Qdrant客户端和存储
    logging.info(f"正在连接到Qdrant，路径: {cfg_db['path']}")
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

    # 2. 加载模型 (注意：这里LlamaIndex需要一个多模态LLM来处理图像，即使我们只用于嵌入)
    # 我们使用OpenAI的CLIP模型作为图像嵌入器
    Settings.embed_model = HuggingFaceEmbedding(model_name=cfg_models["text_embedding"])
    Settings.multi_modal_llm = OpenAIMultiModal(
        embed_model=cfg_models["image_embedding"], device=cfg_models["device"]
    )

    # 3. 读取源数据
    logging.info(f"正在从 '{cfg_source['data_dir']}' 递归读取所有文档...")
    documents = SimpleDirectoryReader(
        cfg_source["data_dir"], recursive=True
    ).load_data()
    logging.info(f"成功加载 {len(documents)} 个文档（包含文本和图片）。")
    if not documents:
        logging.warning("未找到任何文档，程序将退出。")
        return

    # 4. 构建并持久化索引
    logging.info("开始构建多模态索引，这可能需要一些时间...")
    index = MultiModalVectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )
    logging.info("✅ 索引构建完成并已成功存入Qdrant。")


if __name__ == "__main__":
    build_index()
