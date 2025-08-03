import os
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from transformers import CLIPProcessor, CLIPModel
from mcp.server.fastmcp import FastMCP
from chromadb import Settings
import torch
from PIL import Image
import numpy as np


# 定义路径
TEXT_VECTOR_PATH = "./db/knowledge/text"
IMAGE_VECTOR_PATH = "./db/knowledge/image"

# 加载文档和图像
def load_documents_from_dir(base_dir: str):
    docs = []
    base_dir = os.path.abspath(base_dir)

    # 文本处理
    txt_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    md_header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "H1"), ("##", "H2"), ("###", "H3")])

    # 图像处理
    image_paths = []
    
    for root, _, files in os.walk(base_dir):
        for fname in files:
            fpath = os.path.join(root, fname)

            # 处理TXT文件
            if fname.lower().endswith(".txt"):
                with open(fpath, encoding="utf-8") as f:
                    content = f.read()
                rel_path = os.path.relpath(fpath, base_dir)
                parts = rel_path.split(os.sep)[:-1]
                if not parts:
                    continue
                source = parts[0]
                tags = list(dict.fromkeys(parts))
                # 分块
                sub_docs = txt_splitter.create_documents([content])
                for d in sub_docs:
                    d.metadata.update({
                        "source": source,
                        "tags": ",".join(tags),
                        "file_path": rel_path,
                        "file_type": "txt"
                    })
                docs.extend(sub_docs)

            # 处理MD文件
            elif fname.lower().endswith(".md"):
                with open(fpath, encoding="utf-8") as f:
                    content = f.read()
                rel_path = os.path.relpath(fpath, base_dir)
                parts = rel_path.split(os.sep)[:-1]
                if not parts:
                    continue
                source = parts[0]
                tags = list(dict.fromkeys(parts))
                chapter_docs = md_header_splitter.split_text(content)
                for chap_doc in chapter_docs:
                    sub_docs = txt_splitter.create_documents([chap_doc.page_content])
                    for d in sub_docs:
                        d.metadata.update({
                            "source": source,
                            "tags": ",".join(tags),
                            "file_path": rel_path,
                            "file_type": "md",
                            "header": chap_doc.metadata.get("header", ""),
                        })
                    docs.extend(sub_docs)

            # 处理图片文件
            elif fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(fpath)

    return docs, image_paths


# 图像处理
class CLIPImageEmbedding:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch16", device: str = "cpu"):
        # Load the CLIP model and processor
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name, use_fast=False)
        self.device = device

    def embed_images(self, image_paths: list) -> list:
        embeddings = []
        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                features = self.model.get_image_features(**inputs)
            features = features / features.norm(p=2, dim=-1, keepdim=True)  # Normalize
            embeddings.append(features.squeeze().cpu().numpy())
        return embeddings
    
    # 添加 embed_query 方法来处理查询
    def embed_query(self, query: str) -> list:
        # CLIP模型处理文本查询
        inputs = self.processor(text=query, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)  # Normalize
        return text_features.squeeze().cpu().numpy()

    # 适配 embed_documents 方法
    def embed_documents(self, texts: list) -> list:
        # 使用 CLIP 模型的特征维度来生成空的嵌入
        # 直接从模型输出获取特征维度，CLIP的输出维度通常是 768
        features = self.model.get_image_features(**self.processor(images=Image.new('RGB', (224, 224)), return_tensors="pt").to(self.device))
        feature_size = features.shape[1]  # 获取特征的维度
        return np.zeros((len(texts), feature_size))  # 适配为文本嵌入的形状


# 设置向量库和检索
def setup_vectorstore(documents, embeddings, vector_path):
    import os
    import shutil

    # 如果需要删除旧的数据库
    if os.path.exists(vector_path):
        try:
            store = Chroma(persist_directory=vector_path, embedding_function=embeddings)
            
            # 删除所有文档：使用通配符匹配所有文档
            store._collection.delete(where={"metadata": {"*": "*"}})
            
            if store._collection.count() != len(documents):
                raise ValueError("数量不一致，重建 DB")
            return store
        except Exception:
            shutil.rmtree(vector_path)

    # 如果没有，则使用新的文档和嵌入来初始化向量库
    return Chroma.from_documents(documents, embedding=embeddings, persist_directory=vector_path)


# 加载文档并生成嵌入
text_documents, image_paths = load_documents_from_dir("./knowledge/local")

# 处理图像并生成嵌入
clip_embedding = CLIPImageEmbedding()
image_embeddings = clip_embedding.embed_images(image_paths)

# 将图像嵌入转换为文档，包含图像路径和嵌入
image_documents = []
for idx, embedding in enumerate(image_embeddings):
    image_document = Document(
        page_content=str(embedding),  # 可以存储嵌入或其他元数据
        metadata={"file_path": image_paths[idx], "file_type": "image"}
    )
    image_documents.append(image_document)

# 文本嵌入
text_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5", model_kwargs={"device": "cpu"})

# 设置文本和图像向量库
text_vectorstore = setup_vectorstore(documents=text_documents, embeddings=text_embeddings, vector_path=TEXT_VECTOR_PATH)
image_vectorstore = setup_vectorstore(documents=image_documents, embeddings=clip_embedding, vector_path=IMAGE_VECTOR_PATH)

# 检索器
text_retriever = text_vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
image_retriever = image_vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# 格式化输出
def format_docs(docs):
    return " | ".join(d.page_content for d in docs) if docs else "没有找到相关信息"


# 同时支持文本和图像检索
def multimodal_retrieval(query: str):
    text_results = text_retriever.invoke(query)
    image_results = image_retriever.invoke(query)
    
    # 合并文本和图像的检索结果
    combined_results = text_results + image_results
    
    return format_docs(combined_results)


m = FastMCP("multimodal_knowledge", port=9000)


@m.tool()
def search_multimodal_knowledge(query: str) -> str:
    """同时在文本和图像知识库中检索相关信息"""
    return multimodal_retrieval(query)


if __name__ == "__main__":
    m.run(transport="streamable-http")
