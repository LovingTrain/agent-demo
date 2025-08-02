import os
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from mcp.server.fastmcp import FastMCP
from chromadb import Settings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


def load_documents_from_dir(base_dir: str):
    docs = []
    base_dir = os.path.abspath(base_dir)

    # 分割器
    txt_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    md_header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "H1"), ("##", "H2"), ("###", "H3")])

    for root, _, files in os.walk(base_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            if fname.lower().endswith(".txt"):
                # TXT 文件直接按固定长度切分
                with open(fpath, encoding="utf-8") as f:
                    content = f.read()
                rel_path = os.path.relpath(fpath, base_dir)
                parts = rel_path.split(os.sep)[:-1]
                if not parts:
                    continue
                source = parts[0]
                tags = list(dict.fromkeys(parts))
                # 先分块
                sub_docs = txt_splitter.create_documents([content])
                for d in sub_docs:
                    d.metadata.update({
                        "source": source,
                        "tags": ",".join(tags),
                        "file_path": rel_path,
                        "file_type": "txt"
                    })
                docs.extend(sub_docs)

            elif fname.lower().endswith(".md"):
                # MD 文件先按章节，再对每章节分块
                with open(fpath, encoding="utf-8") as f:
                    content = f.read()
                rel_path = os.path.relpath(fpath, base_dir)
                parts = rel_path.split(os.sep)[:-1]
                if not parts:
                    continue
                source = parts[0]
                tags = list(dict.fromkeys(parts))
                # 先按标题分章节
                chapter_docs = md_header_splitter.split_text(content)
                # 每个章节再切块
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

            # 其他类型可扩展

    return docs


# 向量库和检索初始化（和 v0.1 一致）
documents = load_documents_from_dir("./knowledge/local")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
db_path = "./db/knowledge"


def setup_vectorstore():
    import os
    import shutil

    if os.path.exists(db_path):
        try:
            store = Chroma(
                persist_directory=db_path,
                embedding_function=embeddings,
                client_settings=Settings(anonymized_telemetry=False),
            )
            if store._collection.count() != len(documents):
                raise ValueError("数量不一致，重建 DB")
            return store
        except Exception:
            shutil.rmtree(db_path)
    return Chroma.from_documents(
        documents, embedding=embeddings, persist_directory=db_path
    )


vectorstore = setup_vectorstore()
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})


def format_docs(docs):
    return " | ".join(d.page_content for d in docs) if docs else "没有找到相关信息"


m = FastMCP("knowledge", port=9000)


@m.tool()
def search_game_knowledge(query: str) -> str:
    """搜索游戏知识库中的角色、元素、技能、BOSS 等内容。"""
    return format_docs(retriever.invoke(query))


# Mock 角色面板数据
雷国_mock数据 = {
    "队伍": ["雷电将军", "行秋", "香菱", "班尼特"],
    "示例属性": {
        "雷电将军": {
            "等级": 90,
            "武器": "薙草之稻光",
            "圣遗物": "绝缘4",
            "天赋等级": [10, 10, 10],
        },
        "行秋": {
            "等级": 90,
            "武器": "祭礼剑",
            "圣遗物": "沉沦4",
            "天赋等级": [8, 12, 8],
        },
        "香菱": {
            "等级": 90,
            "武器": "「渔获」",
            "圣遗物": "绝缘4",
            "天赋等级": [10, 13, 12],
        },
        "班尼特": {
            "等级": 90,
            "武器": "风鹰剑",
            "圣遗物": "宗室4",
            "天赋等级": [9, 13, 11],
        },
    },
    "备注": "这是经典雷国配队Mock数据，仅供测试",
}


@m.tool()
def query_character_by_uid(uid: str) -> str:
    """根据UID获取角色面板信息（Mock：始终返回雷国队经典配置）"""
    return str(雷国_mock数据)


@m.tool()
def calc_damage(character_json: str, team: str = "") -> str:
    """输入角色面板JSON和配队描述，输出预估伤害（Mock版）"""
    # 只需演示多步串联即可
    return f"已收到角色面板，配队为：{team if team else '雷国'}。模拟计算总爆发伤害为：153264。"


if __name__ == "__main__":
    m.run(transport="streamable-http")
