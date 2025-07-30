# server.py
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from mcp.server.fastmcp import FastMCP

# 向量库和检索初始化（和 v0.1 一致）
documents = [
    Document(
        page_content="派蒙喜欢吃蜜酱胡萝卜煎肉", metadata={"source": "character-doc"}
    ),
    Document(
        page_content="元素分为：火、水、冰、雷、风、岩、草，其中火、水、冰、雷属于活性元素...",
        metadata={"source": "element-doc"},
    ),
    Document(page_content="安柏和优菈是拉拉关系", metadata={"source": "character-doc"}),
    Document(
        page_content="班尼特的元素爆发会在当前位置释放一个圆形区域...",
        metadata={"source": "character-doc"},
    ),
    Document(
        page_content="BOSS草龙会释放无法躲避的草属性攻击...",
        metadata={"source": "boss-doc"},
    ),
]
embeddings = DashScopeEmbeddings(model="text-embedding-v1")
db_path = "./chroma_db"


def setup_vectorstore():
    import os
    import shutil

    if os.path.exists(db_path):
        try:
            store = Chroma(persist_directory=db_path, embedding_function=embeddings)
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
