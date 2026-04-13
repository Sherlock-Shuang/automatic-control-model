import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

DB_PATH = "./chroma_db"

_vectorstore = None

def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        print("⏳ 正在加载本地向量模型和图文数据库 (Chroma)...")
        # Ensure we don't hit threading issues by caching it per process
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning)
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
        _vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    return _vectorstore

def similarity_search(query_text: str, top_k: int = 3) -> list:
    """
    Perform a vector similarity search locally using Chroma DB.
    """
    try:
        vs = get_vectorstore()
        results = vs.similarity_search(query_text, k=top_k)
        return [{"content": doc.page_content, "metadata": doc.metadata} for doc in results]
    except Exception as e:
        print(f"Chroma DB search error: {e}")
        return []
