"""Read an existing course index without silently creating an empty database."""

import os
import json
import sqlite3
import threading
from contextlib import closing
from inspect import signature
from pathlib import Path

from backend.services.storage_paths import StoragePathError, storage_access_path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_configured_path = Path(os.getenv("CHROMA_DB_PATH", "chroma_db"))
DB_PATH = str((_configured_path if _configured_path.is_absolute() else PROJECT_ROOT / _configured_path).resolve())
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "langchain")
_vectorstore = None
_load_lock = threading.Lock()


class KnowledgeBaseUnavailable(RuntimeError):
    """The existing local course index cannot currently be searched."""


def _embedding_model_options():
    # A normal course query must not unexpectedly wait for model downloads.
    value = os.getenv("EMBEDDING_LOCAL_FILES_ONLY", "true").strip().lower()
    if value not in {"true", "false", "1", "0"}:
        raise KnowledgeBaseUnavailable("EMBEDDING_LOCAL_FILES_ONLY 必须为 true 或 false。")
    return {"local_files_only": value in {"true", "1"}}


def _validate_existing_database():
    database_file = Path(DB_PATH) / "chroma.sqlite3"
    if not database_file.is_file():
        raise KnowledgeBaseUnavailable("教材知识库未安装，请将已有 chroma_db 资料恢复到项目目录。")
    manifest_file = Path(DB_PATH) / "rebuild_manifest.json"
    if manifest_file.exists():
        try:
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            ready = (manifest["status"] == "complete"
                     and type(manifest.get("chunk_count")) is int and manifest["chunk_count"] > 0
                     and manifest["collection_name"] == COLLECTION_NAME
                     and manifest["embedding_model"] == os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"))
        except (OSError, ValueError, TypeError, KeyError):
            ready = False
        if not ready:
            raise KnowledgeBaseUnavailable("重建知识库尚未完成或模型配置不匹配，请使用已验证的完整索引。")
    try:
        with closing(sqlite3.connect(database_file.as_uri() + "?mode=ro", uri=True)) as connection:
            count = connection.execute(
                'SELECT COUNT(*) FROM embeddings e JOIN segments s ON e.segment_id = s.id '
                'JOIN collections c ON s.collection = c.id WHERE c.name = ?',
                (COLLECTION_NAME,),
            ).fetchone()[0]
    except sqlite3.Error as exc:
        raise KnowledgeBaseUnavailable("教材知识库无法读取或版本不兼容，请检查已有知识库文件。") from exc
    if count == 0:
        raise KnowledgeBaseUnavailable("教材知识库为空或指定课程集合不存在，请恢复已建立的课程索引。")
    if manifest_file.exists() and count != manifest.get("chunk_count"):
        raise KnowledgeBaseUnavailable("重建知识库条目数与完成记录不符，请核对索引完整性。")


def get_vectorstore():
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore
    with _load_lock:
        if _vectorstore is not None:
            return _vectorstore
        # Validate in read-only mode before importing/loading embeddings or Chroma.
        _validate_existing_database()
        try:
            access_path = storage_access_path(DB_PATH)
        except StoragePathError as exc:
            raise KnowledgeBaseUnavailable(str(exc)) from exc
        try:
            from langchain_chroma import Chroma
            from langchain_huggingface import HuggingFaceEmbeddings

            embeddings = HuggingFaceEmbeddings(
                model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
                model_kwargs=_embedding_model_options(),
            )
            options = {}
            # Older supported LangChain releases lack this option; the read-only
            # check above still prevents their usual missing/empty-db creation.
            if "create_collection_if_not_exists" in signature(Chroma).parameters:
                options["create_collection_if_not_exists"] = False
            vectorstore = Chroma(
                persist_directory=access_path,
                collection_name=COLLECTION_NAME,
                embedding_function=embeddings,
                **options,
            )
            _vectorstore = vectorstore
        except Exception as exc:
            raise KnowledgeBaseUnavailable(
                "教材检索服务加载失败，请使用项目 .venv 运行环境检查，确认依赖兼容且本地嵌入模型已准备好。"
            ) from exc
    return _vectorstore


def similarity_search(query_text: str, top_k: int = 3) -> list:
    """Return textbook text, excluding generated image descriptions as evidence."""
    if not isinstance(query_text, str) or not query_text.strip():
        return []
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 20:
        raise ValueError("教材检索条数必须是 1 到 20 之间的整数。")
    try:
        results = get_vectorstore().similarity_search(
            query_text, k=top_k, filter={"source_type": {"$ne": "image"}},
        )
    except KnowledgeBaseUnavailable:
        raise
    except Exception as exc:
        raise KnowledgeBaseUnavailable("教材检索暂不可用，请检查本地知识库后重试。") from exc
    # Legacy textbook text has only Chapter metadata. Preserve those records,
    # while also guarding against a backend that returns an image despite filter.
    return [
        {"content": doc.page_content, "metadata": dict(doc.metadata)}
        for doc in results if doc.metadata.get("source_type") != "image"
    ]
