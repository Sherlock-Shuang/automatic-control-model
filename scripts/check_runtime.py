"""Offline runtime checks; --search additionally opens the existing Chroma index.

The default database check uses SQLite mode=ro. Explicit --search loads an already
cached embedding model and Chroma, which may perform its normal schema migration.
Configuration follows the application's .env. No credentials are printed, no AI
service is called, and missing models are never downloaded.
"""

import argparse
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import sqlite3
import sys
from contextlib import closing


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPENDENCIES = (
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("streamlit", "streamlit"),
    ("langchain_openai", "langchain-openai"),
    ("langchain_chroma", "langchain-chroma"),
    ("langchain_huggingface", "langchain-huggingface"),
    ("sentence_transformers", "sentence-transformers"),
    ("transformers", "transformers"),
    ("torch", "torch"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("pyarrow", "pyarrow"),
    ("fitz", "PyMuPDF"),
    ("PIL", "Pillow"),
    ("dotenv", "python-dotenv"),
)


def set_offline_mode():
    # Force offline even if the invoking terminal has explicitly enabled downloads.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["EMBEDDING_LOCAL_FILES_ONLY"] = "true"
    os.environ["ANONYMIZED_TELEMETRY"] = "False"


def load_configuration():
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
    except Exception as error:
        print(f"[失败] 无法加载项目配置（{exception_types(error)}）。请检查项目环境中的 python-dotenv。")
        return False
    return True


def exception_types(error):
    """Show diagnostics without arbitrary exception text, URLs, or environment data."""
    names = []
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        names.append(type(error).__name__)
        error = error.__cause__ or error.__context__
    return " -> ".join(names)


def check_interpreter():
    expected = PROJECT_ROOT / ".venv"
    if Path(sys.prefix).resolve() != expected.resolve():
        print("[失败] 当前没有使用项目的 .venv，请运行 .\\.venv\\Scripts\\python.exe scripts/check_runtime.py")
        return False
    print(f"[通过] 项目隔离环境，Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True


def check_dependencies():
    passed = True
    for module, distribution in DEPENDENCIES:
        try:
            importlib.import_module(module)
            version = metadata.version(distribution)
        except Exception as error:
            print(f"[失败] {distribution} 无法导入（{exception_types(error)}）。请在项目 .venv 中安装完整 requirements.txt。")
            passed = False
        else:
            print(f"[通过] {distribution} {version}")
    return passed


def database_count(database_dir, collection_name):
    """Read only an existing nonempty collection; never construct a Chroma client."""
    database = Path(database_dir).resolve() / "chroma.sqlite3"
    if not database.is_file():
        raise FileNotFoundError("Existing textbook database is missing")
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM embeddings e JOIN segments s ON e.segment_id = s.id "
            "JOIN collections c ON s.collection = c.id WHERE c.name = ?",
            (collection_name,),
        ).fetchone()[0]
    if count == 0:
        raise ValueError("Existing textbook collection is empty or missing")
    return count


def check_database():
    try:
        from backend.services import local_db

        local_db._validate_existing_database()
        count = database_count(local_db.DB_PATH, local_db.COLLECTION_NAME)
    except Exception as error:
        print(f"[失败] 已有教材索引缺失、为空、重建未完成、配置不匹配或无法读取（{exception_types(error)}）。请恢复完整索引并核对项目配置；本工具不会新建空索引。")
        return False
    print(f"[通过] 教材索引只读检查：{count} 条记录。")
    return True


def compact_metadata(values):
    fields = ("source_type", "chapter", "Chapter", "page", "page_number", "source", "image_path")
    return {
        key: " ".join(str(values[key]).split())[:100]
        for key in fields if values.get(key) is not None
    }


def check_search():
    print("[检索] 正在加载本地已有模型和 Chroma；Chroma 可能迁移已有索引格式，此步骤不是只读检查。")
    try:
        from backend.services.local_db import similarity_search

        results = similarity_search("闭环系统稳定性与特征方程", top_k=3)
        if not results:
            raise ValueError("No textbook search results")
    except Exception as error:
        print(f"[失败] 本地教材检索不可用（{exception_types(error)}）。请检查模型缓存和项目依赖；未尝试下载模型。")
        return False
    print(f"[通过] 本地教材检索返回 {len(results)} 条记录。")
    for index, result in enumerate(results, 1):
        print(f"  来源 {index}: {json.dumps(compact_metadata(result.get('metadata', {})), ensure_ascii=False)}")
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", action="store_true", help="离线加载已有模型并检索；Chroma 可能进行正常索引迁移")
    args = parser.parse_args(argv)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    configuration_ok = load_configuration()
    set_offline_mode()
    print("本地环境检查：沿用项目配置，不输出密钥、不调用远程模型、不下载模型。")
    interpreter_ok = check_interpreter()
    dependencies_ok = check_dependencies()
    database_ok = check_database()
    search_ok = True
    if args.search:
        if configuration_ok and interpreter_ok and dependencies_ok and database_ok:
            search_ok = check_search()
        else:
            print("[跳过] 基础检查未通过，未加载教材模型。")
            search_ok = False
    return 0 if all((configuration_ok, interpreter_ok, dependencies_ok, database_ok, search_ok)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
