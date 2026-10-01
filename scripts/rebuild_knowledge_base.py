"""Rebuild an incompatible Chroma index from its read-only SQLite documents.

The default is a dry run. A real run only uses the locally cached embedding
model, writes to a new/empty destination, and never opens the source in Chroma.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
COLLECTION_NAME = "langchain"
PROVENANCE_KEYS = {"source_document_id", "chunk_index", "char_start", "char_end"}


class RebuildError(ValueError):
    """Refuse a rebuild that could lose data or overwrite an existing index."""


@dataclass(frozen=True)
class SourceDocument:
    document_id: str
    text: str
    metadata: dict[str, str | int | float | bool]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    metadata: dict[str, str | int | float | bool]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_paths(source: Path, destination: Path) -> tuple[Path, Path]:
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    database = source / "chroma.sqlite3" if source.is_dir() else source
    if not database.is_file():
        raise RebuildError("Source SQLite database does not exist.")
    source_directory = database.parent
    if (destination == source_directory or source_directory in destination.parents
            or destination in source_directory.parents):
        raise RebuildError("Source and destination must be separate, non-nested directories.")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise RebuildError("Destination must be absent or an empty directory; nothing will be overwritten.")
    return database, destination


def _metadata_value(row: sqlite3.Row, columns: set[str]) -> str | int | float | bool:
    value_columns = [name for name in ("string_value", "int_value", "float_value", "bool_value")
                     if name in columns and row[name] is not None]
    if len(value_columns) != 1:
        raise RebuildError("Every metadata value must have exactly one supported scalar type.")
    name = value_columns[0]
    value = row[name]
    if name == "string_value" and type(value) is str:
        return value
    if name == "int_value" and type(value) is int:
        return value
    if name == "float_value" and type(value) in (int, float) and math.isfinite(value):
        return float(value)
    if name == "bool_value" and type(value) is int and value in (0, 1):
        return bool(value)
    raise RebuildError("Metadata contains a malformed scalar value.")


def read_source_documents(database: Path) -> list[SourceDocument]:
    """Extract every document in the named collection, with no Chroma imports."""
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        required = {
            "collections": {"id", "name"},
            "segments": {"id", "collection"},
            "embeddings": {"id", "embedding_id", "segment_id"},
            "embedding_metadata": {"id", "key", "string_value", "int_value", "float_value"},
        }
        schema = {}
        for table, fields in required.items():
            schema[table] = {row["name"] for row in connection.execute(f"PRAGMA table_info({table})")}
            if not fields <= schema[table]:
                raise RebuildError(f"Unsupported SQLite schema in {table}.")
        unknown_values = {name for name in schema["embedding_metadata"]
                          if name.endswith("_value")} - {"string_value", "int_value", "float_value", "bool_value"}
        if unknown_values:
            raise RebuildError("Unsupported metadata value columns; refusing to discard data.")
        collections = connection.execute("SELECT id FROM collections WHERE name=?", (COLLECTION_NAME,)).fetchall()
        if len(collections) != 1:
            raise RebuildError("Exactly one langchain collection must exist.")
        records = connection.execute(
            "SELECT e.id, e.embedding_id FROM embeddings e "
            "JOIN segments s ON e.segment_id=s.id WHERE s.collection=? ORDER BY e.id",
            (collections[0]["id"],),
        ).fetchall()
        if not records:
            raise RebuildError("Source collection has no documents.")
        metadata_rows = connection.execute(
            "SELECT m.* FROM embedding_metadata m JOIN embeddings e ON m.id=e.id "
            "JOIN segments s ON e.segment_id=s.id WHERE s.collection=? ORDER BY m.id, m.key",
            (collections[0]["id"],),
        )
        by_id: dict[int, dict[str, Any]] = {}
        for row in metadata_rows:
            key = row["key"]
            if type(key) is not str or not key:
                raise RebuildError("Metadata has a missing or invalid key.")
            values = by_id.setdefault(row["id"], {})
            if key in values:
                raise RebuildError("Duplicate metadata key in source.")
            values[key] = _metadata_value(row, schema["embedding_metadata"])
        documents = []
        seen_ids: set[str] = set()
        for row in records:
            document_id = row["embedding_id"]
            if type(document_id) is not str or not document_id.strip() or document_id in seen_ids:
                raise RebuildError("Source document ID is missing, invalid or duplicated.")
            seen_ids.add(document_id)
            values = by_id.get(row["id"], {}).copy()
            text = values.pop("chroma:document", None)
            if type(text) is not str or not text.strip():
                raise RebuildError(f"Document {document_id!r} has missing, empty or non-text content.")
            if PROVENANCE_KEYS & values.keys():
                raise RebuildError("Source metadata already uses a reserved provenance key.")
            documents.append(SourceDocument(document_id, text, values))
        return documents
    except sqlite3.Error as exc:
        raise RebuildError("Cannot read the source SQLite schema/documents.") from exc
    finally:
        connection.close()


def split_documents(
    documents: list[SourceDocument], chunk_size: int = 450, overlap: int = 75,
    token_count: Callable[[str], int] | None = None, max_tokens: int | None = None,
) -> list[Chunk]:
    """Keep exact source slices and verify tokens before the model can truncate."""
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise RebuildError("Chunk size must be positive and overlap must be smaller than it.")
    if (token_count is None) != (max_tokens is None) or (max_tokens is not None and max_tokens <= 0):
        raise RebuildError("Token counter and a positive model limit must be supplied together.")
    chunks = []
    for document in documents:
        start, index = 0, 0
        while start < len(document.text):
            end = min(start + chunk_size, len(document.text))
            if token_count is not None:
                while token_count(document.text[start:end]) > max_tokens:
                    if end - start <= 1:
                        raise RebuildError("Even one source character exceeds the embedding token limit.")
                    end = start + (end - start) // 2
            text = document.text[start:end]
            metadata = dict(document.metadata)
            metadata.update(source_document_id=document.document_id, chunk_index=index,
                            char_start=start, char_end=end)
            chunk_id = hashlib.sha256(f"{document.document_id}\0{start}\0{end}".encode("utf-8")).hexdigest()
            chunks.append(Chunk(chunk_id, text, metadata))
            index += 1
            if end == len(document.text):
                break
            start = max(start + 1, end - overlap)
    if not chunks:
        raise RebuildError("Refusing to produce an empty index.")
    return chunks


def force_offline() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["EMBEDDING_LOCAL_FILES_ONLY"] = "true"
    os.environ["ANONYMIZED_TELEMETRY"] = "False"


class LocalEncoder:
    def __init__(self, model_name: str):
        force_offline()
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, local_files_only=True, device="cpu")
        self.max_tokens = int(self.model.max_seq_length)
        self.dimension = int(self.model.get_sentence_embedding_dimension())
        # Hidden prompt prefixes would invalidate character/token provenance.
        if getattr(self.model, "default_prompt_name", None):
            raise RebuildError("Embedding models with an implicit prompt are not supported.")

    def token_count(self, text: str) -> int:
        return len(self.model.tokenizer(text, add_special_tokens=True, truncation=False,
                                        return_attention_mask=False)["input_ids"])

    def encode(self, texts: list[str]) -> list[list[float]]:
        if any(self.token_count(text) > self.max_tokens for text in texts):
            raise RebuildError("Embedding input exceeds model token limit; refusing silent truncation.")
        return self.model.encode(texts, batch_size=len(texts), show_progress_bar=False,
                                 normalize_embeddings=False, convert_to_numpy=True).tolist()


def create_collection(destination: Path):
    force_offline()
    import chromadb
    from chromadb.config import Settings
    from backend.services.storage_paths import storage_access_path

    client = chromadb.PersistentClient(path=storage_access_path(destination),
                                      settings=Settings(anonymized_telemetry=False))
    return client.create_collection(name=COLLECTION_NAME, embedding_function=None)


_REOPEN_CHECK_CODE = """
import json, os, sys
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['EMBEDDING_LOCAL_FILES_ONLY'] = 'true'
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
payload = json.load(sys.stdin)
try:
    import chromadb
    from chromadb.config import Settings
    client = chromadb.PersistentClient(path=payload['path'], settings=Settings(anonymized_telemetry=False))
    collection = client.get_collection(payload['collection'], embedding_function=None)
    count = collection.count()
    result = collection.query(query_embeddings=[payload['query_vector']],
                              n_results=min(3, payload['expected_count']), include=['distances'])
    print(json.dumps({'count': count, 'ids': result['ids'][0], 'distances': result['distances'][0]}))
except Exception as exc:
    print(json.dumps({'error_type': type(exc).__name__}))
    sys.exit(1)
"""


def verify_persisted_index(destination: Path, expected_ids: set[str], query_vector: list[float]) -> dict:
    """Verify disk files and reopen in a process with no in-memory Chroma cache."""
    from backend.services.storage_paths import storage_access_path

    database = destination / "chroma.sqlite3"
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        segments = connection.execute(
            "SELECT s.id FROM segments s JOIN collections c ON s.collection=c.id "
            "WHERE c.name=? AND s.type='urn:chroma:segment/vector/hnsw-local-persisted'",
            (COLLECTION_NAME,),
        ).fetchall()
    finally:
        connection.close()
    if len(segments) != 1:
        raise RebuildError("Persisted index must contain exactly one HNSW vector segment.")
    # A metadata pickle alone is not an index: native HNSW may fail to write
    # Windows Unicode paths while metadata and in-memory queries still succeed.
    folder = destination / segments[0][0]
    binaries = ("header.bin", "data_level0.bin", "length.bin", "link_lists.bin")
    if any(not (folder / name).is_file() for name in binaries):
        raise RebuildError("Persisted HNSW binary files are missing; in-memory success is insufficient.")
    if (folder / "header.bin").stat().st_size == 0:
        raise RebuildError("Persisted HNSW header is empty.")
    payload = {"path": storage_access_path(destination), "collection": COLLECTION_NAME,
               "expected_count": len(expected_ids), "query_vector": query_vector}
    try:
        result = subprocess.run(
            [sys.executable, "-c", _REOPEN_CHECK_CODE], input=json.dumps(payload),
            capture_output=True, text=True, encoding="utf-8", timeout=60, check=False, shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RebuildError("Independent-process index reopen check timed out.") from exc
    if result.returncode != 0:
        raise RebuildError("Independent-process index reopen/query failed; index is not ready.")
    try:
        check = json.loads(result.stdout)
    except (ValueError, TypeError) as exc:
        raise RebuildError("Independent-process index check returned an invalid result.") from exc
    if (check.get("count") != len(expected_ids) or not check.get("ids")
            or any(id_ not in expected_ids for id_ in check["ids"])):
        raise RebuildError("Independent-process index count or retrieval validation failed.")
    return {"verified": True, "count": check["count"], "ids": check["ids"],
            "distances": check.get("distances", []), "hnsw_binary_files": list(binaries)}


def _validated_vectors(encoder, texts: list[str]) -> list[list[float]]:
    vectors = encoder.encode(texts)
    if len(vectors) != len(texts) or any(
        len(vector) != encoder.dimension or any(not math.isfinite(value) for value in vector)
        for vector in vectors
    ):
        raise RebuildError("Embedding output has missing, invalid or inconsistent vectors.")
    return vectors


def write_index(
    database: Path, destination: Path, documents: list[SourceDocument], chunks: list[Chunk],
    encoder, model_name: str, chunk_size: int, overlap: int, batch_size: int,
) -> dict[str, Any]:
    # Recheck immediately before writing, after model loading and tokenization.
    validate_paths(database, destination)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1, "status": "building", "created_at": datetime.now(timezone.utc).isoformat(),
        "source_sqlite": str(database), "source_sqlite_sha256": sha256_file(database),
        "rebuild_script_sha256": sha256_file(Path(__file__).resolve()),
        "source_document_count": len(documents), "source_character_count": sum(len(d.text) for d in documents),
        "chunk_count": len(chunks), "collection_name": COLLECTION_NAME, "embedding_model": model_name,
        "embedding_dimension": encoder.dimension, "model_max_tokens": encoder.max_tokens,
        "chunk_size": chunk_size, "chunk_overlap": overlap, "batch_size": batch_size,
        "offline": True, "normalize_embeddings": False,
    }
    manifest_path = destination / "rebuild_manifest.json"

    def save_manifest():
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(manifest_path)

    save_manifest()
    try:
        # Preserve the complete unmodified original documents as well as chunks.
        archive = destination / "source_documents.jsonl"
        with archive.open("x", encoding="utf-8", newline="\n") as stream:
            for document in documents:
                stream.write(json.dumps({"id": document.document_id, "text": document.text,
                                         "metadata": document.metadata}, ensure_ascii=False) + "\n")
        manifest["source_documents_sha256"] = sha256_file(archive)
        collection = create_collection(destination)
        for offset in range(0, len(chunks), batch_size):
            batch = chunks[offset:offset + batch_size]
            vectors = _validated_vectors(encoder, [item.text for item in batch])
            collection.add(ids=[item.chunk_id for item in batch], embeddings=vectors,
                           documents=[item.text for item in batch], metadatas=[item.metadata for item in batch])
            completed = offset + len(batch)
            if completed % 512 == 0 or completed == len(chunks):
                print(json.dumps({"indexed_chunks": completed, "total_chunks": len(chunks)}), flush=True)
        if collection.count() != len(chunks):
            raise RebuildError("Written index count does not match the planned chunk count.")
        # Verify stored content/metadata for every chunk, not only the count.
        for offset in range(0, len(chunks), batch_size):
            batch = chunks[offset:offset + batch_size]
            stored = collection.get(ids=[item.chunk_id for item in batch], include=["documents", "metadatas"])
            rows = dict(zip(stored["ids"], zip(stored["documents"], stored["metadatas"])))
            if len(rows) != len(batch) or any(rows.get(item.chunk_id) != (item.text, item.metadata) for item in batch):
                raise RebuildError("Stored chunk content or metadata differs from the source.")
        smoke_query = "连续时间线性系统闭环极点与稳定性"
        smoke_vector = _validated_vectors(encoder, [smoke_query])[0]
        smoke = collection.query(query_embeddings=[smoke_vector],
                                 n_results=min(3, len(chunks)), include=["documents", "metadatas", "distances"])
        valid_ids = {chunk.chunk_id for chunk in chunks}
        if not smoke.get("ids") or not smoke["ids"][0] or any(
            chunk_id not in valid_ids for chunk_id in smoke["ids"][0]
        ):
            raise RebuildError("Retrieval smoke test returned no valid source chunks.")
        manifest["smoke_test"] = {"query": smoke_query, "ids": smoke["ids"][0],
                                  "distances": smoke.get("distances", [[]])[0]}
        manifest["independent_process_check"] = verify_persisted_index(destination, valid_ids, smoke_vector)
        manifest["status"] = "complete"
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        save_manifest()
        return manifest
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error_type"] = type(exc).__name__
        save_manifest()
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--run", action="store_true", help="Actually build; otherwise only inspect and count.")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--chunk-size", type=int, default=450)
    parser.add_argument("--overlap", type=int, default=75)
    parser.add_argument("--batch-size", type=int, choices=(32, 64), default=32)
    args = parser.parse_args(argv)
    try:
        database, destination = validate_paths(args.source, args.destination)
        documents = read_source_documents(database)
        chunks = split_documents(documents, args.chunk_size, args.overlap)
        if not args.run:
            print(json.dumps({"dry_run": True, "source_document_count": len(documents),
                              "source_character_count": sum(len(d.text) for d in documents),
                              "character_chunk_count": len(chunks), "token_validation": "not_performed",
                              "destination": str(destination)}, ensure_ascii=False, indent=2))
            return 0
        force_offline()
        encoder = LocalEncoder(args.model)
        chunks = split_documents(documents, args.chunk_size, args.overlap,
                                 encoder.token_count, encoder.max_tokens)
        print(json.dumps({"phase": "token_validation_complete", "source_document_count": len(documents),
                          "chunk_count": len(chunks), "model_max_tokens": encoder.max_tokens}), flush=True)
        manifest = write_index(database, destination, documents, chunks, encoder, args.model,
                               args.chunk_size, args.overlap, args.batch_size)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        # Model initialization errors should not expose environment credentials.
        message = str(exc) if isinstance(exc, RebuildError) else "Local rebuild failed; original index remains unchanged."
        print(json.dumps({"error_type": type(exc).__name__, "error": message}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
