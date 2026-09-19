from __future__ import annotations

import array
import json
import math
import uuid

from .catalog import Catalog
from .common import LibraryError, digest, dumps, now
from .ingestion import (
    publish_revision,
    resolve_citations,
    invalidate_wiki,
    remove_fts_document,
)
from .settings import LibrarySettings
from .publications import seal, rollback_publication
from .locking import single_writer


class Embedder:
    def __init__(self, cat: Catalog, client=None):
        from .embedding import EmbeddingClient

        self.cat = cat
        self.identity = cat.settings.embedding_identity
        self.client = client or EmbeddingClient(
            self.identity["endpoint"],
            cat.settings.env.get("EMBEDDING_API_KEY"),
            self.identity["model"],
            self.identity["dimensions"],
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        result, missing = {}, {}
        for text in texts:
            key = digest([self.identity, text])
            row = self.cat.one(
                "SELECT vector FROM embedding_cache WHERE cache_key=?", (key,)
            )
            if row:
                data = array.array("f")
                data.frombytes(row["vector"])
                result[key] = list(data)
            else:
                missing[key] = text
        batch_size = max(
            1, int(self.cat.settings.env.get("EMBEDDING_BATCH_SIZE", "10"))
        )
        items = list(missing.items())
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            vectors = self.client.embed_texts([v for _, v in batch])
            if len(vectors) != len(batch):
                raise LibraryError(
                    "embedding_invalid", "Embedding response count mismatch"
                )
            for (key, _), vector in zip(batch, vectors, strict=True):
                if len(vector) != self.identity["dimensions"] or not all(
                    math.isfinite(v) for v in vector
                ):
                    raise LibraryError(
                        "embedding_invalid", "Embedding dimension or value mismatch"
                    )
                result[key] = vector
                if self.cat.writable:
                    self.cat.db.execute(
                        "INSERT OR IGNORE INTO embedding_cache VALUES(?,?,?,?)",
                        (
                            key,
                            dumps(self.identity),
                            array.array("f", vector).tobytes(),
                            now(),
                        ),
                    )
            if self.cat.writable:
                self.cat.db.commit()
        return [result[digest([self.identity, text])] for text in texts]


class VectorStore:
    def __init__(self, settings: LibrarySettings):
        from pymilvus import MilvusClient

        if not settings.env.get("MILVUS_URI"):
            raise LibraryError("dependency_unavailable", "MILVUS_URI is not configured")
        self.client = MilvusClient(
            uri=settings.env["MILVUS_URI"],
            token=settings.env.get("MILVUS_TOKEN", ""),
            db_name=settings.env.get("MILVUS_DB_NAME", ""),
            timeout=20,
        )
        self.dimensions = settings.embedding_identity["dimensions"]

    def close(self):
        self.client.close()

    def create(self, name: str):
        from pymilvus import DataType

        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=128)
        for field, size in (
            ("document_id", 80),
            ("revision_id", 80),
            ("content_hash", 64),
            ("region", 16),
        ):
            schema.add_field(field, DataType.VARCHAR, max_length=size)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self.dimensions)
        indexes = self.client.prepare_index_params()
        indexes.add_index(
            field_name="vector", index_type="AUTOINDEX", metric_type="COSINE"
        )
        self.client.create_collection(
            collection_name=name,
            schema=schema,
            index_params=indexes,
            consistency_level="Strong",
        )

    def upsert(self, name: str, chunks: list[dict], vectors: list[list[float]]):
        rows = [
            {
                **{
                    k: chunk[k]
                    for k in (
                        "chunk_id",
                        "document_id",
                        "revision_id",
                        "content_hash",
                        "region",
                    )
                },
                "vector": v,
            }
            for chunk, v in zip(chunks, vectors, strict=True)
        ]
        self.client.upsert(collection_name=name, data=rows)

    def ready(self, name: str):
        self.client.flush(collection_name=name)
        self.client.load_collection(collection_name=name)

    def verify(self, name: str, chunks: list[dict]):
        rows = self.client.get(
            collection_name=name,
            ids=[c["chunk_id"] for c in chunks],
            output_fields=["chunk_id", "content_hash", "revision_id"],
            consistency_level="Strong",
        )
        expected = {
            (c["chunk_id"], c["content_hash"], c["revision_id"]) for c in chunks
        }
        actual = {(c["chunk_id"], c["content_hash"], c["revision_id"]) for c in rows}
        if expected != actual:
            raise LibraryError(
                "index_incomplete", "Milvus rows failed read-after-write verification"
            )

    def search(
        self,
        name: str,
        vector: list[float],
        limit: int,
        regions: list[str],
        document_ids: list[str] | None,
    ):
        expression = "region in " + json.dumps(regions)
        if document_ids is not None:
            expression += " and document_id in " + json.dumps(document_ids)
        rows = self.client.search(
            collection_name=name,
            data=[vector],
            limit=limit,
            filter=expression,
            output_fields=["chunk_id", "document_id", "revision_id", "content_hash"],
            search_params={"metric_type": "COSINE"},
            consistency_level="Strong",
        )
        return [
            {**r["entity"], "score": float(r["distance"])}
            for r in (rows[0] if rows else [])
        ]


def index_library(
    cat: Catalog,
    *,
    rebuild: bool = False,
    lexical_only: bool = False,
    store=None,
    embedder=None,
    reporter=lambda _: None,
):
    targets = cat.rows(
        "SELECT d.document_id,COALESCE((SELECT r.revision_id FROM revisions r WHERE r.document_id=d.document_id AND r.status='staged' ORDER BY r.created_at DESC LIMIT 1),d.active_revision) AS revision_id FROM documents d WHERE d.withdrawn=0"
    )
    targets = [r for r in targets if r["revision_id"]]
    if not targets:
        return {"documents": 0, "chunks_written": 0, "mode": "empty"}
    if lexical_only:
        with cat.transaction():
            for target in targets:
                active = cat.one(
                    "SELECT active_revision FROM documents WHERE document_id=?",
                    (target["document_id"],),
                )["active_revision"]
                if active != target["revision_id"]:
                    publish_revision(cat, target["revision_id"])
            resolve_citations(cat)
        return {
            "documents": len(targets),
            "chunks_written": 0,
            "mode": "lexical_only",
            "warning": "Dense coverage requires a subsequent index run",
        }
    identity = dumps(cat.settings.embedding_identity)
    active_id = cat.get("active_generation")
    active = (
        cat.one("SELECT * FROM index_generations WHERE generation_id=?", (active_id,))
        if active_id
        else None
    )
    if active and active["identity"] != identity and not rebuild:
        raise LibraryError(
            "index_identity_mismatch",
            "Embedding configuration changed; rebuild into a new generation",
        )
    own_store = store is None
    store = store or VectorStore(cat.settings)
    generation = active if active and not rebuild else None
    if generation is None:
        generation = cat.one(
            "SELECT * FROM index_generations WHERE state='building' AND identity=? ORDER BY created_at DESC LIMIT 1",
            (identity,),
        )
    written = 0
    try:
        if generation is None:
            gid = "g_" + uuid.uuid4().hex
            name = cat.settings.collection_prefix + "_" + gid
            occupied = cat.one("SELECT count(*) n FROM collection_registry WHERE status!='retired'")["n"]
            if occupied >= 5:
                raise LibraryError("collection_capacity", "Milvus collection limit reached; retire an archived collection first")
            with cat.transaction():
                cat.db.execute("INSERT OR REPLACE INTO collection_registry(collection_name,role,status) VALUES(?,?,?)", (name, "generation", "building"))
                cat.log("collection_create_intent", {"collection": name, "generation_id": gid})
            store.create(name)
            with cat.transaction():
                cat.db.execute(
                    "INSERT INTO index_generations VALUES(?,?,?,?,?)",
                    (gid, name, identity, "building", now()),
                )
            generation = {"generation_id": gid, "collection_name": name}
        gid, name = generation["generation_id"], generation["collection_name"]
        embedder = embedder or Embedder(cat)
        for target in targets:
            last = ""
            while True:
                rows = cat.rows(
                    "SELECT c.* FROM chunks c WHERE c.revision_id=? AND c.chunk_id>? AND NOT EXISTS(SELECT 1 FROM index_rows i WHERE i.generation_id=? AND i.chunk_id=c.chunk_id AND i.content_hash=c.content_hash) ORDER BY c.chunk_id LIMIT 100",
                    (target["revision_id"], last, gid),
                )
                if not rows:
                    break
                reporter(f"index {target['document_id']} +{len(rows)} chunks")
                vectors = embedder.embed([r["embedding_text"] for r in rows])
                store.upsert(name, rows, vectors)
                store.ready(name)
                store.verify(name, rows)
                with cat.transaction():
                    cat.db.executemany(
                        "INSERT OR REPLACE INTO index_rows VALUES(?,?,?)",
                        [(gid, r["chunk_id"], r["content_hash"]) for r in rows],
                    )
                written += len(rows)
                last = rows[-1]["chunk_id"]
        with cat.transaction():
            for target in targets:
                if (
                    cat.one(
                        "SELECT active_revision FROM documents WHERE document_id=?",
                        (target["document_id"],),
                    )["active_revision"]
                    != target["revision_id"]
                ):
                    publish_revision(cat, target["revision_id"])
                cat.db.execute(
                    "INSERT OR REPLACE INTO generation_revisions VALUES(?,?,?)",
                    (gid, target["document_id"], target["revision_id"]),
                )
            resolve_citations(cat)
            cat.db.execute(
                "UPDATE index_generations SET state='ready' WHERE generation_id=?",
                (gid,),
            )
            cat.db.execute("UPDATE collection_registry SET status='online',role='generation' WHERE collection_name=?", (name,))
            publication_id = seal(cat, gid)
            cat.log("index_publish", {"generation_id": gid, "chunks_written": written})
        return {
            "generation_id": gid,
            "collection": name,
            "documents": len(targets),
            "chunks_written": written,
            "mode": "hybrid",
            "publication_id": publication_id,
        }
    except BaseException:
        if generation:
            with cat.transaction():
                cat.job(
                    "build_" + generation["generation_id"],
                    "index_build",
                    None,
                    None,
                    "failed",
                    {"resumable": True},
                )
        raise
    finally:
        if own_store:
            store.close()


def rollback(cat: Catalog, generation_id: str):
    publication = cat.one("SELECT publication_id FROM publications WHERE generation_id=? ORDER BY created_at DESC LIMIT 1", (generation_id,))
    if not publication:
        raise LibraryError("not_found", "Ready index generation not found")
    return rollback_publication(cat, publication["publication_id"])
