import json
from pathlib import Path

import pytest

from paper_rag.corpus.chunks import ChunkDocument
from paper_rag.retrieval.dense.cache import CachedEmbedder, EmbeddingCache, embedding_cache_key
from paper_rag.retrieval.dense.milvus_store import MilvusStore
from paper_rag.retrieval.sparse.bm25 import BM25CorpusIndex


def test_bm25_uses_scope_specific_document_frequency():
    index = BM25CorpusIndex([
        bm25_doc("c1", "rare residual connection"),
        bm25_doc("c2", "common transformer attention"),
        bm25_doc("c3", "rare appendix detail"),
    ])

    all_stats = index.scope_stats(None)
    scoped_stats = index.scope_stats(["c1", "c2"])

    assert all_stats.doc_freqs["rare"] == 2
    assert scoped_stats.doc_freqs["rare"] == 1
    assert [hit.doc_id for hit in index.search("rare", 5, allowed_chunk_ids=["c1", "c2"])] == ["c1"]
    assert index.search("rare", 5, allowed_chunk_ids=["c2"]) == []


def test_bm25_derived_index_roundtrip_and_staleness(tmp_path: Path):
    chunks = [chunk("p::chunk_0000", "p", "residual connection")]
    index_path = tmp_path / "bm25_chunks.json"
    BM25CorpusIndex.from_chunks(chunks).save(index_path)

    loaded = BM25CorpusIndex.load(index_path, chunks)
    assert loaded is not None
    assert loaded.search("residual", 1)[0].doc_id == "p::chunk_0000"

    changed_chunks = [chunk("p::chunk_0000", "p", "different text")]
    assert BM25CorpusIndex.load(index_path, changed_chunks) is None


def test_bm25_save_failure_preserves_existing_index(tmp_path: Path, monkeypatch):
    chunks = [chunk("p::chunk_0000", "p", "residual connection")]
    index_path = tmp_path / "bm25_chunks.json"
    index_path.write_text("old index", encoding="utf-8")
    original_replace = Path.replace

    def fail_replace(path: Path, target: Path) -> Path:
        if path.name.startswith("bm25_chunks.json.tmp-"):
            raise OSError("replace failed")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        BM25CorpusIndex.from_chunks(chunks).save(index_path)

    assert index_path.read_text(encoding="utf-8") == "old index"
    assert not list(tmp_path.glob("bm25_chunks.json.tmp-*"))


def test_milvus_rebuild_uses_staging_and_preserves_old_on_insert_failure():
    client = FakeMilvusClient(collections={"paper_chunks"}, fail_insert=True)
    store = MilvusStore(uri="", token=None, db_name=None, collection_name="paper_chunks", dimensions=1, client=client)

    with pytest.raises(RuntimeError, match="insert failed"):
        store.rebuild_collection([chunk("p::chunk_0000", "p", "residual connection")], [[1.0]])

    assert "paper_chunks" in client.collections
    assert client.aliases == {}
    assert ("drop", "paper_chunks") not in client.ops


def test_milvus_rebuild_switches_alias_after_successful_staging_insert():
    client = FakeMilvusClient(collections={"paper_chunks"})
    store = MilvusStore(uri="", token=None, db_name=None, collection_name="paper_chunks", dimensions=1, client=client)

    inserted = store.rebuild_collection([chunk("p::chunk_0000", "p", "residual connection")], [[1.0]])

    assert inserted == 1
    assert "paper_chunks" not in client.collections
    assert client.aliases["paper_chunks"].startswith("paper_chunks__staging_")
    create_index = next(i for i, op in enumerate(client.ops) if op[0] == "create" and op[1].startswith("paper_chunks__staging_"))
    insert_index = next(i for i, op in enumerate(client.ops) if op[0] == "insert")
    drop_index = client.ops.index(("drop", "paper_chunks"))
    alias_index = next(i for i, op in enumerate(client.ops) if op[0] == "create_alias")
    assert create_index < insert_index < drop_index < alias_index


def test_embedding_cache_appends_and_cached_embedder_only_requests_misses(tmp_path: Path):
    cache_path = tmp_path / "embedding_cache.jsonl"
    cache = EmbeddingCache(cache_path)
    cached_key = embedding_cache_key("model-a", 2, "cached")
    cache.set(cached_key, "cached", [1.0, 1.0])
    cache.save()

    client = FakeEmbeddingClient({"missing": [2.0, 2.0]})
    embedder = CachedEmbedder(
        client,
        EmbeddingCache(cache_path),
        model="model-a",
        dimensions=2,
        batch_size=4,
    )

    vectors = embedder.embed_texts(["cached", "missing"])

    assert vectors == [[1.0, 1.0], [2.0, 2.0]]
    assert client.calls == [["missing"]]

    cache = EmbeddingCache(cache_path)
    cache.set(cached_key, "cached", [3.0, 3.0])
    cache.save()
    assert EmbeddingCache(cache_path).get(cached_key) == [3.0, 3.0]


def test_query_embedding_cache_writes_text_and_rejects_old_rows(tmp_path: Path):
    cache_path = tmp_path / "query_embedding_cache.jsonl"
    client = FakeEmbeddingClient({"查找模型结构": [1.0, 2.0]})
    embedder = CachedEmbedder(
        client,
        EmbeddingCache(cache_path, store_text=True),
        model="model-a",
        dimensions=2,
        batch_size=4,
    )

    embedder.embed_texts(["查找模型结构"])

    row = json.loads(cache_path.read_text(encoding="utf-8"))
    assert row["text"] == "查找模型结构"

    cache_path.write_text(
        json.dumps({"key": row["key"], "vector": row["vector"]}) + "\n",
        encoding="utf-8",
    )
    try:
        EmbeddingCache(cache_path, store_text=True)
    except ValueError as exc:
        assert "缺少 text" in str(exc)
    else:
        raise AssertionError("旧 query embedding 缓存格式不应继续兼容")


class FakeEmbeddingClient:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self.vectors[text] for text in texts]


class FakeMilvusClient:
    def __init__(self, *, collections: set[str] | None = None, fail_insert: bool = False) -> None:
        self.collections = set(collections or set())
        self.aliases: dict[str, str] = {}
        self.fail_insert = fail_insert
        self.ops: list[tuple[str, str]] = []

    def has_collection(self, collection_name: str) -> bool:
        return collection_name in self.collections or collection_name in self.aliases

    def create_collection(self, collection_name: str, **_kwargs) -> None:
        self.collections.add(collection_name)
        self.ops.append(("create", collection_name))

    def insert(self, collection_name: str, rows: list[dict]) -> None:
        self.ops.append(("insert", collection_name))
        if self.fail_insert:
            raise RuntimeError("insert failed")

    def load_collection(self, collection_name: str) -> None:
        self.ops.append(("load", collection_name))

    def drop_collection(self, collection_name: str) -> None:
        self.collections.discard(collection_name)
        self.ops.append(("drop", collection_name))

    def describe_alias(self, alias: str) -> dict:
        if alias not in self.aliases:
            raise RuntimeError("alias not found")
        return {"collection_name": self.aliases[alias]}

    def create_alias(self, *, collection_name: str, alias: str) -> None:
        self.aliases[alias] = collection_name
        self.ops.append(("create_alias", alias))

    def alter_alias(self, *, collection_name: str, alias: str) -> None:
        self.aliases[alias] = collection_name
        self.ops.append(("alter_alias", alias))


def bm25_doc(doc_id: str, text: str):
    from paper_rag.retrieval.sparse.bm25 import BM25Document

    return BM25Document(doc_id=doc_id, text=text, payload={})


def chunk(chunk_id: str, paper_id: str, text: str) -> ChunkDocument:
    return ChunkDocument(
        chunk_id=chunk_id,
        paper_id=paper_id,
        chunk_index=0,
        region="body",
        section_id="sec_intro",
        title="Paper",
        section_path=["Introduction"],
        pages=[1],
        block_ids=["b000001"],
        text=text,
        embedding_text=f"Paper: Paper\nSection: Introduction\n\n{text}",
    )
