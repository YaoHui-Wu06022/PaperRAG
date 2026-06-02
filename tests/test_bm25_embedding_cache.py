from pathlib import Path

from paper_rag.corpus.chunks import ChunkDocument
from paper_rag.retrieval.dense.cache import CachedEmbedder, EmbeddingCache, embedding_cache_key
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


def test_embedding_cache_appends_and_cached_embedder_only_requests_misses(tmp_path: Path):
    cache_path = tmp_path / "embedding_cache.jsonl"
    cache = EmbeddingCache(cache_path)
    cached_key = embedding_cache_key("model-a", 2, "cached")
    cache.set(cached_key, [1.0, 1.0])
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
    cache.set(cached_key, [3.0, 3.0])
    cache.save()
    assert EmbeddingCache(cache_path).get(cached_key) == [3.0, 3.0]


class FakeEmbeddingClient:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self.vectors[text] for text in texts]


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
