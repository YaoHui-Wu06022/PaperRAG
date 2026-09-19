from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_rag.library.catalog import Catalog
from paper_rag.library.common import LibraryError, evidence_id
from paper_rag.library.contracts import (
    Filters,
    PapersRequest,
    ReadRequest,
    SearchRequest,
    WikiApply,
    WikiRequest,
)
from paper_rag.library.ingestion import import_document
from paper_rag.library.reading import get_evidence, papers, read
from paper_rag.library.search import search
from paper_rag.library.settings import LibrarySettings
from paper_rag.library.vectors import index_library, rollback
from paper_rag.library.wiki import apply, lint, read_page


@pytest.fixture
def library(tmp_path):
    settings = LibrarySettings(tmp_path, tmp_path / "library", {})
    with Catalog(settings, writable=True) as cat:
        yield cat


def add_note(
    cat,
    name="notes.md",
    text="# Research\n\nResidual shortcuts ease optimization.\n",
    **kwargs,
):
    path = cat.settings.root / name
    path.write_text(text, encoding="utf-8")
    return import_document(cat, path, **kwargs)


def publish(cat):
    return index_library(cat, lexical_only=True)


def first_ref(cat, revision):
    row = cat.one(
        "SELECT b.*,r.document_id FROM blocks b JOIN revisions r USING(revision_id) WHERE revision_id=? ORDER BY ordinal LIMIT 1",
        (revision,),
    )
    return evidence_id(
        row["document_id"], revision, row["block_id"], 0, len(row["text"])
    )


def test_unicode_import_idempotency_rename_and_metadata(library):
    cat = library
    first = add_note(
        cat,
        "研究笔记.md",
        metadata={"title": "残差网络", "aliases": ["ResNet"], "tags": ["深度学习"]},
    )
    publish(cat)
    path = cat.settings.root / "研究笔记.md"
    renamed = path.with_name("renamed.md")
    path.rename(renamed)
    second = import_document(cat, renamed, metadata={"year": {"publish_year": 2026}})
    assert second["document_id"] == first["document_id"]
    assert second["revision_id"] == first["revision_id"]
    assert second["unchanged"]
    result = papers(cat, PapersRequest(query="ResNet"))
    assert result["items"][0]["metadata"]["year"]["publish_year"] == 2026
    assert result["items"][0]["metadata"]["title"] == "残差网络"
    assert (
        papers(cat, PapersRequest(filters=Filters(tags=["深度学习"])), count=True)[
            "count"
        ]
        == 1
    )


def test_source_edits_keep_identity_and_historical_evidence(library):
    first = add_note(library)
    publish(library)
    ref = first_ref(library, first["revision_id"])
    text = get_evidence(library, ref)["text"]
    second = add_note(
        library, text="# Research\n\nA revised conclusion with additional evidence.\n"
    )
    assert first["document_id"] == second["document_id"]
    assert first["revision_id"] != second["revision_id"]
    assert not get_evidence(library, ref)["historical"]
    publish(library)
    assert get_evidence(library, ref)["historical"]
    assert get_evidence(library, ref)["text"] == text


def test_read_pagination_never_loses_unicode_or_children(library):
    source = add_note(
        library, text="# Topic\n\nIntro 文本。\n\n## Child\n\n" + "这是长段落。 " * 300
    )
    publish(library)
    request = ReadRequest(
        document_id=source["document_id"], section_id="s_0001", max_chars=83
    )
    full = read(library, request.model_copy(update={"max_chars": 64000}))
    expected = "".join(v["text"] for v in full["fragments"])
    actual = ""
    for _ in range(100):
        result = read(library, request)
        actual += "".join(v["text"] for v in result["fragments"])
        if not result["next_cursor"]:
            break
        request = request.model_copy(update={"cursor": result["next_cursor"]})
    assert actual == expected
    assert "Child" in actual
    assert "长段落" in actual


def test_local_block_ids_cannot_cross_documents(library):
    first = add_note(library, "a.md", "# Same\n\nDocument A.\n")
    second = add_note(library, "b.md", "# Same\n\nDocument B.\n")
    publish(library)
    a = first_ref(library, first["revision_id"])
    b = first_ref(library, second["revision_id"])
    assert a != b
    with pytest.raises(LibraryError):
        get_evidence(library, a.replace(first["document_id"], second["document_id"]))


def test_fts_filters_and_graceful_dense_degradation(library):
    source = add_note(
        library, metadata={"title": "Residual network", "year": {"publish_year": 2016}}
    )
    add_note(library, "other.md", "# Transformer\n\nSelf attention in sequence models.")
    publish(library)
    result = search(library, SearchRequest(query="residual", mode="hybrid"))
    assert result["items"][0]["document_id"] == source["document_id"]
    assert result["degraded"] and result["warnings"]
    assert (
        search(
            library,
            SearchRequest(
                query="residual", mode="lexical", filters=Filters(document_ids=[])
            ),
        )["items"]
        == []
    )
    assert (
        search(
            library,
            SearchRequest(
                query="residual", mode="lexical", filters=Filters(year_min=2020)
            ),
        )["items"]
        == []
    )


def test_ambiguous_title_returns_candidates_and_exact_count(library):
    add_note(library, "a.md", "One.", metadata={"title": "Same title"})
    add_note(library, "b.md", "Two.", metadata={"title": "Same title"})
    publish(library)
    result = papers(library, PapersRequest(query="Same title", limit=1))
    assert result["ambiguous"] and result["total"] == 2 and result["next_offset"] == 1
    assert (
        papers(library, PapersRequest(query="Same title", limit=1), count=True)["count"]
        == 2
    )


class FakeStore:
    def __init__(self):
        self.collections = {}
        self.fail = False

    def create(self, name):
        self.collections[name] = {}

    def upsert(self, name, chunks, vectors):
        if self.fail:
            raise RuntimeError("injected vector write failure")
        self.collections[name].update({c["chunk_id"]: c for c in chunks})

    def ready(self, name):
        pass

    def verify(self, name, chunks):
        assert all(c["chunk_id"] in self.collections[name] for c in chunks)

    def search(self, name, vector, limit, regions, document_ids):
        return [c | {"score": 1.0} for c in self.collections[name].values()][:limit]


class FakeEmbedder:
    def __init__(self):
        self.count = 0

    def embed(self, texts):
        self.count += len(texts)
        return [[1.0] * 1024 for _ in texts]


def test_index_failure_preserves_publication_and_retry_is_incremental(library):
    source = add_note(library)
    store, embedder = FakeStore(), FakeEmbedder()
    first = index_library(library, store=store, embedder=embedder)
    count = embedder.count
    index_library(library, store=store, embedder=embedder)
    assert embedder.count == count
    second = add_note(library, text="# New version\n\nChanged text.")
    store.fail = True
    with pytest.raises(RuntimeError):
        index_library(library, store=store, embedder=embedder)
    assert (
        library.one("SELECT active_revision FROM documents")["active_revision"]
        == source["revision_id"]
    )
    store.fail = False
    index_library(library, store=store, embedder=embedder)
    assert (
        library.one("SELECT active_revision FROM documents")["active_revision"]
        == second["revision_id"]
    )
    assert library.get("active_generation") == first["generation_id"]


def test_dense_rejects_same_id_wrong_hash(library):
    add_note(library)
    store, embedder = FakeStore(), FakeEmbedder()
    index_library(library, store=store, embedder=embedder)
    collection = next(iter(store.collections.values()))
    for chunk in collection.values():
        chunk["content_hash"] = "wrong"
    result = search(
        library,
        SearchRequest(query="residual", mode="dense"),
        store=store,
        embedder=embedder,
    )
    assert result["items"] == []


def test_full_rebuild_keeps_old_generation_for_rollback(library):
    source = add_note(library)
    store, embedder = FakeStore(), FakeEmbedder()
    first = index_library(library, store=store, embedder=embedder)
    add_note(library, text="# Updated\n\nNew paper content.")
    second = index_library(library, rebuild=True, store=store, embedder=embedder)
    assert first["generation_id"] != second["generation_id"]
    rollback(library, first["generation_id"])
    assert (
        library.one("SELECT active_revision FROM documents")["active_revision"]
        == source["revision_id"]
    )
    assert len(store.collections) == 2


def test_wiki_citations_manual_edits_and_staleness(library):
    source = add_note(library)
    publish(library)
    ref = first_ref(library, source["revision_id"])
    request = WikiApply(
        page_id="residual",
        kind="concept",
        title="Residual learning",
        expected_revision=0,
        body=f"Source note: [evidence]({ref}).",
        evidence_ids=[ref],
    )
    result = apply(library, request)
    assert result["revision"] == 1
    assert not lint(library)["issues"]
    path = Path(result["path"])
    path.write_text(
        path.read_text(encoding="utf-8") + "\nHuman edit.\n", encoding="utf-8"
    )
    with pytest.raises(LibraryError, match="Markdown changed"):
        apply(library, request.model_copy(update={"expected_revision": 1}))
    assert "Human edit" in read_page(library, WikiRequest(page_id="residual"))["body"]
    add_note(library, text="# Changed\n\nUpdated finding.")
    publish(library)
    assert read_page(library, WikiRequest(page_id="residual"))["stale"]
    with pytest.raises(LibraryError, match="active"):
        apply(library, request.model_copy(update={"expected_revision": 1}))


def test_wiki_checks_quote_and_does_not_promote_generated_sources(library):
    source = add_note(library)
    publish(library)
    ref = first_ref(library, source["revision_id"])
    request = WikiApply(
        page_id="test",
        kind="topic",
        title="Test",
        expected_revision=0,
        body=f"[Evidence]({ref})",
        evidence_ids=[ref],
        quotes={ref: "Invented verbatim quote"},
    )
    with pytest.raises(LibraryError, match="absent"):
        apply(library, request)
    result = apply(library, request.model_copy(update={"quotes": {}}))
    with pytest.raises(LibraryError, match="primary sources"):
        import_document(library, Path(result["path"]))
