import json
from pathlib import Path

import pytest

from test_library_core import (
    library,
    add_note,
    publish,
    first_ref,
    FakeStore,
    FakeEmbedder,
)
from paper_rag.library.common import LibraryError, atomic_text, digest, dumps
from paper_rag.library.contracts import (
    ReadRequest,
    SearchRequest,
    WikiApply,
    WikiRequest,
    EvalRequest,
)
from paper_rag.library.reading import read, get_evidence
from paper_rag.library.search import search
from paper_rag.library.vectors import Embedder, index_library
from paper_rag.library.settings import LibrarySettings
from paper_rag.library.wiki import apply, recover, page_path, read_page
from paper_rag.library.evaluation import evaluate


def test_schema_two_migration_preserves_fts_and_evidence(tmp_path):
    from paper_rag.library.catalog import Catalog

    settings = LibrarySettings(tmp_path, tmp_path / "catalog", {})
    with Catalog(settings, writable=True) as cat:
        doc = add_note(cat)
        publish(cat)
        ref = first_ref(cat, doc["revision_id"])
        original = get_evidence(cat, ref)["text"]
        cat.db.execute("DROP TABLE chunk_fts_rows")
        cat.db.execute("PRAGMA user_version=2")
        cat.db.commit()
    with Catalog(settings, writable=True) as cat:
        assert get_evidence(cat, ref)["text"] == original
        assert (
            cat.one("SELECT count(*) n FROM chunk_fts_rows")["n"]
            == cat.one("SELECT count(*) n FROM chunk_fts")["n"]
        )
        add_note(cat, text="# Updated\n\nNew evidence after schema migration.")
        publish(cat)
        assert search(cat, SearchRequest(query="Updated", mode="lexical"))["items"]
        assert not search(cat, SearchRequest(query="Residual", mode="lexical"))["items"]


def test_embedding_cache_includes_endpoint_model_and_dimension(library):
    class Client:
        calls = 0

        def embed_texts(self, texts):
            self.calls += 1
            return [[0.1] * 1024 for _ in texts]

    client = Client()
    first = Embedder(library, client)
    first.embed(["identical", "identical"])
    first.embed(["identical"])
    assert client.calls == 1
    settings = library.settings
    library.settings = LibrarySettings(
        settings.root,
        settings.home,
        {"EMBEDDING_BASE_URL": "https://different.invalid"},
    )
    Embedder(library, client).embed(["identical"])
    assert client.calls == 2


def test_invalid_embedding_never_enters_cache(library):
    class Client:
        def embed_texts(self, texts):
            return [[float("nan")] * 1024]

    with pytest.raises(LibraryError):
        Embedder(library, Client()).embed(["bad"])
    assert library.one("SELECT count(*) n FROM embedding_cache")["n"] == 0


def test_failed_initial_generation_resumes_same_collection(library):
    add_note(library)
    store = FakeStore()
    store.fail = True
    with pytest.raises(RuntimeError):
        index_library(library, store=store, embedder=FakeEmbedder())
    assert library.get("active_generation") is None
    store.fail = False
    index_library(library, store=store, embedder=FakeEmbedder())
    assert len(store.collections) == 1


def test_metadata_change_updates_fts_without_replacing_evidence(library):
    first = add_note(library, metadata={"title": "Original"})
    publish(library)
    second = add_note(library, metadata={"title": "UniqueRetitledEvidence"})
    assert first["revision_id"] == second["revision_id"]
    assert search(
        library, SearchRequest(query="UniqueRetitledEvidence", mode="lexical")
    )["items"]


def test_read_context_is_paginated_and_markdown_lines_are_exact(library):
    doc = add_note(library, text="# Topic\n\nfirst\nline two\n\nlast paragraph\n")
    publish(library)
    rows = library.rows("SELECT * FROM blocks ORDER BY ordinal")
    block = next(b for b in rows if "first" in b["text"])
    from paper_rag.library.common import evidence_id

    offset = block["text"].index("line two")
    ref = evidence_id(
        doc["document_id"], doc["revision_id"], block["block_id"], offset, offset + 8
    )
    assert get_evidence(library, ref)["line_start"] == 4
    request = ReadRequest(
        evidence_id=ref, context_before=1, context_after=1, max_chars=5
    )
    text = ""
    while True:
        result = read(library, request)
        text += "".join(f["text"] for f in result["fragments"])
        if not result["next_cursor"]:
            break
        request = request.model_copy(update={"cursor": result["next_cursor"]})
    assert "# Topic" in text and "last paragraph" in text


def test_wiki_recovery_restores_previous_publication(library):
    source = add_note(library)
    publish(library)
    ref = first_ref(library, source["revision_id"])
    request = WikiApply(
        page_id="page",
        kind="paper",
        title="Page",
        expected_revision=0,
        body=f"[source]({ref})",
        evidence_ids=[ref],
    )
    apply(library, request)
    path = page_path(library, "page")
    before = path.read_text(encoding="utf-8")
    after = "interrupted update"
    atomic_text(path, after)
    intent = library.settings.home / "wiki/pending/page.json"
    atomic_text(
        intent,
        dumps(
            {
                "page_id": "page",
                "revision": 2,
                "before": before,
                "after_hash": digest(after),
            }
        ),
    )
    assert recover(library, WikiRequest())["pages"][0]["action"] == "restored_previous"
    assert path.read_text(encoding="utf-8") == before
    assert read_page(library, WikiRequest(page_id="page"))["revision"] == 1


def test_wiki_recovery_preserves_user_edits(library):
    path = page_path(library, "new")
    atomic_text(path, "user edit")
    atomic_text(
        library.settings.home / "wiki/pending/new.json",
        dumps(
            {
                "page_id": "new",
                "revision": 1,
                "before": None,
                "after_hash": digest("generated"),
            }
        ),
    )
    with pytest.raises(LibraryError, match="preserve edits"):
        recover(library, WikiRequest())
    assert path.read_text(encoding="utf-8") == "user edit"


def test_evaluation_rejects_shared_source_split_leakage(library, tmp_path):
    source = add_note(library)
    publish(library)
    ref = first_ref(library, source["revision_id"])
    case = {
        "query": "residual",
        "reviewed": True,
        "evidence_ids": [ref],
        "group": "same-source",
    }
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            {
                "corpus_version": library.snapshot(),
                "cases": [
                    case | {"id": "a", "split": "tune"},
                    case | {"id": "b", "split": "test"},
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(LibraryError, match="cross"):
        evaluate(library, EvalRequest(cases_path=str(path)))


def test_evaluation_cannot_credit_matching_title_in_wrong_evidence(
    library, tmp_path, monkeypatch
):
    source = add_note(
        library, "one.md", "Gold evidence.", metadata={"title": "Same title"}
    )
    other = add_note(
        library, "two.md", "Different evidence.", metadata={"title": "Same title"}
    )
    publish(library)
    gold = first_ref(library, source["revision_id"])
    wrong = first_ref(library, other["revision_id"])
    monkeypatch.setattr(
        "paper_rag.library.evaluation.search",
        lambda *a, **k: {
            "items": [{"evidence_ids": [wrong], "title": "Same title"}],
            "next_cursor": None,
            "degraded": False,
            "warnings": [],
            "timings": {"total_ms": 1},
        },
    )
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            {
                "corpus_version": library.snapshot(),
                "cases": [
                    {
                        "id": "one",
                        "query": "Same title",
                        "reviewed": True,
                        "evidence_ids": [gold],
                        "group": "g",
                        "split": "test",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert (
        evaluate(library, EvalRequest(cases_path=str(path)))["summary"]["test"]["hit"]
        == 0
    )
