from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from paper_rag.config import Settings
from paper_rag.lexical import build_fts_query, normalize_lexical_text
from paper_rag.llamaindex import retrievers
from paper_rag.llamaindex import service
from paper_rag.llamaindex import translation
from paper_rag.llamaindex.query_rewriter import QueryRewrite, QueryRewriterError
from paper_rag.llamaindex.translation import _format_failure, prepare_lexical_query


class FakeTranslator:
    def __init__(self, provider: str, result: str | None = None, error: Exception | None = None) -> None:
        self.provider = provider
        self.result = result
        self.error = error

    def translate(self, text: str) -> str:
        if self.error:
            raise self.error
        return self.result or text


def test_english_query_filters_function_words(tmp_path: Path):
    settings = Settings.load(tmp_path)
    result = prepare_lexical_query("What is a BM25 query for RAG?", settings)

    assert result.query == "bm25 query rag"
    assert set(result.stopwords_removed) == {"what", "is", "a", "for"}
    assert result.translation_used is False


def test_chinese_query_translates_and_preserves_technical_tokens(tmp_path: Path, monkeypatch):
    settings = Settings.load(tmp_path)
    monkeypatch.setattr(
        translation,
        "_make_translator",
        lambda current: FakeTranslator("tencent", result="what is retrieval"),
    )

    result = prepare_lexical_query("什么是 BM25 和 Qwen3", settings)

    assert result.query == "retrieval bm25 qwen3"
    assert result.translation_provider == "tencent"


def test_translation_failure_uses_original_lexical_query(tmp_path: Path, monkeypatch):
    settings = replace(Settings.load(tmp_path), bm25_translation_retry_count=0)
    monkeypatch.setattr(
        "paper_rag.llamaindex.translation._make_translator",
        lambda current: FakeTranslator("tencent", error=RuntimeError("unavailable")),
    )

    result = prepare_lexical_query("什么是 BM25", settings)

    assert result.translation_used is False
    assert result.translation_fallback is True
    assert "bm25" in result.query
    assert len(result.warnings) == 1


def test_translation_skips_overlong_query(tmp_path: Path, monkeypatch):
    settings = replace(Settings.load(tmp_path), bm25_translation_max_chars=3)
    called = False

    def unexpected_translator(current):
        nonlocal called
        called = True
        raise AssertionError("translation should be skipped")

    monkeypatch.setattr("paper_rag.llamaindex.translation._make_translator", unexpected_translator)
    result = prepare_lexical_query("中文问题", settings)

    assert called is False
    assert result.translation_used is False
    assert result.translation_fallback is False
    assert "translation_skipped:query_too_long" in result.warnings


def test_fts_query_filters_stopwords_without_changing_index(tmp_path: Path):
    assert build_fts_query("what is a retrieval of RAG", remove_stopwords=True) == '"retrieval" OR "rag"'


def test_metadata_style_query_removes_generic_words_and_splits_hyphens():
    query, removed = normalize_lexical_text("please find attention-related papers in the library")

    assert query == "attention"
    assert set(removed) == {"please", "find", "related", "papers", "in", "the", "library"}


def test_semantically_relevant_negation_and_comparison_words_are_kept():
    query, removed = normalize_lexical_text("not no without versus vs less more what")

    assert query == "not no without versus vs less more"
    assert removed == ["what"]


def test_translated_query_is_sent_to_lexical_retriever(tmp_path: Path, monkeypatch):
    settings = Settings.load(tmp_path)
    monkeypatch.setattr(
        "paper_rag.llamaindex.translation._make_translator",
        lambda current: FakeTranslator("tencent", result="what is retrieval"),
    )
    seen: list[str] = []
    monkeypatch.setattr(
        retrievers,
        "search_chunks",
        lambda settings, query, paper_ids, limit, regions, **kwargs: seen.append(query) or [],
    )

    retrievers.SQLiteLexicalRetriever(settings).retrieve("什么是检索")

    assert seen == ["retrieval"]


def test_metadata_search_uses_translated_free_text_query(tmp_path: Path, monkeypatch):
    settings = Settings.load(tmp_path)
    monkeypatch.setattr(
        "paper_rag.llamaindex.translation._make_translator",
        lambda current: FakeTranslator("tencent", result="attention mechanism"),
    )
    seen: list[str] = []
    monkeypatch.setattr(
        service,
        "search_catalog",
        lambda settings, query, filters, limit, **kwargs: seen.append(query) or [],
    )

    result = service.search(settings, "什么是注意力机制", limit=5)

    assert result["status"] == "ok"
    assert seen == ["attention mechanism"]
    assert result["data"]["query_debug"]["translation_used"] is True
    assert result["data"]["query_debug"]["translation_provider"] == "tencent"


def test_hybrid_keeps_original_query_for_semantic_retriever(tmp_path: Path, monkeypatch):
    settings = Settings.load(tmp_path)
    monkeypatch.setattr(
        retrievers,
        "search_chunks",
        lambda settings, query, paper_ids, limit, regions, **kwargs: [],
    )
    monkeypatch.setattr(
        "paper_rag.llamaindex.translation._make_translator",
        lambda current: FakeTranslator("tencent", result="what is hybrid retrieval"),
    )
    seen: list[str] = []

    class FakeSemanticRetriever:
        def retrieve(self, query):
            seen.append(query.query_str)
            return []

    hybrid = retrievers.HybridRetriever(settings, FakeSemanticRetriever(), mode="hybrid")
    assert hybrid.retrieve("什么是混合检索") == []
    assert seen == ["什么是混合检索"]
    assert hybrid.lexical_debug["translation_used"] is True


class FakeRewriter:
    def __init__(self, result: QueryRewrite | None = None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error

    def rewrite(self, _query: str) -> QueryRewrite:
        if self.error:
            raise self.error
        assert self.result is not None
        return self.result


def test_query_rewriter_keeps_core_term_and_phrase(tmp_path: Path, monkeypatch):
    settings = replace(Settings.load(tmp_path), query_rewriter_enabled=True, query_rewriter_api_key="test-key")

    def translate(_self, text: str) -> str:
        return {"注意力": "attention", "注意力机制": "attention mechanism"}[text]

    monkeypatch.setattr(translation.TencentTranslator, "translate", translate)
    result = prepare_lexical_query(
        "论文库中有哪些和注意力机制有关的论文",
        settings,
        rewriter=FakeRewriter(QueryRewrite(("注意力机制",))),
    )

    assert result.query == "attention mechanism"
    assert result.fts_query == '"attention mechanism"'
    assert result.rewriter_used is True
    assert result.debug()["core_terms"] == ["attention mechanism"]


def test_query_rewriter_failure_uses_existing_translation_fallback(tmp_path: Path, monkeypatch):
    settings = replace(Settings.load(tmp_path), query_rewriter_enabled=True, query_rewriter_api_key="test-key")
    monkeypatch.setattr(
        "paper_rag.llamaindex.translation._make_translator",
        lambda current: FakeTranslator("tencent", result="attention mechanism"),
    )

    result = prepare_lexical_query(
        "注意力机制",
        settings,
        rewriter=FakeRewriter(error=QueryRewriterError("bad response")),
    )

    assert result.rewriter_used is False
    assert result.rewriter_fallback is True
    assert result.query == "attention mechanism"
    assert any("query_rewriter" in warning for warning in result.warnings)


def test_query_rewriter_requires_at_least_one_term(tmp_path: Path, monkeypatch):
    settings = replace(Settings.load(tmp_path), query_rewriter_enabled=True, query_rewriter_api_key="test-key")
    monkeypatch.setattr(
        "paper_rag.llamaindex.translation._make_translator",
        lambda current: FakeTranslator("tencent", result="attention"),
    )
    result = prepare_lexical_query(
        "注意力机制",
        settings,
        rewriter=FakeRewriter(QueryRewrite(())),
    )

    assert result.rewriter_used is False
    assert result.rewriter_fallback is True


def test_translation_failure_warning_redacts_credentials():
    warning = _format_failure("tencent", RuntimeError("secret_key=secret-value request failed"))

    assert warning.startswith("translation_failed:tencent:RuntimeError:")
    assert "secret-value" not in warning
    assert "<redacted>" in warning
