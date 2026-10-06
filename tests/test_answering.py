from __future__ import annotations

from dataclasses import replace

from paper_rag.answering import ANSWER_CONTEXTS, AnswerContextStore, build_citation_registry, validate_answer_payload
from paper_rag.mcp.tools.answer import library_validate_answer


def _item(source_id: str = "S1") -> dict[str, object]:
    return {
        "source_id": source_id,
        "paper_id": "2309.06180",
        "canonical_id": "2309.06180v1",
        "chunk_id": "chunk-1",
        "source_chunk_ids": ["chunk-1", "chunk-2"],
        "section_path": ["2", "2.1"],
        "section_label": "PagedAttention",
        "page_start": 3,
        "page_end": 4,
        "type": "text",
        "evidence_role": "direct",
        "window_id": "W1",
        "continuity_status": "complete",
        "text": "PagedAttention manages KV cache blocks.",
    }


def test_citation_registry_only_uses_real_item_fields():
    registry = build_citation_registry([_item()])

    assert registry == {
        "S1": {
            "paper_id": "2309.06180",
            "canonical_id": "2309.06180v1",
            "chunk_id": "chunk-1",
            "section_path": ["2", "2.1"],
            "section_label": "PagedAttention",
            "page_start": 3,
            "page_end": 4,
            "type": "text",
            "evidence_role": "direct",
            "window_id": "W1",
            "continuity_status": "complete",
            "source_chunk_ids": ["chunk-1", "chunk-2"],
        }
    }


def test_valid_answer_is_accepted_and_source_metadata_is_resolved():
    ANSWER_CONTEXTS.clear()
    context = ANSWER_CONTEXTS.create(
        query="how does PagedAttention work",
        task="reason",
        mode="hybrid",
        filters={},
        regions=("content",),
        items=[_item()],
        context_text="[S1] 2309.06180v1 p.4: PagedAttention manages KV cache blocks.",
        truncated=False,
    )

    result = library_validate_answer(
        context.context_id,
        "answered",
        "PagedAttention manages KV cache blocks.[S1]",
        [{"claim_id": "C1", "text": "PagedAttention manages KV cache blocks.", "citation_ids": ["S1"]}],
        ["S1"],
    )

    assert result["status"] == "ok"
    assert result["data"]["validation"] == {"valid": True, "errors": []}
    assert result["data"]["citations"][0]["paper_id"] == "2309.06180"
    assert result["data"]["presentation"]["answer_text"].endswith("[S1]")


def test_unknown_citations_and_fabricated_claim_metadata_are_rejected():
    store = AnswerContextStore()
    context = store.create(
        query="query",
        task="fact",
        mode="hybrid",
        filters={},
        regions=("content",),
        items=[_item()],
        context_text="[S1] evidence",
        truncated=False,
    )
    errors = validate_answer_payload(
        context,
        answer_status="answered",
        answer="Claim [S2]",
        claims=[
            {
                "claim_id": "C1",
                "text": "Claim",
                "citation_ids": ["S2"],
                "paper_id": "fake-paper",
            }
        ],
        citations=["S2"],
    )

    codes = {error["code"] for error in errors}
    assert "unsupported_claim_field" in codes
    assert "unknown_inline_citation" in codes
    assert "unknown_citation_id" in codes


def test_missing_claim_citation_is_rejected():
    store = AnswerContextStore()
    context = store.create(
        query="query",
        task="fact",
        mode="hybrid",
        filters={},
        regions=("content",),
        items=[_item()],
        context_text="[S1] evidence",
        truncated=False,
    )
    errors = validate_answer_payload(
        context,
        answer_status="answered",
        answer="Claim",
        claims=[{"claim_id": "C1", "text": "Claim", "citation_ids": []}],
        citations=[],
    )

    assert any(error["code"] == "missing_claim_citation" for error in errors)


def test_non_string_top_level_citation_is_rejected():
    store = AnswerContextStore()
    context = store.create(
        query="query",
        task="fact",
        mode="hybrid",
        filters={},
        regions=("content",),
        items=[_item()],
        context_text="[S1] evidence",
        truncated=False,
    )
    errors = validate_answer_payload(
        context,
        answer_status="answered",
        answer="Claim [S1]",
        claims=[{"claim_id": "C1", "text": "Claim", "citation_ids": ["S1"]}],
        citations=[{"source_id": "S1"}],
    )

    assert any(error["code"] == "invalid_citation_id" for error in errors)


def test_inline_citation_must_be_bound_to_a_claim():
    store = AnswerContextStore()
    context = store.create(
        query="query",
        task="fact",
        mode="hybrid",
        filters={},
        regions=("content",),
        items=[_item()],
        context_text="[S1] evidence",
        truncated=False,
    )
    errors = validate_answer_payload(
        context,
        answer_status="insufficient_evidence",
        answer="Unresolved statement [S1]",
        claims=[],
        citations=[],
    )

    assert any(error["code"] == "unlinked_inline_citation" for error in errors)


def test_expired_context_is_rejected(monkeypatch):
    store = AnswerContextStore(ttl_seconds=10)
    context = store.create(
        query="query",
        task="fact",
        mode="hybrid",
        filters={},
        regions=("content",),
        items=[_item()],
        context_text="[S1] evidence",
        truncated=False,
    )
    store._contexts[context.context_id] = replace(context, created_at=0)

    monkeypatch.setattr("paper_rag.answering.time.monotonic", lambda: 100)
    assert store.get(context.context_id) is None
