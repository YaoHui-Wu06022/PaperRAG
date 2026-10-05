from __future__ import annotations

from types import SimpleNamespace


def test_library_citation_resolves_title_without_metadata_search(monkeypatch):
    from paper_rag.mcp.tools import catalog as catalog_tools

    record = SimpleNamespace(
        paper_id="1706.03762",
        base_id="1706.03762",
        canonical_id="1706.03762v7",
        title="Attention Is All You Need",
    )
    monkeypatch.setattr(catalog_tools, "get_settings", lambda: object())
    monkeypatch.setattr(catalog_tools, "scan_catalog", lambda _settings: [record])
    monkeypatch.setattr(
        catalog_tools,
        "get_references",
        lambda _settings, paper_id: {
            "paper_id": paper_id,
            "items": [
                {
                    "matched_paper_id": "1512.03385",
                    "resolution": "local",
                    "ordinal": 1,
                }
            ],
            "scope": "local_catalog",
        },
    )
    monkeypatch.setattr(
        catalog_tools,
        "_load_titles",
        lambda _settings, _paper_ids: {"1512.03385": "Deep Residual Learning for Image Recognition"},
    )

    result = catalog_tools.library_citation(
        paper_title="Attention Is All You Need",
        mode="references",
    )

    assert result["status"] == "ok"
    assert result["data"]["paper_id"] == "1706.03762"
    assert result["data"]["presentation"]["answer_text"].startswith(
        "论文《Attention Is All You Need》共有 1 条参考文献。"
    )

