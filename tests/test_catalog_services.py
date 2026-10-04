from __future__ import annotations

import json
from pathlib import Path
import sqlite3

import pytest

import paper_rag.catalog.service as catalog_service
from paper_rag.catalog.service import (
    CatalogIndexNotReady,
    get_asset_status,
    get_assets,
    rebuild_catalog,
    scan_catalog,
    search_catalog,
)
from paper_rag.config import Settings
from paper_rag.query.service import query_papers
from paper_rag.query.schemas import QueryIntent


def _settings(tmp_path: Path) -> Settings:
    settings = Settings.load(tmp_path)
    source = settings.arxiv_data_dir / "1706.03762"
    source.mkdir(parents=True)
    (source / "paper.pdf").write_bytes(b"%PDF-test")
    (source / "metadata.json").write_text(
        json.dumps(
            {
                "base_id": "1706.03762",
                "canonical_id": "1706.03762v7",
                "title": "Attention Is All You Need",
                "authors": ["Alice"],
                "abstract": "A Transformer paper.",
                "categories": ["cs.CL"],
            }
        ),
        encoding="utf-8",
    )
    return settings


def test_catalog_scans_assets_and_rebuilds_sqlite(tmp_path: Path):
    settings = _settings(tmp_path)

    records = scan_catalog(settings)
    assert len(records) == 1
    assert records[0].state == "ready_for_ingest"
    assert get_asset_status(settings, "1706.03762")["mineru"] == "missing"
    assert get_assets(settings, "1706.03762")["assets"]["pdf"]["present"] is True

    result = rebuild_catalog(settings)
    assert result["papers"] == 1
    assert settings.paper_catalog_db_path.is_file()


def test_catalog_search_is_structured_and_does_not_use_jev(tmp_path: Path):
    settings = _settings(tmp_path)

    with pytest.raises(CatalogIndexNotReady):
        search_catalog(settings, "Transformer")

    rebuild_catalog(settings)
    records = search_catalog(settings, "Transformer")
    assert [record.base_id for record in records] == ["1706.03762"]
    assert [record.base_id for record in search_catalog(settings, "Alice")] == ["1706.03762"]
    assert [record.base_id for record in search_catalog(settings, "cs.CL")] == ["1706.03762"]
    assert [record.base_id for record in search_catalog(settings, "1706.03762")] == ["1706.03762"]


def test_catalog_rebuild_is_atomic_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    settings = _settings(tmp_path)
    rebuild_catalog(settings)
    database = settings.paper_catalog_db_path
    before = database.read_bytes()

    def fail(*_args, **_kwargs):
        raise sqlite3.DatabaseError("test failure")

    monkeypatch.setattr(catalog_service, "_create_database", fail)
    with pytest.raises(sqlite3.DatabaseError):
        rebuild_catalog(settings)

    assert database.read_bytes() == before
    assert not list(database.parent.glob(f".{database.name}.*.tmp"))


def test_paper_query_reports_missing_content_without_asset_status(tmp_path: Path):
    settings = _settings(tmp_path)
    result = query_papers(settings, "1706.03762 如何实现这个方法")

    assert result.decision.intent == QueryIntent.PAPER_CONTENT
    assert result.capabilities["content_available"] is False
    assert result.capabilities["missing_assets"] == ["1706.03762:mineru/full.md"]
    assert result.to_dict()["read_only"] is True


def test_paper_query_discovery_reports_unready_index_without_writing(tmp_path: Path):
    settings = _settings(tmp_path)
    result = query_papers(settings, "找 Transformer 论文")

    assert result.capabilities["status"] == "index_not_ready"
    assert not settings.paper_catalog_db_path.exists()


def test_paper_query_discovery_uses_the_synced_fts_index(tmp_path: Path):
    settings = _settings(tmp_path)
    rebuild_catalog(settings)

    result = query_papers(settings, "找 Transformer 论文")

    assert result.capabilities == {"index_ready": True}
    assert [item["base_id"] for item in result.items] == ["1706.03762"]
