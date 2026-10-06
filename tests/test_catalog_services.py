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
from paper_rag.catalog.service import citation_graph
from paper_rag.catalog.references import PaperIdentity, extract_references, normalize_doi, resolve_references
from paper_rag.config import Settings


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
    graph_path = settings.paper_catalog_db_path.with_name("citation_graph.json")
    assert graph_path.is_file()
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    assert graph["scope"] == "local_catalog"
    assert graph["paper_count"] == 1


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


def test_metadata_query_remains_separate_from_body_rag(tmp_path: Path):
    settings = _settings(tmp_path)
    rebuild_catalog(settings)

    records = search_catalog(settings, "Transformer", {"category": "cs.CL"})
    assert [item.base_id for item in records] == ["1706.03762"]


def test_citation_graph_walks_multiple_hops_without_chunks(tmp_path: Path):
    settings = _settings(tmp_path)
    rebuild_catalog(settings)
    with sqlite3.connect(settings.paper_catalog_db_path) as connection:
        connection.executemany(
            "INSERT INTO citation_edges VALUES (?, ?, 'cites', 'local')",
            [("1706.03762", "1111.11111"), ("1111.11111", "2222.22222")],
        )
        connection.commit()

    graph = citation_graph(settings, "1706.03762", direction="out", depth=2)
    assert graph["nodes"] == ["1111.11111", "1706.03762", "2222.22222"]
    assert {edge["depth"] for edge in graph["edges"]} == {1, 2}
    assert graph["edges"][0]["path"] == ["1706.03762", "1111.11111"]
    assert graph["edges"][1]["path"] == ["1706.03762", "1111.11111", "2222.22222"]


def test_citation_graph_keeps_incoming_two_hop_direction(tmp_path: Path):
    settings = _settings(tmp_path)
    rebuild_catalog(settings)
    with sqlite3.connect(settings.paper_catalog_db_path) as connection:
        connection.executemany(
            "INSERT INTO citation_edges VALUES (?, ?, 'cites', 'local')",
            [("3333.33333", "1706.03762"), ("4444.44444", "3333.33333")],
        )
        connection.commit()

    graph = citation_graph(settings, "1706.03762", direction="in", depth=2)
    assert graph["edges"][0]["path"] == ["3333.33333", "1706.03762"]
    assert graph["edges"][1]["path"] == ["4444.44444", "3333.33333", "1706.03762"]


def test_citation_graph_version_suffix_and_node_filters(tmp_path: Path):
    settings = _settings(tmp_path)
    rebuild_catalog(settings)
    with sqlite3.connect(settings.paper_catalog_db_path) as connection:
        connection.executemany(
            "INSERT INTO citation_edges VALUES (?, ?, 'cites', 'local')",
            [("1706.03762", "1111.11111"), ("2222.22222", "1706.03762")],
        )
        connection.commit()

    graph = citation_graph(settings, "1706.03762v7", direction="both", depth=1)
    assert len(graph["edges"]) == 2


def test_citation_graph_rejects_depth_above_two(tmp_path: Path):
    settings = _settings(tmp_path)
    rebuild_catalog(settings)

    with pytest.raises(ValueError, match="depth must be 1 or 2"):
        citation_graph(settings, "1706.03762", depth=3)


def test_citation_graph_defaults_to_two_hops(tmp_path: Path):
    settings = _settings(tmp_path)
    rebuild_catalog(settings)

    graph = citation_graph(settings, "1706.03762")
    assert graph["depth"] == 2


def test_reference_resolution_removes_only_version_suffix():
    refs, _warnings = extract_references(
        [
            {"type": "heading", "text": "References", "text_level": 1},
            {"type": "ref_text", "ref_text": "[1] arXiv:2101.00001v2"},
            {"type": "heading", "text": "Appendix", "text_level": 1},
            {"type": "text", "text": "not a reference"},
        ],
        source_paper_id="1706.03762",
        source_canonical_id="1706.03762v7",
        local_ids={"2101.00001"},
    )
    assert refs[0].resolution == "local"


def test_references_after_appendix_are_still_extracted():
    refs, _warnings = extract_references(
        [
            {"type": "text", "text": "Appendix", "text_level": 1},
            {"type": "text", "text": "Appendix evidence"},
            {"type": "text", "text": "References", "text_level": 1},
            {"type": "ref_text", "ref_text": "[1] arXiv:2101.00001"},
        ],
        source_paper_id="1905.03175",
        source_canonical_id="1905.03175v1",
        local_ids={"2101.00001"},
    )
    assert len(refs) == 1
    assert refs[0].resolution == "local"


def test_reference_doi_normalization_and_exact_match():
    refs, _warnings = extract_references(
        [
            {"type": "heading", "text": "References", "text_level": 1},
            {"type": "ref_text", "ref_text": "[1] doi:10.1000/ABC.123)."},
        ],
        source_paper_id="1706.03762",
        source_canonical_id="1706.03762v7",
        local_ids=set(),
    )
    identities = [PaperIdentity("2101.00001", "2101.00001", "2101.00001v2", "A Local Paper", ("Alice Smith",), "2021-01-01", "https://doi.org/10.1000/abc.123")]
    resolved, stats, _warnings = resolve_references(refs, identities)
    assert normalize_doi("doi:10.1000/ABC.123).") == "10.1000/abc.123"
    assert resolved[0].matched_paper_id == "2101.00001"
    assert resolved[0].match_method == "doi_exact"
    assert stats["doi_exact"] == 1


def test_reference_title_author_year_match_and_external_is_not_local_edge():
    refs, _warnings = extract_references(
        [
            {"type": "heading", "text": "References", "text_level": 1},
            {"type": "ref_text", "ref_text": "[1] A Local Paper. Alice Smith. 2021."},
            {"type": "ref_text", "ref_text": "[2] arXiv:9999.00001v1"},
        ],
        source_paper_id="1706.03762",
        source_canonical_id="1706.03762v7",
        local_ids=set(),
    )
    identities = [PaperIdentity("2101.00001", "2101.00001", "2101.00001v2", "A Local Paper", ("Alice Smith",), "2021-01-01")]
    resolved, stats, _warnings = resolve_references(refs, identities)
    assert resolved[0].match_method == "title_author_year"
    assert resolved[0].matched_paper_id == "2101.00001"
    assert resolved[1].resolution == "external"
    assert resolved[1].matched_paper_id is None
    assert stats["title_author_year"] == 1
    assert stats["external"] == 1


def test_reference_duplicate_target_keeps_one_canonical_match():
    refs, _warnings = extract_references(
        [
            {"type": "heading", "text": "References", "text_level": 1},
            {"type": "ref_text", "ref_text": "[1] A Local Paper. Alice Smith. 2021a."},
            {"type": "ref_text", "ref_text": "[2] A Local Paper. arXiv:2101.00001. 2021b."},
        ],
        source_paper_id="1706.03762",
        source_canonical_id="1706.03762v7",
        local_ids={"2101.00001"},
    )
    identities = [PaperIdentity("2101.00001", "2101.00001", "2101.00001v2", "A Local Paper", ("Alice Smith",), "2021-01-01")]
    resolved, _stats, warnings = resolve_references(refs, identities)
    assert resolved[1].match_method == "arxiv_exact"
    assert resolved[0].duplicate_of_reference_id == resolved[1].reference_id
    assert any(item.startswith("duplicates:") for item in warnings)
