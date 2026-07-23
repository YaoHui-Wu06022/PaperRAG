import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from paper_rag.corpus.chunks import filter_content_retrieval_chunks, load_chunk_documents
from paper_rag.ingest import extract as extract_module
from paper_rag.ingest.extract import extract_paper_data
from paper_rag.ingest.manifest import Manifest, ManifestRecord, effective_year
from paper_rag.ingest.mineru import MinerUError
from paper_rag.ingest.pipeline import IngestSummary, build_existing_output_index, ensure_mineru_output, lookup_metadata


@dataclass
class MetadataSourceHit:
    title: str
    authors: list[str]
    year: int
    venue: str | None = None
    preprint_year: int | None = None


class FakeArxiv:
    def lookup_exact_title(self, title: str, *, retry_delay_seconds: float):
        return MetadataSourceHit(
            title=title,
            authors=["Alice Smith 0001"],
            year=2015,
            venue="arXiv",
            preprint_year=2015,
        )


class FakeDblp:
    def lookup_exact_title(self, title: str, *, limit: int, retry_delay_seconds: float):
        return MetadataSourceHit(
            title=title,
            authors=["Alice Smith 0001", "Bob Lee"],
            year=2016,
            venue="CVPR 2016",
        )


class FakeSemanticScholar:
    def lookup_exact_title(self, title: str, *, retry_delay_seconds: float):
        raise AssertionError("DBLP formal venue should stop Semantic Scholar fallback")


def test_manifest_roundtrip_and_metadata_lookup_prefers_formal_venue(settings):
    record = ManifestRecord(
        file_hash="hash-a",
        status="active",
        title="Deep Residual Learning for Image Recognition",
        year=2015,
    )
    manifest = Manifest(settings.manifest_path)
    manifest.records[record.file_hash] = record
    manifest.save()

    loaded = Manifest.load(settings.manifest_path).records["hash-a"]
    assert loaded.year == {"preprint_year": None, "publish_year": 2015}
    assert effective_year(loaded.year) == 2015

    match = lookup_metadata(
        "Deep Residual Learning for Image Recognition",
        FakeDblp(),
        FakeSemanticScholar(),
        FakeArxiv(),
    )
    assert match is not None
    assert match.year == {"preprint_year": 2015, "publish_year": 2016}
    assert match.authors == ["Alice Smith", "Bob Lee"]
    assert match.venue == "CVPR 2016"


def test_extract_mineru_output_splits_blocks_chunks_references_and_appendix(settings, tmp_path: Path):
    mineru_dir = tmp_path / "mineru"
    mineru_dir.mkdir()
    content = [[
        title("A Tiny Paper"),
        title("Abstract"),
        paragraph("We introduce residual connections for optimization."),
        title("1 Introduction"),
        paragraph("Residual connections ease training of deep networks."),
        image("Figure 1: Residual block architecture.", "images/residual-block.jpg"),
        table(
            "Table 1: Model comparison.",
            "<table><tr><th>Model</th><th>Accuracy</th></tr>"
            "<tr><td>ResNet</td><td>76.2</td></tr></table>",
        ),
        title("Appendix"),
        paragraph("Extra ablation details live in the appendix."),
        title("References"),
        reference_list(
            "[7] He K. Deep Residual Learning for Image Recognition. 2016.",
            "Smith J. A reference without a printed number. 2017.",
            "[2] Zhang X. A later entry with a non-sequential number. 2018.",
        ),
    ]]
    (mineru_dir / "upload-id_content_list_v2.json").write_text(json.dumps(content, ensure_ascii=False), encoding="utf-8")

    result = extract_paper_data(
        mineru_dir,
        settings.paper_data_dir / "tiny_paper",
        {
            "title": "A Tiny Paper",
            "author": ["Alice Smith"],
            "year": {"preprint_year": None, "publish_year": 2020},
            "venue": "UnitTest",
            "pdf_path": "tiny.pdf",
        },
        chunk_target_chars=80,
        chunk_overlap_chars=10,
    )

    assert result.block_count >= 3
    assert result.reference_count == 3
    references = read_jsonl(result.references_path)
    assert references[0]["raw_text"].startswith("[7] He K.")
    assert [reference["ref_index"] for reference in references] == [1, 2, 3]
    assert [reference["reference_id"] for reference in references] == ["ref_001", "ref_002", "ref_003"]

    blocks = read_jsonl(result.blocks_path)
    image_block = next(block for block in blocks if block["type"] == "image")
    assert image_block["caption"] == "Figure 1: Residual block architecture."
    assert image_block["source_path"] == "images/residual-block.jpg"

    table_block = next(block for block in blocks if block["type"] == "table")
    assert table_block["caption"] == "Table 1: Model comparison."
    assert "<table>" in table_block["html"]
    assert "Columns: Model, Accuracy." in table_block["text"]
    assert "Row 1: Model = ResNet; Accuracy = 76.2." in table_block["text"]

    chunks = read_jsonl(result.chunks_path)
    regions = {chunk["region"] for chunk in chunks}
    assert {"abstract", "body", "appendix"} <= regions
    assert all("Deep Residual Learning" not in chunk["text"] for chunk in chunks)
    body_text = "\n".join(chunk["text"] for chunk in chunks if chunk["region"] == "body")
    assert "Figure 1: Residual block architecture." in body_text
    assert "Row 1: Model = ResNet; Accuracy = 76.2." in body_text

    loaded_chunks = load_chunk_documents(settings.paper_data_dir)
    content_chunks = filter_content_retrieval_chunks(loaded_chunks)
    assert {chunk.region for chunk in content_chunks} == {"abstract", "body"}


def test_mineru_reuse_requires_exact_title_or_source_hash(settings):
    output_dir = make_mineru_output(settings.mineru_output_dir, "old_output", "A Tiny Paper")
    index = build_existing_output_index(settings.mineru_output_dir)

    reused = ensure_mineru_output(
        settings,
        ManifestRecord(file_hash="hash-a", status="new"),
        settings.pdf_dir / "2014_A_Tiny_Paper.pdf",
        "hash-a",
        index,
        IngestSummary(),
    )

    assert reused == output_dir
    sidecar = json.loads((output_dir / "_paper_rag_source.json").read_text(encoding="utf-8"))
    assert sidecar["file_hash"] == "hash-a"


def test_mineru_reuse_rejects_fuzzy_or_ambiguous_title_matches(settings):
    make_mineru_output(settings.mineru_output_dir, "extended", "A Tiny Paper Extended")
    fuzzy_index = build_existing_output_index(settings.mineru_output_dir)

    with pytest.raises(MinerUError, match="不精确"):
        ensure_mineru_output(
            settings,
            ManifestRecord(file_hash="hash-a", status="new"),
            settings.pdf_dir / "2014_A_Tiny_Paper.pdf",
            "hash-a",
            fuzzy_index,
            IngestSummary(),
        )

    settings.mineru_output_dir.mkdir(parents=True, exist_ok=True)
    for child in settings.mineru_output_dir.iterdir():
        if child.is_dir():
            import shutil

            shutil.rmtree(child)
    make_mineru_output(settings.mineru_output_dir, "one", "A Tiny Paper")
    make_mineru_output(settings.mineru_output_dir, "two", "A Tiny Paper")
    ambiguous_index = build_existing_output_index(settings.mineru_output_dir)

    with pytest.raises(MinerUError, match="不唯一"):
        ensure_mineru_output(
            settings,
            ManifestRecord(file_hash="hash-a", status="new"),
            settings.pdf_dir / "2014_A_Tiny_Paper.pdf",
            "hash-a",
            ambiguous_index,
            IngestSummary(),
        )


def test_mineru_reuse_manifest_and_archive_paths_have_priority(settings):
    manifest_output = make_mineru_output(settings.mineru_output_dir, "manifest_output", "Different Title")
    reused = ensure_mineru_output(
        settings,
        ManifestRecord(file_hash="hash-a", status="active", mineru_output_path=str(manifest_output)),
        settings.pdf_dir / "2014_A_Tiny_Paper.pdf",
        "hash-a",
        build_existing_output_index(settings.mineru_output_dir),
        IngestSummary(),
    )
    assert reused == manifest_output

    archived_output = make_mineru_output(settings.archive_dir, "archived_output", "Archived Title")
    restored = ensure_mineru_output(
        settings,
        ManifestRecord(file_hash="hash-b", status="deleted", archived_mineru_output_path=str(archived_output)),
        settings.pdf_dir / "2015_Archived_Title.pdf",
        "hash-b",
        build_existing_output_index(settings.mineru_output_dir),
        IngestSummary(),
    )
    assert restored == settings.mineru_output_dir / "archived_output"
    assert restored.exists()


def test_extract_paper_data_failure_preserves_existing_output(settings, tmp_path: Path, monkeypatch):
    mineru_dir = make_mineru_output(tmp_path, "mineru", "Stable Paper")
    target = settings.paper_data_dir / "stable_paper"
    target.mkdir(parents=True)
    (target / "metadata.json").write_text("old metadata", encoding="utf-8")

    original_write_jsonl = extract_module.write_jsonl

    def fail_on_chunks(path: Path, rows: list[dict]) -> None:
        if path.name == "chunks.jsonl":
            raise RuntimeError("write failed")
        original_write_jsonl(path, rows)

    monkeypatch.setattr(extract_module, "write_jsonl", fail_on_chunks)

    with pytest.raises(RuntimeError, match="write failed"):
        extract_module.extract_paper_data(
            mineru_dir,
            target,
            {
                "title": "Stable Paper",
                "author": [],
                "year": {"preprint_year": None, "publish_year": 2020},
                "venue": "UnitTest",
                "pdf_path": "stable.pdf",
            },
        )

    assert (target / "metadata.json").read_text(encoding="utf-8") == "old metadata"


def make_mineru_output(root: Path, name: str, paper_title: str) -> Path:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    content = [[
        title(paper_title),
        title("Abstract"),
        paragraph("A short abstract."),
        title("1 Introduction"),
        paragraph("A short body."),
    ]]
    (directory / "upload-id_content_list_v2.json").write_text(
        json.dumps(content, ensure_ascii=False),
        encoding="utf-8",
    )
    return directory


def title(text: str) -> dict:
    return {"type": "title", "content": {"title_content": [{"content": text}]}}


def paragraph(text: str) -> dict:
    return {"type": "paragraph", "content": {"paragraph_content": [{"content": text}]}}


def image(caption: str, path: str) -> dict:
    return {
        "type": "image",
        "content": {
            "image_caption": [{"content": caption}],
            "image_source": {"path": path},
        },
    }


def table(caption: str, html: str) -> dict:
    return {
        "type": "table",
        "content": {
            "table_caption": [{"content": caption}],
            "html": html,
        },
    }


def reference_list(*texts: str) -> dict:
    return {
        "type": "list",
        "content": {
            "list_type": "reference_list",
            "list_items": [{"item_content": [{"content": text}]} for text in texts],
        },
    }


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
