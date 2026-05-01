from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from paper_rag.config import Settings
from paper_rag.dataprocess.ingest import clean_author_list, clean_author_name, lookup_metadata, run_ingest
from paper_rag.utils import sha256_file


class MissingClient:
    def __init__(self) -> None:
        self.called = False
        self.calls: list[dict] = []

    def lookup_exact_title(self, *args, **kwargs):
        self.called = True
        self.calls.append(kwargs)
        return None


class ErrorClient:
    def __init__(self, message: str) -> None:
        self.message = message
        self.called = False

    def lookup_exact_title(self, *args, **kwargs):
        self.called = True
        raise RuntimeError(self.message)


class Match:
    def __init__(self, title: str, authors: list[str], year: int, venue: str) -> None:
        self.title = title
        self.authors = authors
        self.year = year
        self.preprint_year = year
        self.venue = venue


class MatchClient:
    def __init__(self, match: Match) -> None:
        self.match = match
        self.called = False
        self.calls: list[dict] = []

    def lookup_exact_title(self, *args, **kwargs):
        self.called = True
        self.calls.append(kwargs)
        return self.match


class IngestMetadataLookupTests(unittest.TestCase):
    def test_clean_author_name_removes_dblp_disambiguation_suffix(self) -> None:
        self.assertEqual(clean_author_name("Zhifeng Li 0001"), "Zhifeng Li")
        self.assertEqual(clean_author_name("Yu Qiao 0001"), "Yu Qiao")
        self.assertEqual(clean_author_name("John Smith"), "John Smith")

    def test_clean_author_list_filters_blank_and_preserves_order_without_deduping(self) -> None:
        self.assertEqual(
            clean_author_list([" Yu Qiao 0001 ", " ", "Yu Qiao 0001", "John Smith"]),
            ["Yu Qiao", "Yu Qiao", "John Smith"],
        )

    def test_uses_arxiv_before_formal_sources(self) -> None:
        arxiv = MatchClient(Match("Some Paper", ["A. Author"], 2024, "ArXiv"))
        messages: list[str] = []

        match = lookup_metadata(
            "Some Paper",
            MatchClient(Match("Some Paper", ["Formal Author"], 2025, "ICML")),
            MissingClient(),
            arxiv,
            report=messages.append,
        )

        self.assertIsNotNone(match)
        self.assertEqual(match.source, "ArXiv+DBLP")
        self.assertEqual(match.year, {"preprint_year": 2024, "publish_year": 2025})
        self.assertEqual(match.venue, "ICML")
        self.assertEqual(match.authors, ["Formal Author"])
        self.assertTrue(arxiv.called)
        self.assertTrue(any("ArXiv matched preprint year" in message for message in messages))

    def test_lookup_metadata_cleans_external_authors(self) -> None:
        match = lookup_metadata(
            "Some Paper",
            MatchClient(Match("Some Paper", ["Zhifeng Li 0001", "Yu Qiao 0001"], 2025, "ICML")),
            MissingClient(),
            MissingClient(),
        )

        self.assertIsNotNone(match)
        self.assertEqual(match.authors, ["Zhifeng Li", "Yu Qiao"])

    def test_skips_corr_and_uses_semantic_scholar_formal_metadata(self) -> None:
        semantic = MatchClient(Match("Long Short-Term Memory", ["Sepp Hochreiter"], 1997, "Neural Computation"))
        arxiv = MissingClient()

        match = lookup_metadata("Long Short-Term Memory", MatchClient(Match("Long Short-Term Memory", ["A"], 1997, "CoRR")), semantic, arxiv)

        self.assertIsNotNone(match)
        self.assertEqual(match.source, "Semantic Scholar")
        self.assertTrue(arxiv.called)
        self.assertEqual(match.year, {"preprint_year": None, "publish_year": 1997})

    def test_formal_venue_year_overrides_semantic_year(self) -> None:
        semantic = MatchClient(Match(
            "Squeeze-and-Excitation Networks",
            ["Jie Hu"],
            2017,
            "2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition",
        ))

        match = lookup_metadata("Squeeze-and-Excitation Networks", MissingClient(), semantic, MissingClient())

        self.assertIsNotNone(match)
        self.assertEqual(match.source, "Semantic Scholar")
        self.assertEqual(match.year, {"preprint_year": None, "publish_year": 2018})
        self.assertEqual(match.venue, "2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition")

    def test_formal_venue_year_overrides_dblp_year(self) -> None:
        dblp = MatchClient(Match(
            "Squeeze-and-Excitation Networks",
            ["Jie Hu"],
            2017,
            "2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition",
        ))

        match = lookup_metadata("Squeeze-and-Excitation Networks", dblp, MissingClient(), MissingClient())

        self.assertIsNotNone(match)
        self.assertEqual(match.source, "DBLP")
        self.assertEqual(match.year, {"preprint_year": None, "publish_year": 2018})

    def test_formal_publish_year_falls_back_to_source_year_without_venue_year(self) -> None:
        semantic = MatchClient(Match("Long Short-Term Memory", ["Sepp Hochreiter"], 1997, "Neural Computation"))

        match = lookup_metadata("Long Short-Term Memory", MissingClient(), semantic, MissingClient())

        self.assertIsNotNone(match)
        self.assertEqual(match.year, {"preprint_year": None, "publish_year": 1997})

    def test_arxiv_only_match_has_no_venue(self) -> None:
        match = lookup_metadata("Some Paper", MissingClient(), MissingClient(), MatchClient(Match("Some Paper", ["A"], 2024, "ArXiv")))
        self.assertIsNotNone(match)
        self.assertEqual(match.source, "ArXiv")
        self.assertEqual(match.year, {"preprint_year": 2024, "publish_year": None})
        self.assertIsNone(match.venue)

    def test_uses_source_delay_as_retry_delay(self) -> None:
        dblp = MissingClient()
        semantic = MissingClient()
        arxiv = MissingClient()

        lookup_metadata(
            "Some Paper",
            dblp,
            semantic,
            arxiv,
            dblp_retry_delay_seconds=1.5,
            semantic_scholar_retry_delay_seconds=5.5,
            arxiv_retry_delay_seconds=3.5,
        )

        self.assertEqual(arxiv.calls[0]["retry_delay_seconds"], 3.5)
        self.assertEqual(dblp.calls[0]["retry_delay_seconds"], 1.5)
        self.assertEqual(semantic.calls[0]["retry_delay_seconds"], 5.5)

    def test_run_ingest_cleans_existing_manifest_authors_without_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = Settings.load(root)
            settings.pdf_dir.mkdir(parents=True)
            settings.mineru_output_dir.mkdir(parents=True)
            settings.paper_data_dir.mkdir(parents=True)
            pdf_path = settings.pdf_dir / "Some_Paper.pdf"
            pdf_path.write_bytes(b"%PDF-1.4 fake")
            file_hash = sha256_file(pdf_path)
            settings.manifest_path.parent.mkdir(parents=True, exist_ok=True)
            settings.manifest_path.write_text(
                json.dumps(
                    {
                        "file_hash": file_hash,
                        "status": "active",
                        "pdf_path": str(pdf_path),
                        "title": "Some Paper",
                        "author": ["Zhifeng Li 0001", "Yu Qiao 0001"],
                        "year": {"preprint_year": None, "publish_year": 2025},
                        "venue": "ICML",
                        "mineru_output_path": None,
                        "archived_mineru_output_path": None,
                        "paper_data_path": None,
                        "message": None,
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            (settings.data_dir / "paper_annotations.json").write_text(
                json.dumps({
                    file_hash: {
                        "title": "Old Title",
                        "aliases": ["Some alias"],
                        "tags": ["manual tag"],
                    },
                }),
                encoding="utf-8",
            )
            mineru_dir = settings.mineru_output_dir / "Some_Paper"
            mineru_dir.mkdir()

            def fake_extract(_mineru_output, paper_data_dir, metadata, **_kwargs):
                paper_data_dir.mkdir(parents=True)
                (paper_data_dir / "metadata.json").write_text(
                    json.dumps(metadata, ensure_ascii=False),
                    encoding="utf-8",
                )
                return SimpleNamespace(
                    title=metadata["title"],
                    paper_data_dir=paper_data_dir,
                    block_count=1,
                    reference_count=0,
                    chunk_count=0,
                    warnings=[],
                )

            with (
                patch("paper_rag.dataprocess.ingest.ensure_mineru_output", return_value=mineru_dir),
                patch("paper_rag.dataprocess.ingest.title_from_output", return_value="Some Paper"),
                patch("paper_rag.dataprocess.ingest.rename_pdf_if_needed", return_value=pdf_path),
                patch("paper_rag.dataprocess.ingest.rename_mineru_output_if_needed", return_value=mineru_dir),
                patch("paper_rag.dataprocess.ingest.extract_paper_data", side_effect=fake_extract),
                patch("paper_rag.dataprocess.ingest.build_citation_graph", return_value=SimpleNamespace(node_count=1, edge_count=0, path=settings.paper_data_dir / "citation_graph.json")),
            ):
                run_ingest(settings)

            manifest_record = json.loads(settings.manifest_path.read_text(encoding="utf-8").splitlines()[0])
            metadata_record = json.loads(
                (Path(manifest_record["paper_data_path"]) / "metadata.json").read_text(encoding="utf-8")
            )
            annotations = json.loads((settings.data_dir / "paper_annotations.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest_record["author"], ["Zhifeng Li", "Yu Qiao"])
            self.assertEqual(metadata_record["author"], ["Zhifeng Li", "Yu Qiao"])
            self.assertEqual(annotations[file_hash], {
                "title": "Some Paper",
                "aliases": ["Some alias"],
                "tags": ["manual tag"],
            })


if __name__ == "__main__":
    unittest.main()
