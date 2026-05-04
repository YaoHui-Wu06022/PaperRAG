from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from paper_rag.config import Settings
from paper_rag.ingest.citation_graph import build_citation_graph
from paper_rag.ingest.manifest import Manifest


class CitationGraphTests(unittest.TestCase):
    def test_builds_local_active_citation_graph(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            source_dir = data / "paper_data" / "Center_Loss_abc"
            target_dir = data / "paper_data" / "ResNet_def"
            senet_dir = data / "paper_data" / "SENet_sen"
            lstm_dir = data / "paper_data" / "LSTM_lstm"
            deleted_dir = data / "paper_data" / "Deleted_xyz"
            source_dir.mkdir(parents=True)
            target_dir.mkdir(parents=True)
            senet_dir.mkdir(parents=True)
            lstm_dir.mkdir(parents=True)
            deleted_dir.mkdir(parents=True)
            (source_dir / "references.jsonl").write_text(
                "\n".join([
                    json.dumps({
                        "ref_index": 1,
                        "raw_text": "Kaiming He et al. Deep Residual Learning for Image Recognition. CVPR, 2016.",
                        "page": 9,
                        "source_block_id": "b9",
                    }),
                    json.dumps({
                        "ref_index": 2,
                        "raw_text": "Yandong Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV, 2016.",
                        "page": 9,
                        "source_block_id": "b9",
                    }),
                    json.dumps({
                        "ref_index": 3,
                        "raw_text": "Jie Hu, Li Shen, and Gang Sun. Squeeze-and-excitation networks. In CVPR, 2018.",
                        "page": 9,
                        "source_block_id": "b9",
                    }),
                    json.dumps({
                        "ref_index": 4,
                        "raw_text": "Jianpeng Cheng, Li Dong, and Mirella Lapata. Long short-term memory-networks for machine reading. arXiv preprint arXiv:1601.06733, 2016.",
                        "page": 9,
                        "source_block_id": "b9",
                    }),
                    json.dumps({
                        "ref_index": 5,
                        "raw_text": "Sepp Hochreiter and J. Schmidhuber. Long short-term memory. Neural Computation, 1997.",
                        "page": 9,
                        "source_block_id": "b9",
                    }),
                    json.dumps({
                        "ref_index": 6,
                        "raw_text": "A deleted local paper should not become a graph target.",
                        "page": 9,
                        "source_block_id": "b9",
                    }),
                ]) + "\n",
                encoding="utf-8",
            )
            (target_dir / "references.jsonl").write_text("", encoding="utf-8")
            (senet_dir / "references.jsonl").write_text("", encoding="utf-8")
            (lstm_dir / "references.jsonl").write_text("", encoding="utf-8")
            (data / "manifest.jsonl").write_text(
                "\n".join([
                    json.dumps({
                        "file_hash": "abc",
                        "status": "active",
                        "title": "A Discriminative Feature Learning Approach for Deep Face Recognition",
                        "author": ["Yandong Wen"],
                        "year": {"preprint_year": None, "publish_year": 2016},
                        "venue": "ECCV",
                        "paper_data_path": str(source_dir),
                    }, ensure_ascii=False),
                    json.dumps({
                        "file_hash": "def",
                        "status": "active",
                        "title": "Deep Residual Learning for Image Recognition",
                        "author": ["Kaiming He"],
                        "year": {"preprint_year": 2015, "publish_year": 2016},
                        "venue": "CVPR",
                        "paper_data_path": str(target_dir),
                    }, ensure_ascii=False),
                    json.dumps({
                        "file_hash": "sen",
                        "status": "active",
                        "title": "Squeeze-and-Excitation Networks",
                        "author": ["Jie Hu", "Li Shen", "Gang Sun"],
                        "year": {"preprint_year": 2017, "publish_year": 2018},
                        "venue": "2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition",
                        "paper_data_path": str(senet_dir),
                    }, ensure_ascii=False),
                    json.dumps({
                        "file_hash": "lstm",
                        "status": "active",
                        "title": "Long Short-Term Memory",
                        "author": ["Sepp Hochreiter", "J. Schmidhuber"],
                        "year": {"preprint_year": None, "publish_year": 1997},
                        "venue": "Neural Computation",
                        "paper_data_path": str(lstm_dir),
                    }, ensure_ascii=False),
                    json.dumps({
                        "file_hash": "xyz",
                        "status": "deleted",
                        "title": "A deleted local paper",
                        "author": [],
                        "year": {"preprint_year": None, "publish_year": 2018},
                        "venue": None,
                        "paper_data_path": str(deleted_dir),
                    }, ensure_ascii=False),
                ]) + "\n",
                encoding="utf-8",
            )
            settings = Settings.load(root)
            result = build_citation_graph(settings, Manifest.load(data / "manifest.jsonl"))
            graph = json.loads(result.path.read_text(encoding="utf-8"))

        self.assertEqual(result.node_count, 4)
        self.assertEqual(result.edge_count, 3)
        self.assertEqual(result.path.name, "citation_graph.json")
        self.assertEqual(
            {node["paper_id"] for node in graph["nodes"]},
            {"Center_Loss_abc", "ResNet_def", "SENet_sen", "LSTM_lstm"},
        )
        edge_by_ref = {edge["ref_index"]: edge for edge in graph["edges"]}
        self.assertEqual(set(edge_by_ref), {1, 3, 5})
        self.assertEqual(edge_by_ref[1]["source_paper_id"], "Center_Loss_abc")
        self.assertEqual(edge_by_ref[1]["target_paper_id"], "ResNet_def")
        self.assertEqual(edge_by_ref[1]["match_type"], "canonical_title")
        self.assertEqual(edge_by_ref[3]["target_paper_id"], "SENet_sen")
        self.assertEqual(edge_by_ref[5]["target_paper_id"], "LSTM_lstm")
        self.assertNotIn(4, edge_by_ref)


if __name__ == "__main__":
    unittest.main()
