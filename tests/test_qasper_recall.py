from __future__ import annotations

import os
import unittest

from config import load_config
from evaluation.dataset_benchmark import run_dataset_benchmark
from services.health import ensure_startup_ready


class QasperRecallBenchmarkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config()
        ensure_startup_ready(
            cls.config,
            require_mineru=False,
            require_llm=False,
            require_local_cache=False,
            require_milvus=True,
        )
        cls.limit = max(1, int(os.getenv("QASPER_RECALL_TEST_LIMIT", "5")))

    def test_qasper_recall_smoke(self) -> None:
        result = run_dataset_benchmark(
            self.config,
            dataset_key="qasper",
            limit=self.limit,
            seed=42,
            shuffle=False,
            with_llm=False,
            with_ragas=False,
        )

        summary = result.summary
        samples = int(summary["samples"])
        hit_rate = float(summary["retrieval_hit_rate_at_k"])
        mrr = float(summary["retrieval_mrr_at_k"])
        avg_latency_ms = float(summary["avg_retrieval_latency_ms"])

        self.assertEqual(summary["dataset"], "qasper")
        self.assertFalse(bool(summary["with_llm"]))
        self.assertGreater(samples, 0)
        self.assertEqual(len(result.detail_df), samples)
        self.assertTrue(0.0 <= hit_rate <= 1.0)
        self.assertTrue(0.0 <= mrr <= 1.0)
        self.assertGreater(avg_latency_ms, 0.0)
        self.assertFalse(result.detail_df["retrieval_hit"].isna().any())
        self.assertGreater(
            hit_rate,
            0.0,
            "Qasper recall benchmark returned zero hits; retrieval likely regressed.",
        )


if __name__ == "__main__":
    unittest.main()
