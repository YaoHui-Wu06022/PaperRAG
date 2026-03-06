from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd
from datasets import Dataset


@dataclass
class EvalSample:
    question: str
    ground_truth: str


def _run_ragas_on_rows(rows: list[dict]) -> pd.DataFrame:
    try:
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as exc:
        raise ImportError(
            "RAGAS not installed. Install `ragas` to run evaluation."
        ) from exc

    dataset = Dataset.from_pandas(pd.DataFrame(rows))
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    )
    return result.to_pandas()


def run_ragas_eval(
    samples: list[EvalSample],
    answer_fn: Callable[[str], tuple[str, list[str]]],
) -> pd.DataFrame:
    """
    `answer_fn(question)` 需要返回 `(answer, retrieved_contexts)`。
    其中 `retrieved_contexts` 为本次回答使用的检索上下文列表。
    """
    rows: list[dict] = []
    for sample in samples:
        answer, contexts = answer_fn(sample.question)
        rows.append(
            {
                "question": sample.question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": sample.ground_truth,
            }
        )
    return _run_ragas_on_rows(rows)


def run_ragas_eval_rows(rows: list[dict]) -> pd.DataFrame:
    """
    直接对已准备好的评估样本运行 RAGAS。
    每行需要包含：question, answer, contexts, ground_truth。
    """
    required = {"question", "answer", "contexts", "ground_truth"}
    normalized: list[dict] = []
    for idx, row in enumerate(rows, 1):
        missing = required - set(row.keys())
        if missing:
            raise ValueError(
                f"Invalid eval row at index {idx}: missing fields {sorted(missing)}"
            )
        normalized.append(
            {
                "question": str(row["question"]),
                "answer": str(row["answer"]),
                "contexts": list(row["contexts"]),
                "ground_truth": str(row["ground_truth"]),
            }
        )
    return _run_ragas_on_rows(normalized)
