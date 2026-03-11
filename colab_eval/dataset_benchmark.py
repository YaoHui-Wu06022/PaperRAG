from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import re
import time
import uuid

import pandas as pd
from datasets import load_dataset
from langchain_core.documents import Document
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

from config import AppConfig
from colab_eval.ragas_eval import run_ragas_eval_rows
from generation.llm import build_llm_client
from generation.prompt import build_qa_prompt
from ingestion.chunking import split_documents_with_parents
from ingestion.embedding import build_embedding_model
from retrieval.vector_store import (
    build_milvus_connection_args,
    _get_langchain_milvus_class,
)
from services.retrieval_service import run_retrieval_flow


SUPPORTED_DATASETS = {
    "squad": {
        "loader": "qa_pair",
        "path": "squad",
        "name": "plain_text",
        "split": "validation",
    },
    "cmrc2018": {
        "loader": "qa_pair",
        "path": "cmrc2018",
        "name": "default",
        "split": "validation",
    },
    "xquad_zh": {
        "loader": "qa_pair",
        "path": "xquad",
        "name": "xquad.zh",
        "split": "validation",
    },
    "scifact": {
        "loader": "mteb_scifact",
        "path": "mteb/scifact",
        "split": "test",
    },
    "qasper": {
        "loader": "converted_qasper",
        "path": "urialon/converted_qasper",
        "name": "default",
        "split": "validation",
    },
}


INSUFFICIENT_EVIDENCE_ANSWER = (
    "根据当前检索到的资料，未找到充分证据支持回答该问题。"
    "请尝试换一种问法，或补充相关文档后再提问。"
)


def _is_insufficient_answer(answer: str) -> bool:
    text = (answer or "").strip()
    if not text:
        return True
    if text == INSUFFICIENT_EVIDENCE_ANSWER:
        return True
    return "证据不足" in text


@dataclass
class DatasetBenchmarkResult:
    summary: dict[str, float | int | str | bool]
    detail_df: pd.DataFrame
    ragas_df: pd.DataFrame | None
    ragas_error: str | None


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def _contains_any_answer(text: str, answers: list[str]) -> bool:
    normalized_text = _normalize_text(text)
    if not normalized_text:
        return False
    for answer in answers:
        normalized_answer = _normalize_text(answer)
        if normalized_answer and normalized_answer in normalized_text:
            return True
    return False


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = max(math.ceil(len(sorted_values) * 0.95) - 1, 0)
    return sorted_values[idx]


def _iter_limited(
    rows: list[dict],
    *,
    limit: int,
    seed: int,
    shuffle: bool,
) -> list[dict]:
    copied = list(rows)
    if shuffle:
        random.Random(seed).shuffle(copied)
    if limit > 0:
        copied = copied[:limit]
    return copied


def _load_qa_pair_rows(
    *,
    path: str,
    name: str,
    split: str,
    limit: int,
    seed: int,
    shuffle: bool,
) -> list[dict]:
    dataset = load_dataset(path, name, split=split)
    if shuffle:
        dataset = dataset.shuffle(seed=seed)
    if limit > 0:
        dataset = dataset.select(range(min(limit, len(dataset))))

    rows: list[dict] = []
    for row in dataset:
        question = str(row.get("question", "")).strip()
        context = str(row.get("context", "")).strip()
        answers_raw = row.get("answers", {})
        answers_list = []
        if isinstance(answers_raw, dict):
            for item in answers_raw.get("text", []):
                text = str(item).strip()
                if text:
                    answers_list.append(text)

        if not question or not context or not answers_list:
            continue

        rows.append(
            {
                "qid": str(row.get("id", "")),
                "question": question,
                "context": context,
                "answers": answers_list,
            }
        )
    return rows


def _load_scifact_rows(
    *,
    path: str,
    split: str,
    limit: int,
    seed: int,
    shuffle: bool,
) -> list[dict]:
    qrels = load_dataset(path, "default", split=split)
    query_ds = load_dataset(path, "queries", split="queries")
    corpus_ds = load_dataset(path, "corpus", split="corpus")

    query_map = {
        str(row.get("_id", "")).strip(): str(row.get("text", "")).strip()
        for row in query_ds
    }
    corpus_map = {
        str(row.get("_id", "")).strip(): {
            "title": str(row.get("title", "")).strip(),
            "text": str(row.get("text", "")).strip(),
        }
        for row in corpus_ds
    }

    positives: dict[str, list[str]] = {}
    for row in qrels:
        score = row.get("score", 0.0)
        try:
            score_value = float(score)
        except (TypeError, ValueError):
            score_value = 0.0
        if score_value <= 0:
            continue
        qid = str(row.get("query-id", "")).strip()
        cid = str(row.get("corpus-id", "")).strip()
        if not qid or not cid:
            continue
        positives.setdefault(qid, []).append(cid)

    rows: list[dict] = []
    for qid, corpus_ids in positives.items():
        question = query_map.get(qid, "").strip()
        if not question or not corpus_ids:
            continue
        primary_cid = corpus_ids[0]
        corpus_row = corpus_map.get(primary_cid)
        if not corpus_row:
            continue

        title = corpus_row["title"]
        text = corpus_row["text"]
        context = f"{title}\n\n{text}".strip() if title else text
        if not context:
            continue
        answers = [title] if title else []

        rows.append(
            {
                "qid": qid,
                "question": question,
                "context": context,
                "answers": answers,
                "context_doc_id": primary_cid,
                "relevant_doc_ids": list(dict.fromkeys(corpus_ids)),
            }
        )
    return _iter_limited(rows, limit=limit, seed=seed, shuffle=shuffle)


_QASPER_INPUT_PATTERN = re.compile(r"^\s*Q:\s*(.*?)\s*Text:\s*(.*)\s*$", re.S)


def _load_qasper_rows(
    *,
    path: str,
    name: str,
    split: str,
    limit: int,
    seed: int,
    shuffle: bool,
) -> list[dict]:
    dataset = load_dataset(path, name, split=split)
    if shuffle:
        dataset = dataset.shuffle(seed=seed)
    if limit > 0:
        dataset = dataset.select(range(min(limit, len(dataset))))

    rows: list[dict] = []
    for row in dataset:
        raw_input = str(row.get("input", "")).strip()
        answer = str(row.get("output", "")).strip()
        if not raw_input:
            continue
        match = _QASPER_INPUT_PATTERN.match(raw_input)
        if not match:
            continue
        question = match.group(1).strip()
        context = match.group(2).strip()
        if not question or not context:
            continue

        pid = str(row.get("pid", "")).strip()
        if not pid:
            pid = str(row.get("id", "")).strip()
        if not pid:
            pid = f"qasper_ctx_{uuid.uuid4().hex[:12]}"

        answers = [answer] if answer else []
        rows.append(
            {
                "qid": pid,
                "question": question,
                "context": context,
                "answers": answers,
                "context_doc_id": pid,
                "relevant_doc_ids": [pid],
            }
        )
    return rows


def _load_qa_rows(
    dataset_key: str,
    *,
    limit: int,
    seed: int,
    shuffle: bool,
) -> list[dict]:
    if dataset_key not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset: {dataset_key}. "
            f"Choose from: {', '.join(sorted(SUPPORTED_DATASETS))}"
        )

    spec = SUPPORTED_DATASETS[dataset_key]
    loader = spec.get("loader", "qa_pair")
    if loader == "qa_pair":
        return _load_qa_pair_rows(
            path=spec["path"],
            name=spec["name"],
            split=spec["split"],
            limit=limit,
            seed=seed,
            shuffle=shuffle,
        )
    if loader == "mteb_scifact":
        return _load_scifact_rows(
            path=spec["path"],
            split=spec["split"],
            limit=limit,
            seed=seed,
            shuffle=shuffle,
        )
    if loader == "converted_qasper":
        return _load_qasper_rows(
            path=spec["path"],
            name=spec["name"],
            split=spec["split"],
            limit=limit,
            seed=seed,
            shuffle=shuffle,
        )
    raise ValueError(f"Unsupported dataset loader: {loader}")


def _build_embeddings(config: AppConfig):
    return build_embedding_model(
        config.embedding_provider,
        config.embedding_model,
        api_key=config.embedding_api_key,
        base_url=config.embedding_base_url,
    )


def _build_llm(config: AppConfig):
    return build_llm_client(
        model=config.llm_model,
        temperature=config.llm_temperature,
        api_key=config.llm_api_key,
        base_url=config.llm_base_url,
        api_mode=config.llm_api_mode,
    )


def _build_temp_vector_store(
    config: AppConfig,
    documents: list[Document],
    embeddings,
    *,
    dataset_key: str,
):
    backend = config.vector_backend.strip().lower()
    if backend == "faiss":
        from langchain_community.vectorstores import FAISS

        store = FAISS.from_documents(documents, embeddings)
        return store, lambda: None, "faiss", ""

    if backend == "milvus":
        Milvus = _get_langchain_milvus_class()

        collection = f"rag_bench_{dataset_key}_{uuid.uuid4().hex[:8]}"
        connection_args = build_milvus_connection_args(
            config.milvus_uri,
            milvus_token=config.milvus_token,
            milvus_db_name=config.milvus_db_name,
        )
        try:
            store = Milvus.from_documents(
                documents=documents,
                embedding=embeddings,
                collection_name=collection,
                connection_args=connection_args,
                drop_old=True,
            )
        except TypeError:
            store = Milvus.from_documents(
                documents,
                embeddings,
                collection_name=collection,
                connection_args=connection_args,
                drop_old=True,
            )

        def _cleanup() -> None:
            try:
                from pymilvus import connections, utility

                alias = f"rag_bench_cleanup_{uuid.uuid4().hex[:8]}"
                connections.connect(alias=alias, **connection_args)
                if utility.has_collection(collection_name=collection, using=alias):
                    utility.drop_collection(collection_name=collection, using=alias)
                connections.disconnect(alias=alias)
            except Exception:
                pass

        return store, _cleanup, "milvus", collection

    raise ValueError(
        f"Unsupported vector backend for dataset benchmark: {backend}. "
        "Use milvus or faiss."
    )


def run_dataset_benchmark(
    config: AppConfig,
    *,
    dataset_key: str,
    limit: int = 100,
    seed: int = 42,
    shuffle: bool = True,
    with_llm: bool = False,
    with_ragas: bool = False,
) -> DatasetBenchmarkResult:
    if with_ragas and not with_llm:
        raise ValueError("with_ragas=True requires with_llm=True")

    qa_rows = _load_qa_rows(
        dataset_key,
        limit=limit,
        seed=seed,
        shuffle=shuffle,
    )
    if not qa_rows:
        raise ValueError("No valid QA rows loaded from dataset.")

    unique_contexts: dict[str, int] = {}
    context_doc_id_by_text: dict[str, str] = {}
    raw_documents: list[Document] = []
    corpus_rows = (
        tqdm(qa_rows, desc="Preparing corpus", unit="qa")
        if tqdm is not None
        else qa_rows
    )
    for row in corpus_rows:
        context = row["context"]
        if context in unique_contexts:
            existing_context_doc_id = context_doc_id_by_text.get(context)
            if existing_context_doc_id:
                row["context_doc_id"] = existing_context_doc_id
                if row.get("relevant_doc_ids"):
                    row["relevant_doc_ids"] = [existing_context_doc_id]
            continue
        doc_id = len(unique_contexts)
        unique_contexts[context] = doc_id
        context_doc_id = str(row.get("context_doc_id", "")).strip()
        if context_doc_id:
            context_doc_id_by_text[context] = context_doc_id
        raw_documents.append(
            Document(
                page_content=context,
                metadata={
                    "source": f"{dataset_key}_context",
                    "doc_id": doc_id,
                    "qa_doc_id": context_doc_id,
                },
            )
        )

    chunked_documents, parent_documents = split_documents_with_parents(
        raw_documents,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        tokenizer_model=config.chunk_tokenizer_model,
        use_structure_split=config.chunk_use_structure_split,
        min_block_chars=config.chunk_min_block_chars,
        chunk_strategy=config.chunk_strategy,
        semantic_hard_max_chars=config.chunk_semantic_hard_max_chars,
    )
    if not chunked_documents:
        raise ValueError("Chunking produced no documents for benchmark.")
    parent_map = {
        str(doc.metadata.get("parent_id", f"parent_{idx}")): doc
        for idx, doc in enumerate(parent_documents)
    }

    embeddings = _build_embeddings(config)
    vector_store, cleanup_fn, vector_backend, vector_collection = _build_temp_vector_store(
        config,
        chunked_documents,
        embeddings,
        dataset_key=dataset_key,
    )

    llm = _build_llm(config) if with_llm else None
    detail_rows: list[dict] = []
    ragas_rows: list[dict] = []
    retrieval_latencies_ms: list[float] = []
    llm_latencies_ms: list[float] = []
    hit_count = 0
    reciprocal_rank_sum = 0.0
    insufficient_count = 0
    answer_hit_count = 0

    try:
        eval_rows = (
            tqdm(qa_rows, desc="Evaluating", unit="qa")
            if tqdm is not None
            else qa_rows
        )
        for row_idx, row in enumerate(eval_rows, start=1):
            question = row["question"]
            answers = row["answers"]
            relevant_doc_ids = {
                str(item).strip()
                for item in row.get("relevant_doc_ids", [])
                if str(item).strip()
            }
            start = time.perf_counter()
            retrieval_result = run_retrieval_flow(
                config,
                question,
                vector_store,
                chunk_corpus=chunked_documents,
                chunk_corpus_key=f"{dataset_key}:{len(chunked_documents)}:{id(chunked_documents)}",
                parent_map=parent_map,
                apply_metadata_filters=False,
            )
            docs = retrieval_result.evidence_docs
            retrieval_latency_ms = (time.perf_counter() - start) * 1000.0
            retrieval_latencies_ms.append(retrieval_latency_ms)

            retrieved_contexts = [doc.page_content for doc in docs]
            retrieved_doc_ids = [
                str(doc.metadata.get("qa_doc_id", "")).strip() for doc in docs
            ]
            matched_rank = 0
            matched_doc_id = ""
            if relevant_doc_ids:
                for idx, doc_id_text in enumerate(retrieved_doc_ids, start=1):
                    if doc_id_text and doc_id_text in relevant_doc_ids:
                        matched_rank = idx
                        matched_doc_id = doc_id_text
                        break
            else:
                for idx, context in enumerate(retrieved_contexts, start=1):
                    if _contains_any_answer(context, answers):
                        matched_rank = idx
                        break

            retrieval_hit = matched_rank > 0
            if retrieval_hit:
                hit_count += 1
                reciprocal_rank_sum += 1.0 / matched_rank

            if not docs:
                insufficient_count += 1

            answer_text = ""
            answer_hit = False
            if llm is not None:
                generation_docs = retrieval_result.generation_docs or docs

                llm_start = time.perf_counter()
                if not docs:
                    answer_text = INSUFFICIENT_EVIDENCE_ANSWER
                else:
                    prompt = build_qa_prompt(question=question, documents=generation_docs)
                    answer_text = llm.generate(prompt)
                llm_latency_ms = (time.perf_counter() - llm_start) * 1000.0
                llm_latencies_ms.append(llm_latency_ms)
                if docs and _is_insufficient_answer(answer_text):
                    insufficient_count += 1
                answer_hit = _contains_any_answer(answer_text, answers)
                if answer_hit:
                    answer_hit_count += 1

                if with_ragas:
                    generation_contexts = [doc.page_content for doc in generation_docs]
                    ragas_rows.append(
                        {
                            "question": question,
                            "answer": answer_text,
                            "contexts": generation_contexts,
                            "ground_truth": answers[0],
                        }
                    )

            detail_rows.append(
                {
                    "question": question,
                    "answers": " || ".join(answers),
                    "relevant_doc_ids": " || ".join(sorted(relevant_doc_ids)),
                    "retrieved_docs": len(docs),
                    "retrieved_doc_ids": " || ".join(
                        item for item in retrieved_doc_ids if item
                    ),
                    "retrieval_hit": retrieval_hit,
                    "matched_rank": matched_rank if matched_rank else None,
                    "matched_doc_id": matched_doc_id or None,
                    "retrieval_latency_ms": retrieval_latency_ms,
                    "insufficient_evidence": not docs,
                    "answer_hit": answer_hit if llm is not None else None,
                    "answer_preview": answer_text[:200] if answer_text else "",
                }
            )
            if tqdm is not None:
                avg_ms = sum(retrieval_latencies_ms) / len(retrieval_latencies_ms)
                eval_rows.set_postfix(
                    hit=hit_count,
                    hit_rate=f"{hit_count / row_idx:.2%}",
                    avg_ms=f"{avg_ms:.0f}",
                )
    finally:
        cleanup_fn()

    samples = len(qa_rows)
    summary: dict[str, float | int | str | bool] = {
        "dataset": dataset_key,
        "samples": samples,
        "corpus_docs": len(chunked_documents),
        "corpus_raw_docs": len(raw_documents),
        "corpus_chunks": len(chunked_documents),
        "corpus_parent_docs": len(parent_documents),
        "vector_backend": vector_backend,
        "vector_collection": vector_collection,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "chunk_tokenizer_model": config.chunk_tokenizer_model,
        "chunk_use_structure_split": config.chunk_use_structure_split,
        "chunk_min_block_chars": config.chunk_min_block_chars,
        "retriever_top_k": config.retriever_top_k,
        "final_top_k": config.final_top_k,
        "retrieval_mode": config.retrieval_mode,
        "hybrid_dense_top_k": config.hybrid_dense_top_k,
        "hybrid_bm25_top_k": config.hybrid_bm25_top_k,
        "hybrid_rrf_k": config.hybrid_rrf_k,
        "query_rewrite_enabled": config.query_rewrite_enabled,
        "query_rewrite_max_variants": config.query_rewrite_max_variants,
        "use_reranker": config.use_reranker,
        "rerank_conditional_enabled": config.rerank_conditional_enabled,
        "rerank_skip_min_top1_score": config.rerank_skip_min_top1_score,
        "rerank_skip_min_score_gap": config.rerank_skip_min_score_gap,
        "rerank_skip_min_rel_gap": config.rerank_skip_min_rel_gap,
        "diversify_by_source": config.diversify_by_source,
        "max_chunks_per_source": config.max_chunks_per_source,
        "retrieval_score_threshold": (
            config.retrieval_score_threshold
            if config.retrieval_score_threshold is not None
            else "None"
        ),
        "retrieval_score_threshold_mode": config.retrieval_score_threshold_mode,
        "retrieval_score_relative_ratio": config.retrieval_score_relative_ratio,
        "retrieval_score_quantile": config.retrieval_score_quantile,
        "reranker_score_threshold": (
            config.reranker_score_threshold
            if config.reranker_score_threshold is not None
            else "None"
        ),
        "retrieval_hit_rate_at_k": hit_count / samples,
        "retrieval_mrr_at_k": reciprocal_rank_sum / samples,
        "avg_retrieval_latency_ms": sum(retrieval_latencies_ms) / samples,
        "p95_retrieval_latency_ms": _p95(retrieval_latencies_ms),
        "insufficient_evidence_rate": insufficient_count / samples,
    }

    if llm is not None and llm_latencies_ms:
        summary["with_llm"] = True
        summary["answer_hit_rate"] = answer_hit_count / samples
        summary["avg_llm_latency_ms"] = sum(llm_latencies_ms) / len(llm_latencies_ms)
        summary["p95_llm_latency_ms"] = _p95(llm_latencies_ms)
    else:
        summary["with_llm"] = False

    ragas_df: pd.DataFrame | None = None
    ragas_error: str | None = None
    if with_ragas:
        try:
            ragas_df = run_ragas_eval_rows(ragas_rows)
        except Exception as exc:
            ragas_error = str(exc)

    return DatasetBenchmarkResult(
        summary=summary,
        detail_df=pd.DataFrame(detail_rows),
        ragas_df=ragas_df,
        ragas_error=ragas_error,
    )


def save_dataset_benchmark_result(
    result: DatasetBenchmarkResult,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "dataset_benchmark_detail.csv"
    result.detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")

    summary_path = output_dir / "dataset_benchmark_summary.json"
    summary_path.write_text(
        json.dumps(result.summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if result.ragas_df is not None:
        ragas_path = output_dir / "dataset_benchmark_ragas.csv"
        result.ragas_df.to_csv(ragas_path, index=False, encoding="utf-8-sig")
