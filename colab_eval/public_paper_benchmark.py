from __future__ import annotations

from collections import Counter
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

from config import AppConfig
from ingestion.chunking import split_documents_with_parents
from ingestion.embedding import build_embedding_model
from retrieval.query_router import route_query
from retrieval.vector_store import (
    _get_langchain_milvus_class,
    build_milvus_connection_args,
)
from services.retrieval_service import run_retrieval_flow

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


SUPPORTED_DATASETS = {
    "qasper": {
        "task_type": "qa",
        "path": "urialon/converted_qasper",
        "name": "default",
        "split": "validation",
    },
    "scifact": {
        "task_type": "retrieval",
        "path": "mteb/scifact",
        "split": "test",
    },
}

PAPER_SUITE_TARGETS = ("qasper", "scifact")


@dataclass
class SingleBenchmarkResult:
    dataset: str
    summary: dict[str, object]
    detail_df: pd.DataFrame


@dataclass
class PublicPaperBenchmarkResult:
    aggregate_summary: dict[str, object]
    dataset_results: dict[str, SingleBenchmarkResult]

def _iter_limited(
    rows: list[dict[str, object]],
    *,
    limit: int,
    seed: int,
    shuffle: bool,
) -> list[dict[str, object]]:
    copied = list(rows)
    if shuffle:
        random.Random(seed).shuffle(copied)
    if limit > 0:
        copied = copied[:limit]
    return copied


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = max(math.ceil(len(sorted_values) * 0.95) - 1, 0)
    return float(sorted_values[idx])


def _load_qasper_rows(
    *,
    path: str,
    name: str,
    split: str,
    limit: int,
    seed: int,
    shuffle: bool,
) -> list[dict[str, object]]:
    dataset = load_dataset(path, name, split=split)
    if shuffle:
        dataset = dataset.shuffle(seed=seed)
    if limit > 0:
        dataset = dataset.select(range(min(limit, len(dataset))))

    input_pattern = re.compile(r"^\s*Q:\s*(.*?)\s*Text:\s*(.*)\s*$", re.S)
    rows: list[dict[str, object]] = []
    for row in dataset:
        raw_input = str(row.get("input", "")).strip()
        answer = str(row.get("output", "")).strip()
        if not raw_input:
            continue
        match = input_pattern.match(raw_input)
        if not match:
            continue
        question = match.group(1).strip()
        context = match.group(2).strip()
        qid = str(row.get("pid", row.get("id", ""))).strip() or uuid.uuid4().hex[:12]
        if not question or not context:
            continue
        rows.append(
            {
                "qid": qid,
                "question": question,
                "context": context,
                "answers": [answer] if answer else [],
                "context_doc_id": qid,
                "relevant_doc_ids": [qid],
                "task_type": "qa",
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
) -> list[dict[str, object]]:
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
        try:
            score = float(row.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        if score <= 0:
            continue
        qid = str(row.get("query-id", "")).strip()
        cid = str(row.get("corpus-id", "")).strip()
        if qid and cid:
            positives.setdefault(qid, []).append(cid)

    rows: list[dict[str, object]] = []
    for qid, corpus_ids in positives.items():
        question = query_map.get(qid, "").strip()
        if not question or not corpus_ids:
            continue
        primary = corpus_map.get(corpus_ids[0])
        if not primary:
            continue
        title = primary["title"]
        context = f"{title}\n\n{primary['text']}".strip() if title else primary["text"]
        if not context:
            continue
        rows.append(
            {
                "qid": qid,
                "question": question,
                "context": context,
                "answers": [],
                "context_doc_id": corpus_ids[0],
                "relevant_doc_ids": list(dict.fromkeys(corpus_ids)),
                "task_type": "retrieval",
                "title": title,
            }
        )
    return _iter_limited(rows, limit=limit, seed=seed, shuffle=shuffle)


def _load_eval_rows(
    dataset_key: str,
    *,
    limit: int,
    seed: int,
    shuffle: bool,
) -> list[dict[str, object]]:
    spec = SUPPORTED_DATASETS[dataset_key]
    if dataset_key == "qasper":
        return _load_qasper_rows(
            path=spec["path"],
            name=spec["name"],
            split=spec["split"],
            limit=limit,
            seed=seed,
            shuffle=shuffle,
        )
    if dataset_key == "scifact":
        return _load_scifact_rows(
            path=spec["path"],
            split=spec["split"],
            limit=limit,
            seed=seed,
            shuffle=shuffle,
        )
    raise ValueError(f"Unsupported dataset: {dataset_key}")


def _build_embeddings(config: AppConfig):
    return build_embedding_model(
        config.embedding_provider,
        config.embedding_model,
        api_key=config.embedding_api_key,
        base_url=config.embedding_base_url,
    )


def _build_temp_vector_store(
    config: AppConfig,
    documents: list[Document],
    embeddings,
    *,
    collection_name: str,
):
    backend = config.vector_backend.strip().lower()
    if backend == "milvus":
        Milvus = _get_langchain_milvus_class()
        connection_args = build_milvus_connection_args(
            config.milvus_uri,
            milvus_token=config.milvus_token,
            milvus_db_name=config.milvus_db_name,
        )
        try:
            store = Milvus.from_documents(
                documents=documents,
                embedding=embeddings,
                collection_name=collection_name,
                connection_args=connection_args,
                drop_old=True,
            )
        except TypeError:
            store = Milvus.from_documents(
                documents,
                embeddings,
                collection_name=collection_name,
                connection_args=connection_args,
                drop_old=True,
            )

        def _cleanup() -> None:
            try:
                from pymilvus import connections, utility

                alias = f"paper_eval_cleanup_{uuid.uuid4().hex[:8]}"
                connections.connect(alias=alias, **connection_args)
                if utility.has_collection(collection_name=collection_name, using=alias):
                    utility.drop_collection(collection_name=collection_name, using=alias)
                connections.disconnect(alias=alias)
            except Exception:
                pass

        return store, _cleanup, "milvus"

    if backend == "faiss":
        from langchain_community.vectorstores import FAISS

        return FAISS.from_documents(documents, embeddings), (lambda: None), "faiss"

    raise ValueError(
        f"Unsupported vector backend for public paper benchmark: {backend}. "
        "Use milvus or faiss."
    )


def _dedupe_context_rows(rows: list[dict[str, object]], dataset_key: str) -> list[Document]:
    by_doc_id: dict[str, dict[str, object]] = {}
    for row in rows:
        doc_id = str(row.get("context_doc_id", "")).strip() or f"{dataset_key}_{uuid.uuid4().hex[:12]}"
        context = str(row.get("context", "")).strip()
        if not context or doc_id in by_doc_id:
            continue
        by_doc_id[doc_id] = row

    raw_documents: list[Document] = []
    for index, (qa_doc_id, row) in enumerate(by_doc_id.items(), start=1):
        title = str(row.get("title", "")).strip()
        context = str(row.get("context", "")).strip()
        if title and not context.startswith(title):
            page_content = f"{title}\n\n{context}"
        else:
            page_content = context
        raw_documents.append(
            Document(
                page_content=page_content,
                metadata={
                    "source": f"{dataset_key}_paper",
                    "doc_id": f"{dataset_key}_doc_{index}",
                    "qa_doc_id": qa_doc_id,
                    "title": title or qa_doc_id,
                    "paper_title": title or qa_doc_id,
                },
            )
        )
    return raw_documents


def _build_paper_corpus(raw_documents: list[Document]) -> list[Document]:
    paper_docs: list[Document] = []
    for doc in raw_documents:
        metadata = dict(doc.metadata or {})
        title = str(metadata.get("title", metadata.get("paper_title", ""))).strip()
        content = doc.page_content.strip()
        if title and not content.startswith(title):
            content = f"{title}\n\n{content}"
        paper_docs.append(Document(page_content=content, metadata=metadata))
    return paper_docs


def _first_match_rank(ids: list[str], relevant_ids: set[str]) -> int:
    if not relevant_ids:
        return 0
    for idx, item in enumerate(ids, start=1):
        if item and item in relevant_ids:
            return idx
    return 0


def _summarize_routes(route_counts: Counter[str]) -> dict[str, int]:
    return {key: int(value) for key, value in sorted(route_counts.items())}


def run_single_public_paper_benchmark(
    config: AppConfig,
    *,
    dataset_key: str,
    limit: int,
    seed: int,
    shuffle: bool,
) -> SingleBenchmarkResult:
    if dataset_key not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset: {dataset_key}. "
            f"Choose from: {', '.join(sorted(SUPPORTED_DATASETS))}"
        )

    rows = _load_eval_rows(dataset_key, limit=limit, seed=seed, shuffle=shuffle)
    if not rows:
        raise ValueError(f"No valid rows loaded from dataset: {dataset_key}")

    task_type = str(SUPPORTED_DATASETS[dataset_key]["task_type"])
    raw_documents = _dedupe_context_rows(rows, dataset_key)
    if not raw_documents:
        raise ValueError(f"No corpus documents built for dataset: {dataset_key}")

    chunk_docs, parent_docs = split_documents_with_parents(
        raw_documents,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        tokenizer_model=config.chunk_tokenizer_model,
        use_structure_split=config.chunk_use_structure_split,
        min_block_chars=config.chunk_min_block_chars,
        chunk_strategy=config.chunk_strategy,
        semantic_hard_max_chars=config.chunk_semantic_hard_max_chars,
    )
    if not chunk_docs:
        raise ValueError(f"Chunking produced no documents for dataset: {dataset_key}")

    parent_map = {
        str(doc.metadata.get("parent_id", f"parent_{idx}")): doc
        for idx, doc in enumerate(parent_docs)
    }
    paper_docs = _build_paper_corpus(raw_documents)
    chunk_corpus_key = f"{dataset_key}:chunks:{len(chunk_docs)}:{id(chunk_docs)}"
    paper_corpus_key = f"{dataset_key}:papers:{len(paper_docs)}:{id(paper_docs)}"

    embeddings = _build_embeddings(config)
    chunk_collection = f"paper_eval_{dataset_key}_chunks_{uuid.uuid4().hex[:8]}"
    paper_collection = f"paper_eval_{dataset_key}_papers_{uuid.uuid4().hex[:8]}"
    vector_store, cleanup_chunks, vector_backend = _build_temp_vector_store(
        config,
        chunk_docs,
        embeddings,
        collection_name=chunk_collection,
    )
    paper_vector_store, cleanup_papers, _ = _build_temp_vector_store(
        config,
        paper_docs,
        embeddings,
        collection_name=paper_collection,
    )

    detail_rows: list[dict[str, object]] = []
    route_counts: Counter[str] = Counter()
    paper_hit_count = 0
    paper_mrr_sum = 0.0
    evidence_hit_count = 0
    evidence_mrr_sum = 0.0
    insufficient_count = 0
    retrieval_latencies_ms: list[float] = []

    try:
        iter_rows = tqdm(rows, desc=f"Evaluating {dataset_key}", unit="q") if tqdm else rows
        for row in iter_rows:
            question = str(row["question"]).strip()
            relevant_doc_ids = {
                str(item).strip()
                for item in row.get("relevant_doc_ids", [])
                if str(item).strip()
            }

            route = route_query(config, question, scope="main")
            route_counts[route.route_type] += 1

            start = time.perf_counter()
            retrieval_result = run_retrieval_flow(
                config,
                question,
                vector_store,
                chunk_corpus=chunk_docs,
                chunk_corpus_key=chunk_corpus_key,
                paper_vector_store=paper_vector_store,
                paper_corpus=paper_docs,
                paper_corpus_key=paper_corpus_key,
                section_corpus=[],
                parent_map=parent_map,
                apply_metadata_filters=False,
                route=route,
            )
            retrieval_latency_ms = (time.perf_counter() - start) * 1000.0
            retrieval_latencies_ms.append(retrieval_latency_ms)

            retrieved_paper_ids = [
                str(doc.metadata.get("qa_doc_id", "")).strip()
                for doc in retrieval_result.paper_docs
            ]
            retrieved_evidence_ids = [
                str(doc.metadata.get("qa_doc_id", "")).strip()
                for doc in retrieval_result.evidence_docs
            ]

            paper_rank = _first_match_rank(retrieved_paper_ids, relevant_doc_ids)
            evidence_rank = _first_match_rank(retrieved_evidence_ids, relevant_doc_ids)
            paper_hit = paper_rank > 0
            evidence_hit = evidence_rank > 0
            if paper_hit:
                paper_hit_count += 1
                paper_mrr_sum += 1.0 / paper_rank
            if evidence_hit:
                evidence_hit_count += 1
                evidence_mrr_sum += 1.0 / evidence_rank

            if not retrieval_result.evidence_docs:
                insufficient_count += 1

            detail_rows.append(
                {
                    "qid": str(row.get("qid", "")).strip(),
                    "question": question,
                    "task_type": task_type,
                    "route_type": route.route_type,
                    "expected_doc_ids": " || ".join(sorted(relevant_doc_ids)),
                    "paper_doc_ids": " || ".join(item for item in retrieved_paper_ids if item),
                    "evidence_doc_ids": " || ".join(item for item in retrieved_evidence_ids if item),
                    "paper_hit": paper_hit,
                    "paper_rank": paper_rank or None,
                    "evidence_hit": evidence_hit,
                    "evidence_rank": evidence_rank or None,
                    "retrieved_papers": len(retrieval_result.paper_docs),
                    "retrieved_evidence_docs": len(retrieval_result.evidence_docs),
                    "retrieval_latency_ms": retrieval_latency_ms,
                    "insufficient_evidence": not retrieval_result.evidence_docs,
                }
            )
    finally:
        cleanup_chunks()
        cleanup_papers()

    samples = len(rows)
    summary: dict[str, object] = {
        "dataset": dataset_key,
        "task_type": task_type,
        "samples": samples,
        "corpus_papers": len(paper_docs),
        "corpus_chunks": len(chunk_docs),
        "corpus_parent_docs": len(parent_docs),
        "vector_backend": vector_backend,
        "chunk_collection": chunk_collection,
        "paper_collection": paper_collection,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "retrieval_mode": config.retrieval_mode,
        "retriever_top_k": config.retriever_top_k,
        "final_top_k": config.final_top_k,
        "query_rewrite_enabled": config.query_rewrite_enabled,
        "query_rewrite_max_variants": config.query_rewrite_max_variants,
        "use_reranker": config.use_reranker,
        "reranker_provider": config.reranker_provider,
        "reranker_model": config.reranker_model,
        "paper_hit_rate_at_k": paper_hit_count / samples,
        "paper_mrr_at_k": paper_mrr_sum / samples,
        "evidence_hit_rate_at_k": evidence_hit_count / samples,
        "evidence_mrr_at_k": evidence_mrr_sum / samples,
        "insufficient_evidence_rate": insufficient_count / samples,
        "avg_retrieval_latency_ms": sum(retrieval_latencies_ms) / samples,
        "p95_retrieval_latency_ms": _p95(retrieval_latencies_ms),
        "route_type_counts": _summarize_routes(route_counts),
        "evaluation_mode": "retrieval_only",
    }

    return SingleBenchmarkResult(
        dataset=dataset_key,
        summary=summary,
        detail_df=pd.DataFrame(detail_rows),
    )


def _weighted_average(dataset_results: dict[str, SingleBenchmarkResult], key: str) -> float:
    total_samples = sum(int(item.summary["samples"]) for item in dataset_results.values())
    if total_samples <= 0:
        return 0.0
    weighted = 0.0
    for item in dataset_results.values():
        weighted += float(item.summary.get(key, 0.0)) * int(item.summary["samples"])
    return weighted / total_samples


def _aggregate_route_counts(dataset_results: dict[str, SingleBenchmarkResult]) -> dict[str, int]:
    merged: Counter[str] = Counter()
    for item in dataset_results.values():
        for key, value in dict(item.summary.get("route_type_counts", {})).items():
            merged[str(key)] += int(value)
    return {key: int(value) for key, value in sorted(merged.items())}


def run_public_paper_benchmark(
    config: AppConfig,
    *,
    target: str = "suite",
    limit: int = 50,
    seed: int = 42,
    shuffle: bool = True,
) -> PublicPaperBenchmarkResult:
    normalized_target = str(target).strip().lower()
    if normalized_target == "suite":
        dataset_keys = list(PAPER_SUITE_TARGETS)
    else:
        if normalized_target not in SUPPORTED_DATASETS:
            raise ValueError(
                f"Unsupported target: {target}. "
                f"Choose from: suite, {', '.join(sorted(SUPPORTED_DATASETS))}"
            )
        dataset_keys = [normalized_target]

    dataset_results: dict[str, SingleBenchmarkResult] = {}
    for dataset_key in dataset_keys:
        dataset_results[dataset_key] = run_single_public_paper_benchmark(
            config,
            dataset_key=dataset_key,
            limit=limit,
            seed=seed,
            shuffle=shuffle,
        )

    aggregate_summary: dict[str, object] = {
        "target": normalized_target,
        "datasets": dataset_keys,
        "samples": sum(int(item.summary["samples"]) for item in dataset_results.values()),
        "paper_hit_rate_at_k": _weighted_average(dataset_results, "paper_hit_rate_at_k"),
        "paper_mrr_at_k": _weighted_average(dataset_results, "paper_mrr_at_k"),
        "evidence_hit_rate_at_k": _weighted_average(dataset_results, "evidence_hit_rate_at_k"),
        "evidence_mrr_at_k": _weighted_average(dataset_results, "evidence_mrr_at_k"),
        "insufficient_evidence_rate": _weighted_average(dataset_results, "insufficient_evidence_rate"),
        "avg_retrieval_latency_ms": _weighted_average(dataset_results, "avg_retrieval_latency_ms"),
        "route_type_counts": _aggregate_route_counts(dataset_results),
        "evaluation_mode": "retrieval_only",
        "dataset_summaries": {
            key: item.summary for key, item in dataset_results.items()
        },
    }

    return PublicPaperBenchmarkResult(
        aggregate_summary=aggregate_summary,
        dataset_results=dataset_results,
    )


def save_public_paper_benchmark_result(
    result: PublicPaperBenchmarkResult,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "public_paper_benchmark_summary.json").write_text(
        json.dumps(result.aggregate_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for dataset_key, dataset_result in result.dataset_results.items():
        dataset_dir = output_dir / dataset_key
        dataset_dir.mkdir(parents=True, exist_ok=True)
        (dataset_dir / "summary.json").write_text(
            json.dumps(dataset_result.summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        dataset_result.detail_df.to_csv(
            dataset_dir / "detail.csv",
            index=False,
            encoding="utf-8-sig",
        )
        failures = dataset_result.detail_df[
            (~dataset_result.detail_df["paper_hit"]) | (~dataset_result.detail_df["evidence_hit"])
        ]
        failures.to_csv(
            dataset_dir / "failures.csv",
            index=False,
            encoding="utf-8-sig",
        )
