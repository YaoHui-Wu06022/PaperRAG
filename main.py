from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import math
import time

import pandas as pd
import requests

from config import load_config
from ingestion.embedding import build_embedding_model
from services.health import build_startup_health_report, ensure_startup_ready
from services.sync_transaction import (
    has_pending_sync_operation,
    recover_pending_sync_operation,
)


def _load_eval_samples(sample_path: Path) -> list[dict[str, str]]:
    if not sample_path.exists():
        raise FileNotFoundError(f"Eval samples not found: {sample_path}")

    suffix = sample_path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict[str, str]] = []
        for idx, line in enumerate(sample_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {idx}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Invalid JSONL row at line {idx}: expected object")
            question = str(item.get("question", "")).strip()
            ground_truth = str(item.get("ground_truth", "")).strip()
            if not question or not ground_truth:
                raise ValueError(
                    f"Invalid JSONL row at line {idx}: question/ground_truth required"
                )
            rows.append({"question": question, "ground_truth": ground_truth})
        return rows

    if suffix == ".csv":
        df = pd.read_csv(sample_path)
        required = {"question", "ground_truth"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV missing required columns: {', '.join(sorted(missing))}"
            )
        rows = []
        for _, row in df.iterrows():
            question = str(row["question"]).strip()
            ground_truth = str(row["ground_truth"]).strip()
            if question and ground_truth:
                rows.append({"question": question, "ground_truth": ground_truth})
        return rows

    raise ValueError("Unsupported sample file type. Use .jsonl or .csv")


def _safe_console_text(text: str) -> str:
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding, errors="replace")


def _is_insufficient_answer(answer: str, canonical: str) -> bool:
    text = (answer or "").strip()
    if not text:
        return True
    if text == canonical:
        return True
    return "证据不足" in text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RAG-based PDF Knowledge Base QA System"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Build vector index")
    ingest_parser.add_argument(
        "--pdf",
        type=Path,
        default=None,
        help="Optional single PDF path. If omitted, ingest data/pdf/*.pdf",
    )
    ingest_parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild index even when ingest cache is hit",
    )

    delete_parser = subparsers.add_parser(
        "delete-doc",
        help="Delete one or more doc_id from vector store and local corpora",
    )
    delete_parser.add_argument(
        "doc_ids",
        nargs="+",
        help="doc_id list, separated by spaces",
    )

    ask_parser = subparsers.add_parser("ask", help="Ask a question")
    ask_parser.add_argument("question", type=str, help="Question string")
    ask_parser.add_argument(
        "--scope",
        type=str,
        choices=["main", "references"],
        default="main",
        help="Choose whether to answer from main content or references only",
    )

    chat_parser = subparsers.add_parser("chat", help="Interactive chat mode")
    chat_parser.add_argument(
        "--exit-key",
        type=str,
        default="/exit",
        help="Input this text to end chat session",
    )
    chat_parser.add_argument(
        "--scope",
        type=str,
        choices=["main", "references"],
        default="main",
        help="Choose whether to answer from main content or references only",
    )

    models_parser = subparsers.add_parser(
        "models",
        help="List available models from AIHubMix",
    )
    models_parser.add_argument(
        "--free-only",
        action="store_true",
        help="Show only models whose id contains 'free'",
    )

    subparsers.add_parser(
        "health",
        help="Run startup validation and dependency health checks",
    )

    eval_parser = subparsers.add_parser(
        "eval",
        help="Run RAGAS evaluation and output metrics",
    )
    eval_parser.add_argument(
        "--samples",
        type=Path,
        default=None,
        help="Eval sample file path (.jsonl or .csv), requires columns question/ground_truth",
    )
    eval_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output CSV path for RAGAS summary",
    )
    eval_parser.add_argument(
        "--skip-ragas",
        action="store_true",
        help="Only output system metrics, skip RAGAS scoring",
    )

    dataset_eval_parser = subparsers.add_parser(
        "eval-dataset",
        help="Benchmark retrieval QA on public datasets (SQuAD/CMRC/XQuAD/SciFact/Qasper)",
    )
    dataset_eval_parser.add_argument(
        "--dataset",
        type=str,
        default="squad",
        choices=["squad", "cmrc2018", "xquad_zh", "scifact", "qasper"],
        help="Dataset key",
    )
    dataset_eval_parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of samples to evaluate",
    )
    dataset_eval_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffle",
    )
    dataset_eval_parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable shuffle before sampling",
    )
    dataset_eval_parser.add_argument(
        "--with-llm",
        action="store_true",
        help="Run generation evaluation (consumes LLM quota)",
    )
    dataset_eval_parser.add_argument(
        "--with-ragas",
        action="store_true",
        help="Run RAGAS on generated answers (requires --with-llm)",
    )
    dataset_eval_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory for detail CSV and summary JSON",
    )

    return parser


def _startup_requirements(args: argparse.Namespace) -> dict[str, bool]:
    command = args.command
    if command == "health":
        return {
            "require_mineru": True,
            "require_llm": True,
            "require_local_cache": False,
            "require_milvus": True,
        }
    return {
        "require_mineru": command == "ingest",
        "require_llm": command in {"ask", "chat", "eval"} or bool(
            command == "eval-dataset" and (args.with_llm or args.with_ragas)
        ),
        "require_local_cache": command in {"ask", "chat", "delete-doc", "eval"},
        "require_milvus": command in {
            "ingest",
            "delete-doc",
            "ask",
            "chat",
            "eval",
            "eval-dataset",
        },
    }


def _build_recovery_embeddings(config):
    return build_embedding_model(
        config.embedding_provider,
        config.embedding_model,
        openai_api_key=config.openai_api_key,
        openai_base_url=config.openai_base_url,
        aihubmix_api_key=config.aihubmix_api_key,
        aihubmix_base_url=config.aihubmix_base_url,
    )


def _recover_pending_sync_if_needed(config) -> None:
    if not has_pending_sync_operation(config):
        return
    embeddings = _build_recovery_embeddings(config)
    status = recover_pending_sync_operation(config, embeddings)
    print(f"Recovered pending sync journal (previous_status={status}).")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config()
    requirements = _startup_requirements(args)

    try:
        _recover_pending_sync_if_needed(config)
        if args.command == "health":
            report = build_startup_health_report(config, **requirements)
            print("Startup health report:")
            for item in report:
                prefix = "OK" if item.ok else "FAIL"
                print(f"- [{prefix}] {item.name}: {item.message}")
            fatal_failures = [item for item in report if item.fatal and not item.ok]
            if fatal_failures:
                sys.exit(1)
            return

        ensure_startup_ready(config, **requirements)

        if args.command == "ingest":
            from pipeline import ingest_documents

            result = ingest_documents(config, args.pdf, force=args.force)
            if result.skipped:
                if result.raw_documents == 0 and result.chunks == 0:
                    print("No new doc_id found. Use `--force` to overwrite existing documents.")
                else:
                    print(
                        "Ingest cache hit: source PDFs and config unchanged, "
                        "skip rebuilding index."
                    )
            else:
                print(
                    f"Indexed {result.raw_documents} pages into {result.chunks} chunks."
                )
            if result.reference_chunks or result.suspicious_reference_chunks:
                print(f"Reference chunks: {result.reference_chunks}")
                print(
                    "Suspicious reference chunks: "
                    f"{result.suspicious_reference_chunks}"
                )
            print(f"Local cache path: {result.cache_dir}")
            return

        if args.command == "delete-doc":
            from pipeline import delete_documents

            result = delete_documents(config, args.doc_ids)
            print("Delete summary:")
            print(f"- requested_doc_ids: {len(result.requested_doc_ids)}")
            print(f"- deleted_doc_ids: {len(result.deleted_doc_ids)}")
            if result.deleted_doc_ids:
                for doc_id in result.deleted_doc_ids:
                    print(f"  - {doc_id}")
            print(f"- removed_chunks: {result.removed_chunks}")
            print(f"- removed_parents: {result.removed_parents}")
            print(f"- removed_reference_chunks: {result.removed_reference_chunks}")
            print(f"- removed_block_rows: {result.removed_block_rows}")
            print(f"- removed_structured_chunk_rows: {result.removed_structured_chunk_rows}")
            print(f"- removed_reference_keyword_rows: {result.removed_reference_keyword_rows}")
            return

        if args.command == "ask":
            from pipeline import answer_question

            result = answer_question(config, args.question, scope=args.scope)
            print("Answer:")
            print(result.answer)
            print(f"\nRetrieval scope: {result.retrieval_scope}")
            print("\nSources:")
            for citation in result.citations:
                print(f"- {citation}")
            return

        if args.command == "chat":
            from pipeline import answer_question

            print("Interactive chat started.")
            print(f"Type `{args.exit_key}` to exit.")
            while True:
                question = input("\nQuestion: ").strip()
                if not question:
                    continue
                if question == args.exit_key:
                    print("Chat ended.")
                    break
                result = answer_question(config, question, scope=args.scope)
                print("\nAnswer:")
                print(result.answer)
                print(f"\nRetrieval scope: {result.retrieval_scope}")
                print("\nSources:")
                for citation in result.citations:
                    print(f"- {citation}")
            return

        if args.command == "models":
            if not config.aihubmix_api_key:
                raise ValueError("AIHUBMIX_API_KEY is empty. Please set it in .env.")
            endpoint = f"{config.aihubmix_base_url.rstrip('/')}/models"
            response = requests.get(
                endpoint,
                headers={"Authorization": f"Bearer {config.aihubmix_api_key}"},
                timeout=60,
            )
            if response.status_code >= 400:
                raise RuntimeError(
                    f"Request failed ({response.status_code}): {response.text}"
                )
            payload = response.json()
            model_ids = sorted(
                item.get("id", "")
                for item in payload.get("data", [])
                if isinstance(item, dict) and item.get("id")
            )
            if args.free_only:
                model_ids = [m for m in model_ids if "free" in m.lower()]
            print(f"Total models: {len(model_ids)}")
            for model_id in model_ids:
                print(f"- {model_id}")
            if not args.free_only:
                free_like = [m for m in model_ids if "free" in m.lower()]
                if free_like:
                    print("\nPotential free models:")
                    for model_id in free_like:
                        print(f"- {model_id}")
                else:
                    print(
                        "\nNo 'free' keyword found in model IDs. "
                        "Use AIHubMix website filters to confirm free quota."
                    )
            return

        if args.command == "eval":
            from evaluation.ragas_eval import run_ragas_eval_rows
            from pipeline import INSUFFICIENT_EVIDENCE_ANSWER, answer_question

            if args.samples is None:
                raise ValueError(
                    "Please provide --samples for eval, e.g. "
                    "`python main.py eval --samples path/to/eval.jsonl --skip-ragas`"
                )
            rows = _load_eval_samples(args.samples)
            if not rows:
                raise ValueError("No valid eval samples found.")

            latencies_ms: list[float] = []
            reject_count = 0
            eval_rows: list[dict] = []

            for row in rows:
                question = row["question"]
                start = time.perf_counter()
                result = answer_question(config, question)
                latencies_ms.append((time.perf_counter() - start) * 1000.0)
                if _is_insufficient_answer(result.answer, INSUFFICIENT_EVIDENCE_ANSWER):
                    reject_count += 1
                eval_rows.append(
                    {
                        "question": question,
                        "answer": result.answer,
                        "contexts": result.contexts,
                        "ground_truth": row["ground_truth"],
                    }
                )

            ragas_df: pd.DataFrame | None = None
            if args.skip_ragas:
                print("RAGAS skipped by --skip-ragas")
            else:
                try:
                    ragas_df = run_ragas_eval_rows(eval_rows)
                    print("RAGAS summary:")
                    print(_safe_console_text(ragas_df.to_string(index=False)))
                except Exception as ragas_exc:
                    print(f"RAGAS failed: {ragas_exc}")

            avg_latency = sum(latencies_ms) / len(latencies_ms)
            p95_idx = max(math.ceil(len(latencies_ms) * 0.95) - 1, 0)
            sorted_lat = sorted(latencies_ms)
            p95_latency = sorted_lat[p95_idx]
            reject_rate = reject_count / len(rows)

            print("\nSystem metrics:")
            print(f"- samples: {len(rows)}")
            print(f"- avg_latency_ms: {avg_latency:.2f}")
            print(f"- p95_latency_ms: {p95_latency:.2f}")
            print(f"- insufficient_evidence_count: {reject_count}")
            print(f"- insufficient_evidence_rate: {reject_rate:.2%}")

            if args.output and ragas_df is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                ragas_df.to_csv(args.output, index=False, encoding="utf-8-sig")
                print(f"\nSaved RAGAS summary CSV: {args.output}")
            elif args.output and ragas_df is None:
                print("\nRAGAS summary unavailable, skip CSV export.")
            return

        if args.command == "eval-dataset":
            from evaluation.dataset_benchmark import (
                run_dataset_benchmark,
                save_dataset_benchmark_result,
            )

            result = run_dataset_benchmark(
                config,
                dataset_key=args.dataset,
                limit=args.limit,
                seed=args.seed,
                shuffle=not args.no_shuffle,
                with_llm=args.with_llm,
                with_ragas=args.with_ragas,
            )

            print("Dataset benchmark summary:")
            for key, value in result.summary.items():
                if isinstance(value, float):
                    print(f"- {key}: {value:.6f}")
                else:
                    print(f"- {key}: {value}")

            if result.ragas_df is not None:
                print("\nRAGAS summary:")
                print(_safe_console_text(result.ragas_df.to_string(index=False)))
            elif args.with_ragas and result.ragas_error:
                print(f"\nRAGAS failed: {result.ragas_error}")

            if args.output_dir:
                save_dataset_benchmark_result(result, args.output_dir)
                print(f"\nSaved benchmark outputs to: {args.output_dir}")
            return
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
