from __future__ import annotations

import argparse
from pathlib import Path
import sys
import requests

from config import load_config
from ingestion.embedding import build_embedding_model
from services.health import build_startup_health_report, ensure_startup_ready
from services.sync_transaction import (
    has_pending_sync_operation,
    recover_pending_sync_operation,
)

# `main.py` 是很薄的一层 CLI 入口。
# 它应该只负责参数解析、启动检查，以及把请求分发到 `pipeline.py` 的编排层。

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

    return parser


def _startup_requirements(args: argparse.Namespace) -> dict[str, bool]:
    """声明每个 CLI 命令要求哪些依赖处于健康状态。"""
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
        "require_llm": command in {"ask", "chat"},
        "require_local_cache": command in {"ask", "chat", "delete-doc"},
        "require_milvus": command in {
            "ingest",
            "delete-doc",
            "ask",
            "chat",
        },
    }


def _build_recovery_embeddings(config):
    # 恢复 sync journal 只需要 embeddings 来重新连上向量库。
    return build_embedding_model(
        config.embedding_provider,
        config.embedding_model,
        openai_api_key=config.openai_api_key,
        openai_base_url=config.openai_base_url,
        aihubmix_api_key=config.aihubmix_api_key,
        aihubmix_base_url=config.aihubmix_base_url,
    )


def _recover_pending_sync_if_needed(config) -> None:
    """在接受新命令前，先完成上次中断的 sync journal 恢复。"""
    if not has_pending_sync_operation(config):
        return
    embeddings = _build_recovery_embeddings(config)
    status = recover_pending_sync_operation(config, embeddings)
    print(f"Recovered pending sync journal (previous_status={status}).")


def _configure_stdio() -> None:
    """保证 CLI 输出在 Windows GBK 控制台里也能正常打印。"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")


def main() -> None:
    """`ingest / ask / chat / delete / health` 的 CLI 入口。"""
    _configure_stdio()
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

    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
