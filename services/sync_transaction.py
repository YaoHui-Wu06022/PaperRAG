from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import shutil
from typing import Any

from config import AppConfig
from services import local_cache_store
from services.local_cache_store import LocalCacheState
from retrieval.vector_store import (
    build_vector_index,
    delete_documents_from_index,
    load_vector_index,
    upsert_vector_index,
)


# This module implements a tiny two-phase commit:
# remote vector-store write first, then local cache commit.
@dataclass
class SyncPlan:
    operation: str
    doc_ids: list[str]
    remote_payload: dict[str, Any]
    target_state: LocalCacheState
    ingest_cache_payload: dict[str, Any] | None = None


def has_pending_sync_operation(config: AppConfig) -> bool:
    return local_cache_store.sync_journal_path(config).exists()


def execute_sync_plan(config: AppConfig, plan: SyncPlan, embeddings):
    """Apply one sync plan with crash recovery via a journal file."""
    journal = _sync_plan_to_journal(plan, status="remote_pending")
    local_cache_store.save_sync_journal(config, journal)

    vector_store = _apply_remote_operation(config, plan, embeddings)
    journal["status"] = "local_pending"
    local_cache_store.save_sync_journal(config, journal)

    _apply_local_state(config, plan)
    local_cache_store.clear_sync_journal(config)
    return vector_store


def recover_pending_sync_operation(config: AppConfig, embeddings) -> str | None:
    """Resume an interrupted sync plan from its journal state.

    `local_pending` means the remote write already succeeded, so recovery
    should only finish the local cache commit.
    """
    payload = local_cache_store.load_sync_journal(config)
    if payload is None:
        return None

    plan = _sync_plan_from_journal(payload)
    status = str(payload.get("status", "unknown")).strip().lower()
    if status in {"remote_pending", "unknown"}:
        _apply_remote_operation(config, plan, embeddings)
    _apply_local_state(config, plan)
    local_cache_store.clear_sync_journal(config)
    return status or "unknown"


def _apply_local_state(config: AppConfig, plan: SyncPlan) -> None:
    local_cache_store.save_local_cache_state(config, plan.target_state)
    if plan.ingest_cache_payload is None:
        local_cache_store.invalidate_ingest_cache(config)
    else:
        local_cache_store.save_ingest_cache_payload(config, plan.ingest_cache_payload)


def _sync_plan_to_journal(plan: SyncPlan, *, status: str) -> dict[str, Any]:
    return {
        "operation": plan.operation,
        "status": status,
        "doc_ids": list(plan.doc_ids),
        "remote_payload": plan.remote_payload,
        "target_state": local_cache_store.local_cache_state_to_payload(plan.target_state),
        "ingest_cache_payload": plan.ingest_cache_payload,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _sync_plan_from_journal(payload: dict[str, Any]) -> SyncPlan:
    return SyncPlan(
        operation=str(payload.get("operation", "")),
        doc_ids=[str(item) for item in payload.get("doc_ids", []) if str(item).strip()],
        remote_payload=dict(payload.get("remote_payload") or {}),
        target_state=local_cache_store.local_cache_state_from_payload(
            dict(payload.get("target_state") or {})
        ),
        ingest_cache_payload=payload.get("ingest_cache_payload"),
    )


def _apply_remote_operation(config: AppConfig, plan: SyncPlan, embeddings):
    """Apply the remote half of a sync plan to the configured vector backend."""
    backend = config.vector_backend.strip().lower()
    reference_strategy = config.references_strategy.strip().lower()
    if reference_strategy not in {"keyword_only", "separate_collection"}:
        reference_strategy = "keyword_only"

    if plan.operation == "upsert":
        if backend == "faiss":
            if plan.target_state.main_chunks:
                return build_vector_index(
                    backend=config.vector_backend,
                    documents=plan.target_state.main_chunks,
                    embeddings=embeddings,
                    persist_dir=config.local_cache_dir,
                    milvus_uri=config.milvus_uri,
                    milvus_token=config.milvus_token,
                    milvus_db_name=config.milvus_db_name,
                )
            _clear_faiss_dir(config)
            return None

        main_documents = local_cache_store.deserialize_documents(
            list(plan.remote_payload.get("main_documents", []))
        )
        reference_documents = local_cache_store.deserialize_documents(
            list(plan.remote_payload.get("reference_documents", []))
        )
        paper_documents = local_cache_store.deserialize_documents(
            list(plan.remote_payload.get("paper_documents", []))
        )
        upsert_doc_ids = [
            str(item)
            for item in plan.remote_payload.get("upsert_doc_ids", plan.doc_ids)
            if str(item).strip()
        ]
        return upsert_vector_index(
            backend=config.vector_backend,
            documents=main_documents,
            embeddings=embeddings,
            persist_dir=config.local_cache_dir,
            milvus_uri=config.milvus_uri,
            milvus_token=config.milvus_token,
            milvus_db_name=config.milvus_db_name,
            milvus_collection=config.milvus_collection,
            milvus_papers_collection=config.milvus_papers_collection,
            doc_ids_to_replace=upsert_doc_ids,
            paper_documents=paper_documents,
            reference_documents=reference_documents,
            references_strategy=reference_strategy,
            milvus_references_collection=config.milvus_references_collection,
        )

    if plan.operation == "delete":
        if backend == "faiss":
            if plan.target_state.main_chunks:
                return build_vector_index(
                    backend=config.vector_backend,
                    documents=plan.target_state.main_chunks,
                    embeddings=embeddings,
                    persist_dir=config.local_cache_dir,
                    milvus_uri=config.milvus_uri,
                    milvus_token=config.milvus_token,
                    milvus_db_name=config.milvus_db_name,
                    milvus_papers_collection=config.milvus_papers_collection,
                    paper_documents=plan.target_state.paper_docs,
                )
            _clear_faiss_dir(config)
            return None

        delete_doc_ids = [
            str(item)
            for item in plan.remote_payload.get("delete_doc_ids", plan.doc_ids)
            if str(item).strip()
        ]
        delete_documents_from_index(
            backend=config.vector_backend,
            embeddings=embeddings,
            persist_dir=config.local_cache_dir,
            doc_ids=delete_doc_ids,
            milvus_uri=config.milvus_uri,
            milvus_token=config.milvus_token,
            milvus_db_name=config.milvus_db_name,
            milvus_collection=config.milvus_collection,
            milvus_papers_collection=config.milvus_papers_collection,
            references_strategy=reference_strategy,
            milvus_references_collection=config.milvus_references_collection,
        )
        try:
            return load_vector_index(
                backend=config.vector_backend,
                embeddings=embeddings,
                persist_dir=config.local_cache_dir,
                milvus_uri=config.milvus_uri,
                milvus_token=config.milvus_token,
                milvus_db_name=config.milvus_db_name,
                milvus_collection=config.milvus_collection,
            )
        except Exception:
            return None

    raise ValueError(f"Unsupported sync operation: {plan.operation}")


def _clear_faiss_dir(config: AppConfig) -> None:
    faiss_dir = config.local_cache_dir / "faiss_index"
    if faiss_dir.exists():
        shutil.rmtree(faiss_dir)
