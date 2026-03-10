from __future__ import annotations

from dataclasses import dataclass

from config import AppConfig
from retrieval.query_router import QueryRoute
from retrieval.vector_store import (
    build_vector_index,
    vector_index_entity_count,
    vector_index_exists,
)
from services import local_cache_store
from services.paper_representation import build_paper_assets


# This module centralizes knowledge-base validation.
# `ask` uses lightweight readiness checks, while `ingest` can do repairs.
@dataclass(frozen=True)
class KnowledgeBaseHealth:
    ok: bool
    repaired: bool
    reasons: tuple[str, ...]


def check_query_readiness(
    config: AppConfig,
    *,
    route: QueryRoute,
) -> KnowledgeBaseHealth:
    """Run the cheap, read-only checks required before answering a question."""
    state = local_cache_store.load_local_cache_state(config)
    reasons: list[str] = []

    if route.retrieval_scope == "references":
        if not state.reference_chunks or not local_cache_store.reference_chunk_corpus_path(config).exists():
            reasons.append("missing_reference_corpus")
        return KnowledgeBaseHealth(ok=not reasons, repaired=False, reasons=tuple(reasons))

    if not state.main_chunks or not local_cache_store.chunk_corpus_path(config).exists():
        reasons.append("missing_main_chunk_corpus")
    if not state.parent_docs or not local_cache_store.parent_corpus_path(config).exists():
        reasons.append("missing_parent_corpus")
    if not state.paper_docs or not local_cache_store.paper_corpus_path(config).exists():
        reasons.append("missing_paper_corpus")
    if not state.section_docs or not local_cache_store.section_summary_corpus_path(config).exists():
        reasons.append("missing_section_summary_corpus")
    if not state.paper_rows or not local_cache_store.paper_catalog_path(config).exists():
        reasons.append("missing_paper_catalog")
    if not local_cache_store.paper_catalog_db_path(config).exists():
        reasons.append("missing_paper_catalog_db")

    if config.vector_backend.strip().lower() == "milvus":
        if not vector_index_exists(
            config.vector_backend,
            config.local_cache_dir,
            milvus_uri=config.milvus_uri,
            milvus_token=config.milvus_token,
            milvus_db_name=config.milvus_db_name,
            milvus_collection=config.milvus_papers_collection,
        ):
            reasons.append(f"missing_remote_collection:{config.milvus_papers_collection}")

    return KnowledgeBaseHealth(ok=not reasons, repaired=False, reasons=tuple(dict.fromkeys(reasons)))


def build_readiness_error_message(
    health: KnowledgeBaseHealth,
    *,
    route: QueryRoute,
) -> str:
    if health.ok:
        return ""

    detail_map = {
        "missing_main_chunk_corpus": "缺少主 chunk 语料缓存",
        "missing_parent_corpus": "缺少 parent 语料缓存",
        "missing_paper_corpus": "缺少 paper-level 语料缓存",
        "missing_section_summary_corpus": "缺少章节摘要缓存",
        "missing_paper_catalog": "缺少论文 catalog",
        "missing_paper_catalog_db": "缺少论文 catalog SQLite",
        "missing_reference_corpus": "缺少参考文献语料缓存",
    }
    details = []
    for reason in health.reasons:
        if reason in detail_map:
            details.append(detail_map[reason])
        elif reason.startswith("missing_remote_collection:"):
            collection = reason.split(":", 1)[1]
            details.append(f"缺少远端 Milvus collection: {collection}")
        else:
            details.append(reason)

    scope_text = "参考文献检索" if route.retrieval_scope == "references" else "论文问答"
    detail_text = "；".join(details) if details else "知识库状态不完整"
    return (
        f"{scope_text}所需知识库未就绪：{detail_text}。"
        "请先运行 `python main.py ingest` 完成升级或修复后再问答。"
    )


def ensure_knowledge_base_consistency(
    config: AppConfig,
    embeddings,
    *,
    repair_remote: bool = True,
    upgrade_local_state: bool = True,
) -> KnowledgeBaseHealth:
    """Validate local/remote KB consistency and optionally repair it.

    This is the heavy check used during ingest/cache-hit validation, not the
    normal query path.
    """
    state = local_cache_store.load_local_cache_state(config)
    repaired = False
    reasons: list[str] = []

    if upgrade_local_state:
        upgraded_state, upgraded = ensure_catalog_state(config, state)
        if upgraded:
            state = upgraded_state
            repaired = True
            reasons.append("upgraded_local_catalog_state")

    local_issues = _validate_local_cache_files(config, state)
    reasons.extend(local_issues)
    remote_issues = _validate_remote_indexes(config, state)
    reasons.extend(remote_issues)

    if remote_issues and repair_remote and state.main_chunks:
        _rebuild_remote_from_local_state(config, state, embeddings)
        repaired = True
        reasons.append("rebuilt_remote_indexes")
        remote_issues = _validate_remote_indexes(config, state)
        reasons.extend(f"post_repair:{item}" for item in remote_issues)

    ok = not _validate_local_cache_files(config, state) and not _validate_remote_indexes(
        config, state
    )
    return KnowledgeBaseHealth(ok=ok, repaired=repaired, reasons=tuple(dict.fromkeys(reasons)))


def ensure_catalog_state(
    config: AppConfig,
    state: local_cache_store.LocalCacheState,
) -> tuple[local_cache_store.LocalCacheState, bool]:
    """Upgrade older local cache layouts into the newer paper/catalog layout."""
    if state.paper_docs and state.section_docs and state.paper_rows:
        return state, False
    if not state.parent_docs or not state.block_rows:
        return state, False

    assets = build_paper_assets(
        state.parent_docs,
        state.block_rows,
        state.reference_chunks,
        source_root=config.data_pdf_dir,
        paper_summary_max_chars=config.paper_summary_max_chars,
        section_summary_max_chars=config.section_summary_max_chars,
    )
    upgraded = local_cache_store.LocalCacheState(
        main_chunks=state.main_chunks,
        parent_docs=state.parent_docs,
        paper_docs=assets.paper_docs,
        section_docs=assets.section_docs,
        reference_chunks=state.reference_chunks,
        block_rows=state.block_rows,
        chunk_rows=state.chunk_rows,
        paper_rows=assets.catalog_rows,
        citation_rows=assets.citation_rows,
        reference_keyword_rows=state.reference_keyword_rows,
    )
    local_cache_store.save_local_cache_state(config, upgraded)
    return upgraded, True


def validate_cache_hit_state(
    config: AppConfig,
    cached: dict | None,
    embeddings,
) -> KnowledgeBaseHealth:
    """Verify that a claimed ingest cache hit still reflects real KB state."""
    state = local_cache_store.load_local_cache_state(config)
    reasons: list[str] = []

    expected_main = int((cached or {}).get("main_chunk_count", 0) or 0)
    expected_papers = int((cached or {}).get("paper_doc_count", 0) or 0)
    expected_refs = int((cached or {}).get("reference_chunk_count", 0) or 0)

    if expected_main and len(state.main_chunks) != expected_main:
        reasons.append("main_chunk_count_mismatch")
    if expected_papers and len(state.paper_docs) != expected_papers:
        reasons.append("paper_doc_count_mismatch")
    if expected_refs and len(state.reference_chunks) != expected_refs:
        reasons.append("reference_chunk_count_mismatch")

    health = ensure_knowledge_base_consistency(
        config,
        embeddings,
        repair_remote=True,
        upgrade_local_state=True,
    )
    reasons.extend(health.reasons)
    return KnowledgeBaseHealth(
        ok=health.ok and not any(item.endswith("_mismatch") for item in reasons),
        repaired=health.repaired,
        reasons=tuple(dict.fromkeys(reasons)),
    )


def _validate_local_cache_files(
    config: AppConfig,
    state: local_cache_store.LocalCacheState,
) -> list[str]:
    issues: list[str] = []
    required_paths = [
        (local_cache_store.chunk_corpus_path(config), bool(state.main_chunks)),
        (local_cache_store.parent_corpus_path(config), bool(state.parent_docs)),
        (local_cache_store.paper_corpus_path(config), bool(state.paper_docs)),
        (local_cache_store.section_summary_corpus_path(config), bool(state.section_docs)),
        (local_cache_store.paper_catalog_path(config), bool(state.paper_rows)),
        (local_cache_store.paper_catalog_db_path(config), bool(state.paper_rows)),
        (local_cache_store.citation_graph_path(config), bool(state.citation_rows)),
    ]
    for path, expected in required_paths:
        if expected and not path.exists():
            issues.append(f"missing_local_file:{path.name}")
    return issues


def _validate_remote_indexes(
    config: AppConfig,
    state: local_cache_store.LocalCacheState,
) -> list[str]:
    issues: list[str] = []
    backend = config.vector_backend.strip().lower()
    if backend != "milvus":
        return issues

    expected_collections = [
        (config.milvus_collection, len(state.main_chunks)),
        (config.milvus_papers_collection, len(state.paper_docs)),
    ]
    if (
        config.references_strategy.strip().lower() == "separate_collection"
        and state.reference_chunks
    ):
        expected_collections.append(
            (config.milvus_references_collection, len(state.reference_chunks))
        )

    for collection_name, expected_entities in expected_collections:
        if expected_entities <= 0:
            continue
        if not vector_index_exists(
            config.vector_backend,
            config.local_cache_dir,
            milvus_uri=config.milvus_uri,
            milvus_token=config.milvus_token,
            milvus_db_name=config.milvus_db_name,
            milvus_collection=collection_name,
        ):
            issues.append(f"missing_remote_collection:{collection_name}")
            continue
        entity_count = vector_index_entity_count(
            config.vector_backend,
            config.local_cache_dir,
            milvus_uri=config.milvus_uri,
            milvus_token=config.milvus_token,
            milvus_db_name=config.milvus_db_name,
            milvus_collection=collection_name,
        )
        if entity_count is not None and entity_count <= 0:
            issues.append(f"empty_remote_collection:{collection_name}")
    return issues


def _rebuild_remote_from_local_state(
    config: AppConfig,
    state: local_cache_store.LocalCacheState,
    embeddings,
) -> None:
    build_vector_index(
        backend=config.vector_backend,
        documents=state.main_chunks,
        embeddings=embeddings,
        persist_dir=config.local_cache_dir,
        milvus_uri=config.milvus_uri,
        milvus_token=config.milvus_token,
        milvus_db_name=config.milvus_db_name,
        milvus_collection=config.milvus_collection,
        milvus_papers_collection=config.milvus_papers_collection,
        paper_documents=state.paper_docs,
        reference_documents=state.reference_chunks,
        references_strategy=config.references_strategy,
        milvus_references_collection=config.milvus_references_collection,
        milvus_drop_old=True,
    )
