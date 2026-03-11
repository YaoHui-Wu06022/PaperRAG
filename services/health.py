from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pymilvus import connections, utility

from config import AppConfig
from services import local_cache_store
from retrieval.vector_store import build_milvus_connection_args


@dataclass(frozen=True)
class HealthCheckResult:
    name: str
    ok: bool
    message: str
    fatal: bool = True


def build_startup_health_report(
    config: AppConfig,
    *,
    require_mineru: bool,
    require_llm: bool,
    require_local_cache: bool,
    require_milvus: bool,
) -> list[HealthCheckResult]:
    results = [
        _check_pending_sync_journal(config),
        _check_local_cache_dir(config),
        _check_embedding_config(config),
    ]
    if require_local_cache:
        results.append(_check_required_local_cache_files(config))
    if require_mineru:
        results.append(_check_mineru_config(config))
    if require_llm:
        results.append(_check_llm_config(config))
    results.append(_check_reranker_config(config))
    if require_milvus:
        results.extend(_check_milvus(config))
    return results


def ensure_startup_ready(
    config: AppConfig,
    *,
    require_mineru: bool,
    require_llm: bool,
    require_local_cache: bool,
    require_milvus: bool,
) -> list[HealthCheckResult]:
    results = build_startup_health_report(
        config,
        require_mineru=require_mineru,
        require_llm=require_llm,
        require_local_cache=require_local_cache,
        require_milvus=require_milvus,
    )
    failures = [item for item in results if item.fatal and not item.ok]
    if failures:
        details = "; ".join(f"{item.name}: {item.message}" for item in failures)
        raise RuntimeError(f"Startup health check failed: {details}")
    return results


def _check_pending_sync_journal(config: AppConfig) -> HealthCheckResult:
    pending = local_cache_store.load_sync_journal(config)
    if pending is None:
        return HealthCheckResult("sync_journal", True, "no pending sync journal", False)
    status = str(pending.get("status", "unknown"))
    return HealthCheckResult(
        "sync_journal",
        False,
        f"pending sync journal detected (status={status})",
        True,
    )


def _check_local_cache_dir(config: AppConfig) -> HealthCheckResult:
    if config.local_cache_dir.exists():
        return HealthCheckResult("local_cache_dir", True, str(config.local_cache_dir), False)
    return HealthCheckResult(
        "local_cache_dir",
        False,
        f"missing cache dir: {config.local_cache_dir}",
    )


def _check_required_local_cache_files(config: AppConfig) -> HealthCheckResult:
    required = [
        local_cache_store.chunk_corpus_path(config),
        local_cache_store.parent_corpus_path(config),
        local_cache_store.reference_keyword_index_path(config),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if not missing:
        return HealthCheckResult("local_cache_files", True, "required cache files present")
    return HealthCheckResult(
        "local_cache_files",
        False,
        f"missing cache files: {', '.join(missing)}",
    )


def _check_mineru_config(config: AppConfig) -> HealthCheckResult:
    if config.mineru_api_token:
        return HealthCheckResult("mineru_config", True, "token present")
    return HealthCheckResult("mineru_config", False, "MINERU_API_TOKEN is empty")


def _check_embedding_config(config: AppConfig) -> HealthCheckResult:
    provider = config.embedding_provider.strip().lower()
    if provider != "openai":
        return HealthCheckResult("embedding_config", True, provider, False)
    ok = bool(config.embedding_api_key)
    message = (
        "EMBEDDING_API_KEY present"
        if ok
        else "EMBEDDING_API_KEY is empty for openai-compatible embeddings"
    )
    return HealthCheckResult("embedding_config", ok, message)


def _check_llm_config(config: AppConfig) -> HealthCheckResult:
    ok = bool(config.llm_api_key)
    message = "LLM_API_KEY present" if ok else "LLM_API_KEY is empty"
    return HealthCheckResult("llm_config", ok, message)


def _check_reranker_config(config: AppConfig) -> HealthCheckResult:
    if not config.use_reranker:
        return HealthCheckResult("reranker_config", True, "disabled", False)
    provider = config.reranker_provider.strip().lower()
    if provider != "jina":
        return HealthCheckResult("reranker_config", True, provider, False)
    ok = bool(config.reranker_api_key)
    message = (
        "RERANKER_API_KEY present"
        if ok
        else "RERANKER_API_KEY is empty for jina reranker"
    )
    return HealthCheckResult("reranker_config", ok, message, False)


def _check_milvus(config: AppConfig) -> list[HealthCheckResult]:
    uri = config.milvus_uri.strip()
    if not uri:
        return [HealthCheckResult("milvus_config", False, "MILVUS_URI is empty")]

    if uri.startswith("https://") and not config.milvus_token:
        return [HealthCheckResult("milvus_config", False, "MILVUS_TOKEN is empty for remote Milvus")]

    connection_args: dict[str, Any] = build_milvus_connection_args(
        config.milvus_uri,
        milvus_token=config.milvus_token,
        milvus_db_name=config.milvus_db_name,
    )
    alias = "rag_health_check"
    try:
        connections.connect(alias=alias, **connection_args)
        collections = utility.list_collections(using=alias)
        return [
            HealthCheckResult("milvus_config", True, uri, False),
            HealthCheckResult(
                "milvus_connectivity",
                True,
                f"connected, collections={len(collections)}",
                False,
            ),
        ]
    except Exception as exc:
        return [
            HealthCheckResult("milvus_connectivity", False, str(exc)),
        ]
    finally:
        try:
            connections.disconnect(alias=alias)
        except Exception:
            pass
