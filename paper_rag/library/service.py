from __future__ import annotations

import json
import sys
from pathlib import Path

from .catalog import Catalog
from .common import LibraryError
from .contracts import (
    CitationRequest,
    DoctorRequest,
    IndexRequest,
    IngestRequest,
    PapersRequest,
    ReadRequest,
    SearchRequest,
    WikiApply,
    WikiRequest,
)
from .ingestion import (
    import_document,
    migrate_legacy,
    publish_revision,
    resolve_citations,
)
from .reading import citations, papers, read
from .search import search
from .vectors import index_library, rollback
from .publications import rollback_publication
from .maintenance import inspect as inspect_index, archive as archive_index, restore as restore_index, retire as retire_index
from .wiki import apply as apply_wiki, lint, prepare, read_page, search_pages, recover


def response(status: str, data=None, *, warnings=None):
    warnings = warnings or (data.get("warnings", []) if isinstance(data, dict) else [])
    if status == "ok" and isinstance(data, dict):
        if data.get("degraded"):
            status = "degraded"
        elif data.get("ambiguous"):
            status = "ambiguous"
        elif "items" in data and not data["items"]:
            status = "empty"
    return {"schema_version": 1, "status": status, "data": data, "warnings": warnings}


def doctor(settings, request: DoctorRequest):
    checks = {
        "python": sys.version.split()[0],
        "database": settings.database.exists(),
        "milvus_configured": bool(settings.env.get("MILVUS_URI")),
        "embedding_configured": bool(settings.env.get("EMBEDDING_API_KEY")),
        "mineru_configured": bool(settings.env.get("MINERU_API_KEY")),
        "generative_llm_calls": False,
    }
    if not settings.database.exists():
        return response(
            "not_initialized",
            checks,
            warnings=[
                {
                    "code": "not_initialized",
                    "message": "Run ingest to initialize the catalog",
                }
            ],
        )
    with Catalog(settings) as cat:
        checks |= {
            "catalog_schema": cat.db.execute("PRAGMA user_version").fetchone()[0],
            "corpus_version": cat.snapshot(),
            "active_generation": cat.get("active_generation"),
            "documents": cat.one("SELECT count(*) n FROM documents")["n"],
            "active_revisions": cat.one(
                "SELECT count(*) n FROM revisions WHERE status='active'"
            )["n"],
            "pending_jobs": cat.one(
                "SELECT count(*) n FROM jobs WHERE state='pending'"
            )["n"],
            "wiki_pages": cat.one("SELECT count(*) n FROM wiki_pages")["n"],
        }
        generation = (
            cat.one(
                "SELECT * FROM index_generations WHERE generation_id=?",
                (cat.get("active_generation"),),
            )
            if cat.get("active_generation")
            else None
        )
        if generation and generation["identity"] != json.dumps(
            settings.embedding_identity,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ):
            checks["index_identity_mismatch"] = True
        if request.remote and generation:
            from .vectors import VectorStore

            store = None
            try:
                store = VectorStore(settings)
                verified = 0
                cursor = cat.db.execute(
                    "SELECT c.* FROM chunks c JOIN documents d ON d.active_revision=c.revision_id WHERE d.withdrawn=0"
                )
                while rows := cursor.fetchmany(100):
                    store.verify(
                        generation["collection_name"], [dict(row) for row in rows]
                    )
                    verified += len(rows)
                checks["remote_verified_chunks"] = verified
            except Exception as exc:
                return response(
                    "degraded",
                    checks,
                    warnings=[
                        {
                            "code": getattr(exc, "code", "dependency_unavailable"),
                            "message": type(exc).__name__,
                        }
                    ],
                )
            finally:
                if store is not None:
                    store.close()
    return response("ok", checks)


def ingest(settings, request: IngestRequest):
    with Catalog(settings, writable=True) as cat:
        results = []
        if request.migrate_legacy:
            results.append(migrate_legacy(cat))
        for source in request.sources:
            results.append(
                import_document(
                    cat,
                    Path(source.path),
                    metadata=source.metadata,
                    mineru_output=(
                        Path(source.mineru_output) if source.mineru_output else None
                    ),
                    document_id=source.document_id,
                )
            )
        if not results:
            raise LibraryError(
                "invalid_request", "sources or migrate_legacy is required"
            )
        index_result = None
        if request.index:
            index_result = index_library(cat, lexical_only=request.lexical_only)
        return response("ok", {"sources": results, "index": index_result})


def index(settings, request: IndexRequest):
    with Catalog(settings, writable=True) as cat:
        if request.rollback_publication:
            return response("ok", rollback_publication(cat, request.rollback_publication))
        if request.rollback_generation:
            return response("ok", rollback(cat, request.rollback_generation))
        if request.collection and request.archive_digest and request.target:
            return response("ok", restore_index(cat, request.collection, request.archive_digest, target=request.target, dry_run=request.dry_run))
        if request.collection and request.archive_digest and request.force:
            return response("ok", retire_index(cat, request.collection, request.archive_digest, dry_run=request.dry_run))
        if request.collection and request.force:
            return response("ok", archive_index(cat, request.collection, force=request.force))
        return response(
            "ok",
            index_library(
                cat, rebuild=request.rebuild, lexical_only=request.lexical_only
            ),
        )


def run(settings, command: str, request):
    if command == "doctor":
        return doctor(settings, request)
    if command == "ingest":
        return ingest(settings, request)
    if command == "index":
        return index(settings, request)
    if command == "index_inspect":
        with Catalog(settings) as cat:
            return response("ok", inspect_index(cat))
    if command in {"index_archive", "index_restore", "index_retire"}:
        with Catalog(settings, writable=True) as cat:
            if not request.collection:
                raise LibraryError("invalid_request", "collection is required")
            if command == "index_archive":
                return response("ok", archive_index(cat, request.collection, force=request.force))
            if not request.archive_digest:
                raise LibraryError("invalid_request", "archive_digest is required")
            if command == "index_restore":
                return response("ok", restore_index(cat, request.collection, request.archive_digest, target=request.target, dry_run=request.dry_run))
            return response("ok", retire_index(cat, request.collection, request.archive_digest, dry_run=request.dry_run))
    writable = command in {"wiki_apply", "wiki_recover"}
    with Catalog(settings, writable=writable) as cat:
        if command == "eval":
            from .evaluation import evaluate

            return response("ok", evaluate(cat, request))
        if command == "papers_find":
            return response("ok", papers(cat, request))
        if command == "papers_count":
            return response("ok", papers(cat, request, count=True))
        if command == "papers_get":
            return response("ok", papers(cat, request, get=True))
        if command == "search":
            return response("ok", search(cat, request))
        if command == "read":
            return response("ok", read(cat, request))
        if command == "citations":
            return response("ok", citations(cat, request))
        if command == "wiki_search":
            return response("ok", search_pages(cat, request))
        if command == "wiki_read":
            return response("ok", read_page(cat, request))
        if command == "wiki_prepare":
            return response("ok", prepare(cat, request))
        if command == "wiki_lint":
            return response("ok", lint(cat))
        if command == "wiki_apply":
            return response("ok", apply_wiki(cat, request))
        if command == "wiki_recover":
            return response("ok", recover(cat, request))
    raise LibraryError("unknown_command", command)
