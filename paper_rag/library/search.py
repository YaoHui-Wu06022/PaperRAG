from __future__ import annotations

import json
import time

from .catalog import Catalog
from .common import LibraryError, digest, evidence_id
from .contracts import SearchRequest
from .reading import cursor_decode, cursor_encode, filter_sql
from .text import match_expression
from .vectors import Embedder, VectorStore


def search(cat: Catalog, request: SearchRequest, *, store=None, embedder=None):
    started = time.perf_counter()
    scope = digest(
        [
            request.model_dump(exclude={"cursor"}),
            cat.snapshot(),
            cat.get("active_generation"),
        ]
    )
    offset = (
        cursor_decode(request.cursor, scope).get("offset", 0) if request.cursor else 0
    )
    if offset > 2000:
        raise LibraryError("invalid_cursor", "Search cursor exceeds candidate budget")
    budget = min(2000, max(request.candidate_limit, offset + request.limit + 1))
    where, params = filter_sql(request.filters)
    regions = "c.region IN (SELECT value FROM json_each(?))"
    region_arg = json.dumps(request.regions)
    rankings = []
    warnings = []
    timings = {}
    if request.mode != "dense":
        begin = time.perf_counter()
        for query in list(dict.fromkeys([request.query] + request.keywords)):
            expression = match_expression(query)
            if not expression:
                continue
            rows = cat.rows(
                "SELECT c.chunk_id,bm25(chunk_fts,0,2,1.5,1) AS score FROM chunk_fts JOIN chunks c ON c.chunk_id=chunk_fts.chunk_id JOIN documents d ON d.document_id=c.document_id AND d.active_revision=c.revision_id WHERE chunk_fts MATCH ? AND "
                + where
                + " AND "
                + regions
                + " ORDER BY score,c.chunk_id LIMIT ?",
                [expression] + params + [region_arg, budget],
            )
            rankings.append(("lexical", rows))
        timings["lexical_ms"] = (time.perf_counter() - begin) * 1000
    dense_incomplete = False
    if request.mode != "lexical":
        begin = time.perf_counter()
        own_store = store is None
        try:
            gid = cat.get("active_generation")
            generation = (
                cat.one("SELECT * FROM index_generations WHERE generation_id=?", (gid,))
                if gid
                else None
            )
            if not generation:
                raise LibraryError("index_unavailable", "No published dense generation")
            if json.loads(generation["identity"]) != cat.settings.embedding_identity:
                raise LibraryError(
                    "index_identity_mismatch",
                    "Dense model configuration differs from published index",
                )
            explicit = request.filters.model_dump(exclude={"schema_version"})
            scoped = any(v is not None and v != [] for v in explicit.values())
            ids = (
                [
                    r["document_id"]
                    for r in cat.rows(
                        "SELECT d.document_id FROM documents d WHERE " + where, params
                    )
                ]
                if scoped
                else None
            )
            valid = []
            if ids != []:
                store = store or VectorStore(cat.settings)
                embedder = embedder or Embedder(cat)
                vector = embedder.embed([request.query])[0]
                fetch = budget
                while True:
                    hits = store.search(
                        generation["collection_name"],
                        vector,
                        fetch,
                        request.regions,
                        ids,
                    )
                    valid = []
                    for hit in hits:
                        row = cat.one(
                            "SELECT c.chunk_id FROM chunks c JOIN documents d ON d.document_id=c.document_id AND d.active_revision=c.revision_id WHERE c.chunk_id=? AND c.revision_id=? AND c.content_hash=? AND "
                            + where
                            + " AND "
                            + regions,
                            [hit["chunk_id"], hit["revision_id"], hit["content_hash"]]
                            + params
                            + [region_arg],
                        )
                        if row:
                            valid.append(hit)
                    if len(valid) >= budget or len(hits) < fetch:
                        break
                    if fetch >= 2000:
                        dense_incomplete = True
                        warnings.append(
                            {
                                "code": "incomplete",
                                "message": "Dense candidate budget exhausted after stale-version filtering",
                            }
                        )
                        break
                    fetch = min(2000, fetch * 2)
                missing = cat.one(
                    "SELECT count(*) n FROM chunks c JOIN documents d ON d.active_revision=c.revision_id WHERE "
                    + where
                    + " AND "
                    + regions
                    + " AND NOT EXISTS(SELECT 1 FROM index_rows i WHERE i.generation_id=? AND i.chunk_id=c.chunk_id AND i.content_hash=c.content_hash)",
                    params + [region_arg, gid],
                )["n"]
                if missing:
                    dense_incomplete = True
                    warnings.append(
                        {
                            "code": "dense_coverage_gap",
                            "message": f"{missing} visible chunks are not indexed in this generation",
                        }
                    )
            rankings.append(("dense", valid[:budget]))
        except Exception as exc:
            warnings.append(
                {
                    "code": getattr(exc, "code", "dependency_unavailable"),
                    "message": (
                        str(exc)
                        if isinstance(exc, LibraryError)
                        else type(exc).__name__
                    ),
                }
            )
            dense_incomplete = True
        finally:
            if own_store and store is not None:
                store.close()
        timings["dense_ms"] = (time.perf_counter() - begin) * 1000
    # Fuse lexical variants once, then lexical vs dense. Variants must not outvote dense simply by count.
    lexical_scores = {}
    for source, rows in rankings:
        if source == "lexical":
            for rank, row in enumerate(rows, 1):
                lexical_scores[row["chunk_id"]] = lexical_scores.get(
                    row["chunk_id"], 0
                ) + 1 / (60 + rank)
    streams = [
        (
            "lexical",
            [
                {"chunk_id": key}
                for key in sorted(
                    lexical_scores, key=lambda key: (-lexical_scores[key], key)
                )[:budget]
            ],
        )
    ]
    streams.extend((source, rows) for source, rows in rankings if source == "dense")
    scores, sources = {}, {}
    for source, rows in streams:
        for rank, row in enumerate(rows, 1):
            key = row["chunk_id"]
            scores[key] = scores.get(key, 0) + 1 / (60 + rank)
            sources.setdefault(key, []).append(source)
    ordered = sorted(scores, key=lambda key: (-scores[key], key))
    items = []
    seen = set()
    for key in ordered:
        row = cat.one(
            "SELECT c.*,d.metadata,d.source_kind FROM chunks c JOIN documents d ON d.document_id=c.document_id WHERE c.chunk_id=?",
            (key,),
        )
        identity = (row["document_id"], row["revision_id"], row["spans"])
        if identity in seen:
            continue
        seen.add(identity)
        spans = json.loads(row["spans"])
        refs = [
            evidence_id(
                row["document_id"],
                row["revision_id"],
                span["block_id"],
                span["start"],
                span["end"],
            )
            for span in spans
        ]
        items.append(
            {
                "chunk_id": key,
                "document_id": row["document_id"],
                "revision_id": row["revision_id"],
                "title": json.loads(row["metadata"])["title"],
                "source_kind": row["source_kind"],
                "region": row["region"],
                "section_id": row["section_id"],
                "preview": row["text"][:600],
                "evidence_ids": refs,
                "score": scores[key],
                "sources": sources[key],
            }
        )
    stop = offset + request.limit
    timings["total_ms"] = (time.perf_counter() - started) * 1000
    return {
        "items": items[offset:stop],
        "warnings": warnings,
        "degraded": dense_incomplete,
        "next_cursor": (
            cursor_encode({"scope": scope, "offset": stop})
            if stop < len(items)
            else None
        ),
        "candidate_count": len(items),
        "timings": timings,
        "corpus_version": cat.snapshot(),
    }
