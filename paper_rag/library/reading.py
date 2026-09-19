from __future__ import annotations

import base64
import json
from pathlib import Path

from .catalog import Catalog
from .common import LibraryError, digest, dumps, evidence_id, parse_evidence
from .contracts import Filters, PapersRequest, ReadRequest, CitationRequest


def filter_sql(filters: Filters, alias: str = "d"):
    clauses, values = [
        f"{alias}.withdrawn=0",
        f"{alias}.active_revision IS NOT NULL",
    ], []
    if filters.document_ids is not None:
        clauses.append(f"{alias}.document_id IN (SELECT value FROM json_each(?))")
        values.append(dumps(filters.document_ids))
    if filters.source_kind:
        clauses.append(f"{alias}.source_kind=?")
        values.append(filters.source_kind)
    for field, value in (("title", filters.title), ("venue", filters.venue)):
        if value:
            clauses.append(
                f"instr(lower(json_extract({alias}.metadata,'$.{field}')),lower(?))>0"
            )
            values.append(value)
    if filters.author:
        clauses.append(
            f"EXISTS(SELECT 1 FROM json_each({alias}.metadata,'$.authors') WHERE instr(lower(value),lower(?))>0)"
        )
        values.append(filters.author)
    for op, value in ((">=", filters.year_min), ("<=", filters.year_max)):
        if value is not None:
            clauses.append(
                f"COALESCE(json_extract({alias}.metadata,'$.year.publish_year'),json_extract({alias}.metadata,'$.year.preprint_year')) {op} ?"
            )
            values.append(value)
    for tag in filters.tags:
        clauses.append(
            f"EXISTS(SELECT 1 FROM json_each({alias}.metadata,'$.tags') WHERE lower(value)=lower(?))"
        )
        values.append(tag)
    return " AND ".join(clauses), values


def papers(
    cat: Catalog, request: PapersRequest, *, count: bool = False, get: bool = False
):
    if get and not (request.document_id or request.query):
        raise LibraryError(
            "invalid_request", "papers get requires document_id or query"
        )
    where, values = filter_sql(request.filters)
    if request.document_id:
        where += " AND d.document_id=?"
        values.append(request.document_id)
    if request.query:
        where += " AND (instr(lower(json_extract(d.metadata,'$.title')),lower(?))>0 OR EXISTS(SELECT 1 FROM json_each(d.metadata,'$.aliases') WHERE lower(value)=lower(?)))"
        values.extend([request.query, request.query])
    total = cat.one("SELECT count(*) AS n FROM documents d WHERE " + where, values)["n"]
    if count:
        return {
            "count": total,
            "scope": "local_library",
            "corpus_version": cat.snapshot(),
        }
    rows = cat.rows(
        "SELECT d.* FROM documents d WHERE "
        + where
        + " ORDER BY json_extract(d.metadata,'$.title'),d.document_id LIMIT ? OFFSET ?",
        values + [request.limit, request.offset],
    )
    results = [
        {
            "document_id": r["document_id"],
            "revision_id": r["active_revision"],
            "source_kind": r["source_kind"],
            "metadata": json.loads(r["metadata"]),
        }
        for r in rows
    ]
    return {
        "items": results,
        "total": total,
        "ambiguous": total > 1 and bool(request.query or get),
        "next_offset": (
            request.offset + len(results)
            if request.offset + len(results) < total
            else None
        ),
    }


def cursor_encode(payload: dict) -> str:
    return base64.urlsafe_b64encode(dumps(payload).encode()).decode()


def cursor_decode(cursor: str, scope: str) -> dict:
    try:
        value = json.loads(base64.urlsafe_b64decode(cursor))
        if value["scope"] != scope:
            raise ValueError
        return value
    except (ValueError, KeyError, TypeError):
        raise LibraryError(
            "invalid_cursor", "Cursor belongs to a different request or library version"
        ) from None


def get_evidence(cat: Catalog, value: str) -> dict:
    doc, rev, block, start, end = parse_evidence(value)
    row = cat.one(
        "SELECT b.*,r.document_id,r.source_path,r.assets_path,d.active_revision,d.withdrawn,d.source_kind,d.metadata "
        "FROM blocks b JOIN revisions r ON r.revision_id=b.revision_id JOIN documents d ON d.document_id=r.document_id "
        "WHERE b.revision_id=? AND b.block_id=? AND r.document_id=?",
        (rev, block, doc),
    )
    if not row or end > len(row["text"]):
        raise LibraryError(
            "not_found", "Evidence does not resolve to an exact stored source span"
        )
    payload = json.loads(row["payload"])
    line_start = payload.get("line_start")
    line_end = payload.get("line_end")
    if line_start is not None:
        line_start += row["text"][:start].count("\n")
        line_end = line_start + row["text"][start:end].rstrip("\n").count("\n")
    section = cat.one(
        "SELECT payload FROM sections WHERE revision_id=? AND section_id=?",
        (rev, row["section_id"]),
    )
    source_asset = payload.get("source_path")
    asset = None
    if source_asset and row["assets_path"]:
        root = Path(row["assets_path"])
        candidate = (root / source_asset).resolve()
        if candidate.is_relative_to(root.resolve()) and candidate.is_file():
            asset = str(candidate)
    return {
        "evidence_id": value,
        "document_id": doc,
        "revision_id": rev,
        "block_id": block,
        "start": start,
        "end": end,
        "text": row["text"][start:end],
        "title": json.loads(row["metadata"])["title"],
        "source_kind": row["source_kind"],
        "region": row["region"],
        "kind": row["kind"],
        "section_id": row["section_id"],
        "page": payload.get("page"),
        "bbox": payload.get("bbox"),
        "line_start": line_start,
        "line_end": line_end,
        "heading_path": json.loads(section["payload"]).get("path") if section else None,
        "source_path": row["source_path"],
        "asset_path": asset,
        "table_html": payload.get("html"),
        "historical": rev != row["active_revision"],
        "withdrawn": bool(row["withdrawn"]),
    }


def read(cat: Catalog, request: ReadRequest) -> dict:
    if request.evidence_id:
        ev = get_evidence(cat, request.evidence_id)
        doc, rev, block, start, end = parse_evidence(request.evidence_id)
    else:
        if not request.document_id:
            raise LibraryError(
                "invalid_request", "document_id or evidence_id is required"
            )
        doc = request.document_id
        row = cat.one(
            "SELECT active_revision FROM documents WHERE document_id=?", (doc,)
        )
        if not row:
            raise LibraryError("not_found", "Document not found")
        rev = request.revision_id or row["active_revision"]
        if not cat.one(
            "SELECT 1 FROM revisions WHERE document_id=? AND revision_id=?", (doc, rev)
        ):
            raise LibraryError("not_found", "Document revision not found")
        block, start, end = request.block_id, 0, None
    context_ids = None
    if request.context_before or request.context_after:
        if not block:
            raise LibraryError(
                "invalid_request", "Context requires a block_id or evidence_id"
            )
        anchor = cat.one(
            "SELECT ordinal FROM blocks WHERE revision_id=? AND block_id=?",
            (rev, block),
        )
        if not anchor:
            raise LibraryError("not_found", "Block not found")
        before = cat.rows(
            "SELECT block_id FROM blocks WHERE revision_id=? AND ordinal<? ORDER BY ordinal DESC LIMIT ?",
            (rev, anchor["ordinal"], request.context_before),
        )
        after = cat.rows(
            "SELECT block_id FROM blocks WHERE revision_id=? AND ordinal>? ORDER BY ordinal LIMIT ?",
            (rev, anchor["ordinal"], request.context_after),
        )
        context_ids = (
            [b["block_id"] for b in before] + [block] + [b["block_id"] for b in after]
        )
        block, start, end = None, 0, None
    scope = digest(
        [
            doc,
            rev,
            request.section_id,
            block,
            start,
            end,
            request.include_children,
            request.list_sections,
            context_ids,
        ]
    )
    position = (
        cursor_decode(request.cursor, scope)
        if request.cursor
        else {"ordinal": -1, "character": start}
    )
    if request.list_sections:
        rows = cat.rows(
            "SELECT payload FROM sections WHERE revision_id=? ORDER BY ordinal", (rev,)
        )
        offset = position.get("offset", 0)
        items, used = [], 0
        for row in rows[offset:]:
            obj = json.loads(row["payload"])
            size = len(dumps(obj))
            if items and used + size > request.max_chars:
                break
            items.append(obj)
            used += size
        next_offset = offset + len(items)
        return {
            "sections": items,
            "document_id": doc,
            "revision_id": rev,
            "next_cursor": (
                cursor_encode({"scope": scope, "offset": next_offset})
                if next_offset < len(rows)
                else None
            ),
        }
    where, values = "b.revision_id=?", [rev]
    if context_ids:
        where += " AND b.block_id IN (SELECT value FROM json_each(?))"
        values.append(dumps(context_ids))
    if block:
        where += " AND b.block_id=?"
        values.append(block)
    if request.section_id:
        if not cat.one(
            "SELECT 1 FROM sections WHERE revision_id=? AND section_id=?",
            (rev, request.section_id),
        ):
            raise LibraryError("not_found", "Section not found")
        ids = [request.section_id]
        if request.include_children:
            ids = [
                r["section_id"]
                for r in cat.rows(
                    "WITH RECURSIVE tree(section_id) AS (SELECT section_id FROM sections WHERE revision_id=? AND section_id=? UNION ALL SELECT s.section_id FROM sections s JOIN tree t ON s.parent_id=t.section_id WHERE s.revision_id=?) SELECT section_id FROM tree",
                    (rev, request.section_id, rev),
                )
            ]
        where += " AND b.section_id IN (SELECT value FROM json_each(?))"
        values.append(dumps(ids))
    where += " AND b.ordinal>=?"
    values.append(position["ordinal"])
    results, remaining, next_cursor = [], request.max_chars, None
    for row in cat.db.execute(
        "SELECT b.* FROM blocks b WHERE " + where + " ORDER BY b.ordinal,b.block_id",
        values,
    ):
        begin = (
            position["character"]
            if row["ordinal"] == position["ordinal"] or block and not request.cursor
            else 0
        )
        stop = min(end if end is not None else len(row["text"]), len(row["text"]))
        if begin >= stop:
            continue
        if remaining == 0:
            next_cursor = cursor_encode(
                {"scope": scope, "ordinal": row["ordinal"], "character": begin}
            )
            break
        finish = min(stop, begin + remaining)
        item = get_evidence(cat, evidence_id(doc, rev, row["block_id"], begin, finish))
        # HTML can dwarf a text budget; assets remain accessible through the source path.
        item.pop("table_html", None)
        results.append(item)
        remaining -= finish - begin
        if finish < stop:
            next_cursor = cursor_encode(
                {"scope": scope, "ordinal": row["ordinal"], "character": finish}
            )
            break
    return {
        "document_id": doc,
        "revision_id": rev,
        "fragments": results,
        "next_cursor": next_cursor,
    }


def citations(cat: Catalog, request: CitationRequest) -> dict:
    if not cat.one(
        "SELECT 1 FROM documents WHERE document_id=?", (request.document_id,)
    ):
        raise LibraryError("not_found", "Document not found")
    frontier, visited, edges = [request.document_id], set(), {}
    for hop in range(request.depth):
        if not frontier:
            break
        field = (
            "d.document_id"
            if request.direction == "outgoing"
            else "c.target_document_id"
        )
        rows = cat.rows(
            "SELECT c.*,d.document_id AS source_document_id FROM citations c JOIN documents d ON d.active_revision=c.revision_id WHERE d.withdrawn=0 AND "
            + field
            + " IN (SELECT value FROM json_each(?))",
            (dumps(frontier),),
        )
        visited.update(frontier)
        frontier = []
        for row in rows:
            key = (row["revision_id"], row["reference_id"])
            if key in edges:
                continue
            payload = json.loads(row.pop("payload"))
            row["page"] = payload.get("page")
            row["hop"] = hop + 1
            block = cat.one(
                "SELECT text FROM blocks WHERE revision_id=? AND block_id=?",
                (row["revision_id"], payload.get("source_block_id")),
            )
            row["evidence_id"] = (
                evidence_id(
                    row["source_document_id"],
                    row["revision_id"],
                    payload["source_block_id"],
                    0,
                    len(block["text"]),
                )
                if block and block["text"]
                else None
            )
            edges[key] = row
            target = (
                row["target_document_id"]
                if request.direction == "outgoing"
                else row["source_document_id"]
            )
            if target and target not in visited:
                frontier.append(target)
    items = sorted(
        edges.values(),
        key=lambda x: (x["hop"], x["source_document_id"], x["reference_id"]),
    )
    stop = request.offset + request.limit
    return {
        "scope": "local_library",
        "items": items[request.offset : stop],
        "total": len(items),
        "next_offset": stop if stop < len(items) else None,
    }
