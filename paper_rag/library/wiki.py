"""Mechanical wiki maintenance. The host agent supplies all authored content."""

from __future__ import annotations

import json
import re

from .catalog import Catalog
from .common import (
    LibraryError,
    atomic_text,
    digest,
    dumps,
    file_hash,
    now,
    parse_evidence,
)
from .contracts import WikiApply, WikiRequest
from .reading import get_evidence
from .text import lexical, match_expression

EVIDENCE_PATTERN = r"ev:[a-zA-Z0-9_-]+:[a-zA-Z0-9_-]+:[a-zA-Z0-9_-]+:\d+:\d+"


def page_path(cat: Catalog, page_id: str):
    return cat.settings.home / "wiki" / "pages" / (page_id + ".md")


def render(request: WikiApply, revision: int) -> str:
    # JSON values are valid YAML scalars; no title can break frontmatter.
    front = {
        "page_id": request.page_id,
        "kind": request.kind,
        "title": request.title,
        "revision": revision,
        "evidence_ids": request.evidence_ids,
        "links": request.links,
    }
    return (
        "---\n"
        + "\n".join(k + ": " + dumps(v) for k, v in front.items())
        + "\n---\n\n"
        + request.body.rstrip()
        + "\n"
    )


def stale_dependents(cat: Catalog, page_id: str):
    dependents = cat.rows(
        "WITH RECURSIVE affected(page_id) AS (SELECT page_id FROM wiki_links WHERE target_id=? UNION SELECT w.page_id FROM wiki_links w JOIN affected a ON w.target_id=a.page_id) SELECT page_id FROM affected",
        (page_id,),
    )
    for row in dependents:
        cat.db.execute(
            "UPDATE wiki_pages SET stale=1 WHERE page_id=?", (row["page_id"],)
        )
        cat.job(
            "refresh_" + row["page_id"],
            "wiki_refresh",
            None,
            None,
            "pending",
            {"changed_page": page_id},
        )


def apply(cat: Catalog, request: WikiApply):
    if (cat.settings.home / "wiki" / "pending" / (request.page_id + ".json")).exists():
        raise LibraryError(
            "interrupted_publication",
            "Run wiki recover before publishing this page again",
        )
    declared = set(request.evidence_ids)
    cited = set(re.findall(EVIDENCE_PATTERN, request.body))
    if declared != cited:
        raise LibraryError(
            "invalid_citation", "Body citations and evidence_ids must match exactly"
        )
    if set(request.quotes) - declared:
        raise LibraryError("invalid_citation", "Quote references must be declared")
    linked = set(re.findall(r"\[\[([a-z0-9_-]+)\]\]", request.body))
    if linked != set(request.links) or request.page_id in linked:
        raise LibraryError(
            "invalid_links",
            "Declare every [[page_id]] link; self links are not allowed",
        )
    evidence = [get_evidence(cat, value) for value in declared]
    for ev in evidence:
        if ev["withdrawn"] or ev["historical"]:
            raise LibraryError(
                "stale_evidence",
                "Refresh to active, non-withdrawn source evidence before publishing",
            )
        quote = request.quotes.get(ev["evidence_id"])
        if quote is not None and (not quote or quote not in ev["text"]):
            raise LibraryError(
                "quote_mismatch",
                "A supplied verbatim quote is absent from its source span",
            )
    path = page_path(cat, request.page_id)
    previous = path.read_text(encoding="utf-8") if path.exists() else None
    intent = cat.settings.home / "wiki" / "pending" / (request.page_id + ".json")
    written = False
    try:
        with cat.transaction():
            old = cat.one(
                "SELECT * FROM wiki_pages WHERE page_id=?", (request.page_id,)
            )
            current = old["current_revision"] if old else 0
            if current != request.expected_revision:
                raise LibraryError(
                    "revision_conflict", "Wiki revision changed; read and rebase first"
                )
            actual_hash = file_hash(path) if path.exists() else None
            expected_hash = request.expected_file_hash or (
                old["file_hash"] if old else None
            )
            if actual_hash != expected_hash:
                raise LibraryError(
                    "manual_edit_conflict",
                    "Markdown changed on disk; preserve edits and supply expected_file_hash",
                )
            # Check again under the publication lock: a concurrent source publish must not race validation.
            for ev in evidence:
                doc = cat.one(
                    "SELECT active_revision,withdrawn FROM documents WHERE document_id=?",
                    (ev["document_id"],),
                )
                if doc["active_revision"] != ev["revision_id"] or doc["withdrawn"]:
                    raise LibraryError(
                        "stale_evidence", "Source changed while preparing this page"
                    )
            links = []
            for target in request.links:
                row = cat.one(
                    "SELECT current_revision,stale FROM wiki_pages WHERE page_id=?",
                    (target,),
                )
                if not row:
                    raise LibraryError(
                        "broken_link", "Linked wiki page does not exist: " + target
                    )
                links.append((request.page_id, target, row["current_revision"]))
            revision = current + 1
            text = render(request, revision)
            snapshot = (
                cat.settings.home
                / "wiki"
                / "history"
                / request.page_id
                / f"{revision:06d}.md"
            )
            atomic_text(snapshot, text)
            # This intent permits deterministic recovery of a process interruption during publication.
            intent = (
                cat.settings.home / "wiki" / "pending" / (request.page_id + ".json")
            )
            atomic_text(
                intent,
                dumps(
                    {
                        "page_id": request.page_id,
                        "revision": revision,
                        "before": previous,
                        "after_hash": digest(text),
                    }
                ),
            )
            atomic_text(path, text)
            written = True
            cat.db.execute(
                "INSERT INTO wiki_pages VALUES(?,?,?,?,?,0) ON CONFLICT(page_id) DO UPDATE SET kind=excluded.kind,title=excluded.title,current_revision=excluded.current_revision,file_hash=excluded.file_hash,stale=0",
                (request.page_id, request.kind, request.title, revision, digest(text)),
            )
            cat.db.execute(
                "INSERT INTO wiki_revisions VALUES(?,?,?,?,?,?)",
                (
                    request.page_id,
                    revision,
                    request.body,
                    dumps(request.evidence_ids),
                    dumps(request.links),
                    now(),
                ),
            )
            cat.db.execute(
                "DELETE FROM wiki_dependencies WHERE page_id=?", (request.page_id,)
            )
            cat.db.executemany(
                "INSERT OR IGNORE INTO wiki_dependencies VALUES(?,?,?)",
                [
                    (request.page_id, ev["document_id"], ev["revision_id"])
                    for ev in evidence
                ],
            )
            cat.db.execute("DELETE FROM wiki_links WHERE page_id=?", (request.page_id,))
            cat.db.executemany("INSERT INTO wiki_links VALUES(?,?,?)", links)
            cat.db.execute("DELETE FROM wiki_fts WHERE page_id=?", (request.page_id,))
            cat.db.execute(
                "INSERT INTO wiki_fts VALUES(?,?,?)",
                (request.page_id, lexical(request.title), lexical(request.body)),
            )
            stale_dependents(cat, request.page_id)
            cat.db.execute(
                "UPDATE jobs SET state='done',updated_at=? WHERE job_id=?",
                (now(), "refresh_" + request.page_id),
            )
            if request.kind == "paper":
                for ev in evidence:
                    cat.db.execute(
                        "UPDATE jobs SET state='done',updated_at=? WHERE job_id=?",
                        (now(), "curate_" + ev["revision_id"]),
                    )
            cat.log("wiki_publish", {"page_id": request.page_id, "revision": revision})
    except BaseException:
        if written:
            if previous is None:
                path.unlink(missing_ok=True)
            else:
                atomic_text(path, previous)
        intent.unlink(missing_ok=True)
        raise
    intent.unlink(missing_ok=True)
    refresh_index(cat)
    return {
        "page_id": request.page_id,
        "revision": revision,
        "path": str(path),
        "mechanical_validation": "passed",
        "semantic_validation": "host_agent_responsibility",
    }


def refresh_index(cat: Catalog):
    lines = [
        "# Research Wiki",
        "",
        "Generated catalog; page contents are authored by the host agent.",
        "",
    ]
    for page in cat.rows("SELECT * FROM wiki_pages ORDER BY kind,title,page_id"):
        lines.append(
            f"- [{page['title']}](pages/{page['page_id']}.md) — {page['kind']}"
            + (" (needs review)" if page["stale"] else "")
        )
    atomic_text(cat.settings.home / "wiki" / "index.md", "\n".join(lines) + "\n")


def read_page(cat: Catalog, request: WikiRequest):
    page = cat.one("SELECT * FROM wiki_pages WHERE page_id=?", (request.page_id,))
    if not page:
        raise LibraryError("not_found", "Wiki page not found")
    revision = request.revision or page["current_revision"]
    row = cat.one(
        "SELECT * FROM wiki_revisions WHERE page_id=? AND revision=?",
        (request.page_id, revision),
    )
    if not row:
        raise LibraryError("not_found", "Wiki revision not found")
    path = page_path(cat, request.page_id)
    edited = revision == page["current_revision"] and (
        not path.exists() or file_hash(path) != page["file_hash"]
    )
    body = path.read_text(encoding="utf-8") if edited and path.exists() else row["body"]
    stop = request.offset + request.max_chars
    deps = []
    for doc, rev in sorted(
        {parse_evidence(value)[:2] for value in json.loads(row["evidence"])}
    ):
        source = cat.one(
            "SELECT active_revision,withdrawn FROM documents WHERE document_id=?",
            (doc,),
        )
        deps.append({"document_id": doc, "revision_id": rev, **source})
    stale = bool(page["stale"]) or any(
        d["revision_id"] != d["active_revision"] or d["withdrawn"] for d in deps
    )
    return {
        "page_id": request.page_id,
        "title": page["title"],
        "revision": revision,
        "stale": stale,
        "manual_edits": edited,
        "file_hash": file_hash(path) if path.exists() else None,
        "body": body[request.offset : stop],
        "evidence_ids": json.loads(row["evidence"]),
        "dependencies": deps,
        "next_offset": stop if stop < len(body) else None,
    }


def search_pages(cat: Catalog, request: WikiRequest):
    expression = match_expression(request.query)
    if expression:
        rows = cat.rows(
            "SELECT w.* FROM wiki_fts JOIN wiki_pages w ON w.page_id=wiki_fts.page_id WHERE wiki_fts MATCH ? ORDER BY bm25(wiki_fts),w.page_id LIMIT ? OFFSET ?",
            (expression, request.limit + 1, request.offset),
        )
    else:
        rows = cat.rows(
            "SELECT * FROM wiki_pages ORDER BY title,page_id LIMIT ? OFFSET ?",
            (request.limit + 1, request.offset),
        )
    return {
        "items": rows[: request.limit],
        "next_offset": (
            request.offset + request.limit if len(rows) > request.limit else None
        ),
        "source_kind": "generated_wiki",
    }


def prepare(cat: Catalog, request: WikiRequest):
    where = "state='pending' AND kind IN ('curate','wiki_refresh')"
    params = []
    if request.document_id:
        where += " AND document_id=?"
        params.append(request.document_id)
    jobs = cat.rows(
        "SELECT * FROM jobs WHERE "
        + where
        + " ORDER BY updated_at,job_id LIMIT ? OFFSET ?",
        params + [request.limit + 1, request.offset],
    )
    result = []
    for job in jobs[: request.limit]:
        doc = cat.one(
            "SELECT document_id,active_revision,metadata FROM documents WHERE document_id=?",
            (job["document_id"],),
        )
        result.append(
            {
                "job": job,
                "document": doc,
                "instruction": "Use papers/read/search to inspect primary evidence; publish authored content with wiki apply",
            }
        )
    return {
        "tasks": result,
        "next_offset": (
            request.offset + request.limit if len(jobs) > request.limit else None
        ),
    }


def lint(cat: Catalog):
    issues = []
    for page in cat.rows("SELECT * FROM wiki_pages"):
        path = page_path(cat, page["page_id"])
        if not path.exists() or file_hash(path) != page["file_hash"]:
            issues.append(
                {"page_id": page["page_id"], "code": "manual_edit_or_missing_file"}
            )
        row = cat.one(
            "SELECT * FROM wiki_revisions WHERE page_id=? AND revision=?",
            (page["page_id"], page["current_revision"]),
        )
        for value in json.loads(row["evidence"]):
            try:
                ev = get_evidence(cat, value)
                if ev["withdrawn"] or ev["historical"]:
                    issues.append(
                        {
                            "page_id": page["page_id"],
                            "code": "stale_evidence",
                            "evidence_id": value,
                        }
                    )
            except LibraryError:
                issues.append(
                    {
                        "page_id": page["page_id"],
                        "code": "invalid_citation",
                        "evidence_id": value,
                    }
                )
        if page["stale"]:
            issues.append({"page_id": page["page_id"], "code": "needs_review"})
    for row in cat.rows(
        "SELECT w.page_id,w.target_id FROM wiki_links w JOIN wiki_pages p ON p.page_id=w.target_id WHERE w.target_revision!=p.current_revision OR p.stale=1"
    ):
        issues.append(row | {"code": "stale_link"})
    for path in (cat.settings.home / "wiki" / "pending").glob("*.json"):
        issues.append({"page_id": path.stem, "code": "interrupted_publication"})
    return {
        "issues": issues,
        "mechanical_valid": not issues,
        "semantic_support_checked": False,
    }


def recover(cat: Catalog, request: WikiRequest):
    """Resolve filesystem/SQLite publication intents without overwriting edits."""
    restored = []
    for intent in sorted((cat.settings.home / "wiki" / "pending").glob("*.json")):
        if request.page_id and request.page_id != intent.stem:
            continue
        payload = json.loads(intent.read_text(encoding="utf-8"))
        page_id = intent.stem
        if payload["page_id"] != page_id or not re.fullmatch(
            r"[a-z0-9][a-z0-9_-]{0,79}", page_id
        ):
            raise LibraryError("invalid_publication_intent", "Invalid page identity")
        path = page_path(cat, page_id)
        before = payload["before"]
        with cat.transaction():
            actual = file_hash(path) if path.exists() else None
            before_hash = digest(before) if before is not None else None
            if actual not in {before_hash, payload["after_hash"]}:
                raise LibraryError(
                    "manual_edit_conflict",
                    "Page changed after interruption; preserve edits before recovery",
                )
            current = cat.one(
                "SELECT current_revision,file_hash FROM wiki_pages WHERE page_id=?",
                (page_id,),
            )
            if current and current["current_revision"] == payload["revision"]:
                if actual != current["file_hash"]:
                    raise LibraryError(
                        "manual_edit_conflict", "Published page differs from disk"
                    )
                action = "finalized"
            else:
                if before is None:
                    path.unlink(missing_ok=True)
                else:
                    atomic_text(path, before)
                action = "restored_previous"
            cat.log("wiki_recover", {"page_id": page_id, "action": action})
        intent.unlink()
        restored.append({"page_id": page_id, "action": action})
    refresh_index(cat)
    return {"pages": restored}
