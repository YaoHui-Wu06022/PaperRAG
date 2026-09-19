"""Immutable source import. Existing legacy data is read, never rewritten."""

from __future__ import annotations

import json
import re
import shutil
import uuid
from pathlib import Path

from paper_rag.ingest import extract
from paper_rag.ingest.mineru import MinerUClient
from .catalog import Catalog
from .common import (
    LibraryError,
    atomic_text,
    digest,
    dumps,
    file_hash,
    json_lines,
    now,
    read_json,
)
from .text import CHUNKER_VERSION, chunks_for, lexical

NORMALIZER = "mineru-blocks-v2.1"


def snapshot_source(cat: Catalog, source: Path, source_hash: str) -> Path:
    destination = (
        cat.settings.home / "objects" / source_hash / ("source" + source.suffix.lower())
    )
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp = destination.with_suffix(".tmp-" + uuid.uuid4().hex)
        shutil.copyfile(source, temp)
        if file_hash(temp) != source_hash:
            temp.unlink()
            raise LibraryError("source_changed", "Source changed during import; retry")
        temp.replace(destination)
    return destination


def snapshot_mineru(cat: Catalog, source: Path) -> tuple[str, Path]:
    inventory = [
        (p.relative_to(source).as_posix(), file_hash(p))
        for p in sorted(source.rglob("*"))
        if p.is_file()
    ]
    fingerprint = digest(inventory)
    destination = cat.settings.home / "objects" / "mineru" / fingerprint
    if not destination.exists():
        temp = destination.with_name(fingerprint + ".tmp-" + uuid.uuid4().hex)
        temp.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, temp)
        copied = [
            (p.relative_to(temp).as_posix(), file_hash(p))
            for p in sorted(temp.rglob("*"))
            if p.is_file()
        ]
        if copied != inventory:
            raise LibraryError(
                "source_changed", "MinerU output changed during snapshot"
            )
        temp.rename(destination)
    return fingerprint, destination


def markdown_structure(text: str):
    sections = [
        {
            "section_id": "s_root",
            "parent_id": None,
            "title": "Document",
            "region": "body",
            "order": 0,
            "path": ["Document"],
        }
    ]
    blocks, stack, pending = [], [], []
    section = "s_root"
    fence = False

    def flush():
        if not pending:
            return
        block_text = "".join(line for _, line in pending)
        blocks.append(
            {
                "block_id": f"b{len(blocks):06d}",
                "order": len(blocks),
                "text": block_text,
                "region": "body",
                "type": "paragraph",
                "section_id": section,
                "line_start": pending[0][0],
                "line_end": pending[-1][0],
                "page": None,
            }
        )
        pending.clear()

    for number, line in enumerate(text.splitlines(keepends=True), 1):
        if line.lstrip().startswith(("```", "~~~")):
            fence = not fence
        heading = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line) if not fence else None
        if heading:
            flush()
            level, title = len(heading[1]), heading[2]
            while stack and stack[-1][0] >= level:
                stack.pop()
            section = f"s_{len(sections):04d}"
            sections.append(
                {
                    "section_id": section,
                    "parent_id": stack[-1][1] if stack else "s_root",
                    "title": title,
                    "region": "body",
                    "order": len(sections),
                    "path": [x[2] for x in stack] + [title],
                }
            )
            stack.append((level, section, title))
        pending.append((number, line))
        if not line.strip() and not fence:
            flush()
    flush()
    return sections, blocks, []


def pdf_structure(mineru: Path):
    flat = extract.flatten_pages(
        extract.load_content_list_v2(extract.find_content_list_v2_path(mineru))
    )
    boundaries = extract.find_region_boundaries(flat)
    sections, _ = extract.build_toc(flat, boundaries)
    blocks = extract.build_blocks(flat, boundaries, sections)
    references = extract.build_references(flat, boundaries)
    by_id = {b["block_id"] for b in blocks}
    for reference in references:
        bid = reference["source_block_id"]
        if bid not in by_id:
            original = next((b for b in flat if f"b{b.index:06d}" == bid), None)
            blocks.append(
                {
                    "block_id": bid,
                    "order": len(blocks),
                    "region": "references",
                    "type": "reference",
                    "text": original.text if original else reference["raw_text"],
                    "section_id": None,
                    "page": reference.get("page"),
                    "bbox": original.bbox if original else None,
                }
            )
            by_id.add(bid)
    return sections, blocks, references


def normalize_metadata(metadata: dict, source: Path) -> dict:
    year = metadata.get("year") or {}
    if isinstance(year, int):
        year = {"publish_year": year}
    if not isinstance(year, dict):
        raise LibraryError("invalid_metadata", "year must be an integer or year object")
    authors = metadata.get("authors", metadata.get("author", [])) or []
    if isinstance(authors, str):
        authors = [authors]
    tags = metadata.get("tags") or []
    if isinstance(tags, dict):
        tags = [v for values in tags.values() for v in values]
    return {
        "title": metadata.get("title") or source.stem,
        "authors": authors,
        "year": year,
        "venue": metadata.get("venue"),
        "aliases": metadata.get("aliases", []),
        "tags": tags,
        "doi": metadata.get("doi"),
        "arxiv_id": metadata.get("arxiv_id"),
        "abstract": metadata.get("abstract"),
        "missing_fields": [
            key
            for key, value in (
                ("authors", authors),
                ("year", any(year.values())),
                ("venue", metadata.get("venue")),
            )
            if not value
        ],
    }


def import_document(
    cat: Catalog,
    path: Path,
    *,
    metadata: dict | None = None,
    mineru_output: Path | None = None,
    document_id: str | None = None,
) -> dict:
    path = path.resolve()
    if not path.is_file() or path.suffix.lower() not in {".pdf", ".md", ".markdown"}:
        raise LibraryError(
            "invalid_source", "Source must be an existing PDF or Markdown file"
        )
    if path.is_relative_to(cat.settings.home / "wiki"):
        raise LibraryError(
            "generated_source",
            "Generated wiki pages cannot be imported as primary sources",
        )
    content_hash = file_hash(path)
    existing = cat.one(
        "SELECT document_id FROM revisions WHERE source_hash=? LIMIT 1", (content_hash,)
    )
    located = cat.one("SELECT document_id FROM locations WHERE path=?", (str(path),))
    if existing and document_id and existing["document_id"] != document_id:
        raise LibraryError(
            "identity_conflict", "These source bytes already belong to another document"
        )
    document_id = (
        document_id
        or (existing or located or {}).get("document_id")
        or "d_" + uuid.uuid4().hex
    )
    old = cat.one("SELECT * FROM documents WHERE document_id=?", (document_id,))
    metadata = normalize_metadata(
        (json.loads(old["metadata"]) if old else {}) | (metadata or {}), path
    )
    kind = "paper" if path.suffix.lower() == ".pdf" else "note"
    source = snapshot_source(cat, path, content_hash)
    assets = None
    if kind == "paper":
        if mineru_output is None:
            cached = cat.one(
                "SELECT assets_path FROM revisions WHERE source_hash=? AND assets_path IS NOT NULL LIMIT 1",
                (content_hash,),
            )
            if cached:
                mineru_output = Path(cached["assets_path"])
            else:
                env = cat.settings.env
                token = env.get("MINERU_API_KEY") or env.get("MINERU_API_TOKEN")
                if not token:
                    raise LibraryError(
                        "dependency_unavailable",
                        "MinerU output or MINERU_API_KEY is required",
                    )
                output = cat.settings.home / "staging" / ("mineru-" + uuid.uuid4().hex)
                mineru_output = MinerUClient(
                    token,
                    env.get("MINERU_API_BASE_URL", "https://mineru.net/api/v4"),
                    env.get("MINERU_MODEL_VERSION", "vlm"),
                    env.get("MINERU_LANGUAGE", "en"),
                ).parse_local_pdf(source, output, content_hash)
        parser_hash, assets = snapshot_mineru(cat, mineru_output)
        sections, blocks, references = pdf_structure(assets)
    else:
        parser_hash = "markdown-v1"
        sections, blocks, references = markdown_structure(
            source.read_text(encoding="utf-8-sig")
        )
    if not blocks:
        raise LibraryError("empty_source", "No readable blocks were extracted")
    revision_id = "r_" + digest(
        [document_id, content_hash, parser_hash, NORMALIZER, CHUNKER_VERSION]
    )
    prior = cat.one("SELECT status FROM revisions WHERE revision_id=?", (revision_id,))
    with cat.transaction():
        if old and old["metadata"] != dumps(metadata):
            cat.db.execute(
                "UPDATE documents SET metadata=? WHERE document_id=?",
                (dumps(metadata), document_id),
            )
            if old["active_revision"]:
                cat.db.execute(
                    "UPDATE chunk_fts SET title=? WHERE rowid IN (SELECT f.fts_rowid FROM chunks c JOIN chunk_fts_rows f ON f.chunk_id=c.chunk_id WHERE c.revision_id=?)",
                    (lexical(metadata["title"]), old["active_revision"]),
                )
            invalidate_wiki(cat, document_id)
            cat.set("corpus_version", cat.snapshot() + 1)
        elif not old:
            cat.db.execute(
                "INSERT INTO documents VALUES(?,?,?,?,0,?)",
                (document_id, kind, None, dumps(metadata), now()),
            )
        cat.db.execute(
            "INSERT INTO locations VALUES(?,?) ON CONFLICT(path) DO UPDATE SET document_id=excluded.document_id",
            (str(path), document_id),
        )
        if not prior:
            cat.db.execute(
                "INSERT INTO revisions VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    revision_id,
                    document_id,
                    content_hash,
                    parser_hash,
                    NORMALIZER,
                    CHUNKER_VERSION,
                    str(source),
                    str(assets) if assets else None,
                    "staged",
                    now(),
                ),
            )
            for section in sections:
                cat.db.execute(
                    "INSERT INTO sections VALUES(?,?,?,?,?,?,?)",
                    (
                        revision_id,
                        section["section_id"],
                        section.get("parent_id"),
                        section["title"],
                        section["region"],
                        section.get("order", 0),
                        dumps(section),
                    ),
                )
            for block in blocks:
                cat.db.execute(
                    "INSERT INTO blocks VALUES(?,?,?,?,?,?,?,?)",
                    (
                        revision_id,
                        block["block_id"],
                        block.get("section_id"),
                        block.get("order", 0),
                        block["region"],
                        block["type"],
                        block["text"],
                        dumps(block),
                    ),
                )
            for chunk in chunks_for(
                document_id, revision_id, metadata["title"], blocks, sections
            ):
                cat.db.execute(
                    "INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?,?)",
                    tuple(chunk.values()),
                )
            for reference in references:
                cat.db.execute(
                    "INSERT INTO citations VALUES(?,?,?,?,?)",
                    (
                        revision_id,
                        reference["reference_id"],
                        None,
                        reference["raw_text"],
                        dumps(reference),
                    ),
                )
            cat.job(
                "index_" + revision_id, "index", document_id, revision_id, "pending", {}
            )
        cat.job(
            "curate_" + revision_id,
            "curate",
            document_id,
            revision_id,
            "pending" if not prior else _job_state(cat, "curate_" + revision_id),
            {},
        )
        cat.job(
            "metadata_" + document_id,
            "metadata",
            document_id,
            revision_id,
            "pending" if metadata["missing_fields"] else "done",
            {"missing_fields": metadata["missing_fields"]},
        )
        cat.log(
            "import",
            {
                "document_id": document_id,
                "revision_id": revision_id,
                "unchanged": bool(prior),
            },
        )
    return {
        "document_id": document_id,
        "revision_id": revision_id,
        "state": prior["status"] if prior else "staged",
        "unchanged": bool(prior),
    }


def _job_state(cat: Catalog, job_id: str):
    return (cat.one("SELECT state FROM jobs WHERE job_id=?", (job_id,)) or {}).get(
        "state", "pending"
    )


def invalidate_wiki(cat: Catalog, document_id: str) -> None:
    pages = cat.rows(
        "SELECT DISTINCT page_id FROM wiki_dependencies WHERE document_id=?",
        (document_id,),
    )
    for page in pages:
        cat.db.execute(
            "UPDATE wiki_pages SET stale=1 WHERE page_id=?", (page["page_id"],)
        )
        cat.job(
            "refresh_" + page["page_id"],
            "wiki_refresh",
            document_id,
            None,
            "pending",
            page,
        )


def remove_fts_document(cat: Catalog, document_id: str) -> None:
    cat.db.execute(
        "DELETE FROM chunk_fts WHERE rowid IN (SELECT f.fts_rowid FROM chunks c JOIN chunk_fts_rows f ON f.chunk_id=c.chunk_id WHERE c.document_id=?)",
        (document_id,),
    )
    cat.db.execute(
        "DELETE FROM chunk_fts_rows WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE document_id=?)",
        (document_id,),
    )


def publish_revision(cat: Catalog, revision_id: str) -> None:
    revision = cat.one("SELECT * FROM revisions WHERE revision_id=?", (revision_id,))
    document_id = revision["document_id"]
    document = cat.one("SELECT * FROM documents WHERE document_id=?", (document_id,))
    remove_fts_document(cat, document_id)
    metadata = json.loads(document["metadata"])
    for chunk in cat.db.execute(
        "SELECT c.*,s.title AS section_title FROM chunks c LEFT JOIN sections s ON c.revision_id=s.revision_id AND c.section_id=s.section_id WHERE c.revision_id=?",
        (revision_id,),
    ):
        inserted = cat.db.execute(
            "INSERT INTO chunk_fts VALUES(?,?,?,?)",
            (
                chunk["chunk_id"],
                lexical(metadata["title"]),
                lexical(chunk["section_title"] or ""),
                lexical(chunk["text"]),
            ),
        )
        cat.db.execute(
            "INSERT INTO chunk_fts_rows VALUES(?,?)",
            (chunk["chunk_id"], inserted.lastrowid),
        )
    cat.db.execute(
        "UPDATE revisions SET status='superseded' WHERE document_id=? AND status='active'",
        (document_id,),
    )
    cat.db.execute(
        "UPDATE revisions SET status='active' WHERE revision_id=?", (revision_id,)
    )
    cat.db.execute(
        "UPDATE documents SET active_revision=?,withdrawn=0 WHERE document_id=?",
        (revision_id, document_id),
    )
    cat.job("index_" + revision_id, "index", document_id, revision_id, "done", {})
    invalidate_wiki(cat, document_id)
    cat.set("corpus_version", cat.snapshot() + 1)


def resolve_citations(cat: Catalog) -> None:
    from collections import Counter, defaultdict
    from paper_rag.ingest.citation_graph import (
        normalized_title_key,
        first_author_surname,
    )

    papers = cat.rows(
        "SELECT * FROM documents WHERE active_revision IS NOT NULL AND withdrawn=0 AND source_kind='paper'"
    )
    signature = digest(
        [
            (p["document_id"], p["active_revision"], p["metadata"])
            for p in sorted(papers, key=lambda p: p["document_id"])
        ]
    )
    if cat.get("citation_signature") == signature:
        return
    prepared = []
    for paper in papers:
        meta = json.loads(paper["metadata"])
        title = normalized_title_key(meta["title"])
        prepared.append(
            (
                paper["document_id"],
                title,
                first_author_surname(meta["authors"]),
                {str(y) for y in meta["year"].values() if isinstance(y, int)},
            )
        )
    frequencies = Counter(
        word for _, title, _, _ in prepared for word in set(title.split())
    )
    anchors = defaultdict(list)
    for paper in prepared:
        if paper[1]:
            anchor = min(paper[1].split(), key=lambda word: (frequencies[word], word))
            anchors[anchor].append(paper)
    for row in cat.db.execute(
        "SELECT c.* FROM citations c JOIN documents d ON d.active_revision=c.revision_id"
    ):
        matches = []
        raw = normalized_title_key(row["raw_text"])
        tokens = set(raw.split())
        for document_id, title, surname, years in (
            p for token in tokens for p in anchors.get(token, [])
        ):
            # Exact normalized title, author and year: conservative, local-only edges.
            if (
                title
                and title in raw
                and surname
                and surname.casefold() in raw
                and any(str(y) in row["raw_text"] for y in years)
            ):
                matches.append(document_id)
        cat.db.execute(
            "UPDATE citations SET target_document_id=? WHERE revision_id=? AND reference_id=?",
            (
                matches[0] if len(matches) == 1 else None,
                row["revision_id"],
                row["reference_id"],
            ),
        )
    cat.set("citation_signature", signature)


def migrate_legacy(cat: Catalog) -> dict:
    root = cat.settings.root
    manifest = json_lines(root / "data" / "manifest.jsonl")
    annotations_path = root / "data" / "paper_annotations.json"
    annotations = read_json(annotations_path) if annotations_path.exists() else {}
    results, failures = [], []
    for item in manifest:
        if item.get("status") == "deleted":
            continue
        try:
            migrated = cat.one(
                "SELECT document_id,revision_id,status FROM revisions WHERE source_hash=? ORDER BY created_at DESC LIMIT 1",
                (item["file_hash"],),
            )
            if migrated:
                results.append(
                    {
                        "document_id": migrated["document_id"],
                        "revision_id": migrated["revision_id"],
                        "state": migrated["status"],
                        "unchanged": True,
                    }
                )
                continue
            if not item.get("mineru_output_path"):
                raise LibraryError(
                    "missing_parse", "Legacy source has no successful MinerU output"
                )
            meta = item | annotations.get(item["file_hash"], {})
            results.append(
                import_document(
                    cat,
                    Path(item["pdf_path"]),
                    metadata=meta,
                    mineru_output=Path(item["mineru_output_path"]),
                )
            )
        except Exception as exc:
            failures.append(
                {
                    "source_hash": item.get("file_hash"),
                    "error": type(exc).__name__,
                    "message": str(exc),
                }
            )
    return {"documents": results, "failures": failures}
