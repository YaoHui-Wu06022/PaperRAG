from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager

from .common import LibraryError, dumps, now
from .settings import LibrarySettings

SCHEMA_VERSION = 4
SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS documents(
 document_id TEXT PRIMARY KEY, source_kind TEXT NOT NULL, active_revision TEXT,
 metadata TEXT NOT NULL, withdrawn INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS locations(path TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents);
CREATE TABLE IF NOT EXISTS revisions(
 revision_id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents,
 source_hash TEXT NOT NULL, parser_hash TEXT NOT NULL, normalizer TEXT NOT NULL,
 chunker TEXT NOT NULL, source_path TEXT NOT NULL, assets_path TEXT,
 status TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS revision_source ON revisions(source_hash);
CREATE INDEX IF NOT EXISTS revision_document ON revisions(document_id, status);
CREATE TABLE IF NOT EXISTS sections(
 revision_id TEXT NOT NULL REFERENCES revisions, section_id TEXT NOT NULL, parent_id TEXT,
 title TEXT NOT NULL, region TEXT NOT NULL, ordinal INTEGER NOT NULL, payload TEXT NOT NULL,
 PRIMARY KEY(revision_id,section_id));
CREATE TABLE IF NOT EXISTS blocks(
 revision_id TEXT NOT NULL REFERENCES revisions, block_id TEXT NOT NULL, section_id TEXT,
 ordinal INTEGER NOT NULL, region TEXT NOT NULL, kind TEXT NOT NULL, text TEXT NOT NULL,
 payload TEXT NOT NULL, PRIMARY KEY(revision_id,block_id));
CREATE INDEX IF NOT EXISTS block_section ON blocks(revision_id,section_id,ordinal);
CREATE TABLE IF NOT EXISTS chunks(
 chunk_id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents,
 revision_id TEXT NOT NULL REFERENCES revisions, section_id TEXT, region TEXT NOT NULL,
 text TEXT NOT NULL, embedding_text TEXT NOT NULL, content_hash TEXT NOT NULL, spans TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS chunk_revision ON chunks(revision_id);
CREATE INDEX IF NOT EXISTS chunk_document ON chunks(document_id);
CREATE TABLE IF NOT EXISTS citations(
 revision_id TEXT NOT NULL REFERENCES revisions, reference_id TEXT NOT NULL,
 target_document_id TEXT REFERENCES documents, raw_text TEXT NOT NULL, payload TEXT NOT NULL,
 PRIMARY KEY(revision_id,reference_id));
CREATE INDEX IF NOT EXISTS citation_target ON citations(target_document_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(chunk_id UNINDEXED, title, section, body);
CREATE TABLE IF NOT EXISTS chunk_fts_rows(
 chunk_id TEXT PRIMARY KEY REFERENCES chunks, fts_rowid INTEGER UNIQUE NOT NULL);
CREATE TABLE IF NOT EXISTS jobs(
 job_id TEXT PRIMARY KEY, kind TEXT NOT NULL, document_id TEXT, revision_id TEXT,
 state TEXT NOT NULL, detail TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS embedding_cache(
 cache_key TEXT PRIMARY KEY, identity TEXT NOT NULL, vector BLOB NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS metadata_cache(cache_key TEXT PRIMARY KEY, result TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS index_generations(
 generation_id TEXT PRIMARY KEY, collection_name TEXT UNIQUE NOT NULL, identity TEXT NOT NULL,
 state TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS index_rows(
 generation_id TEXT NOT NULL REFERENCES index_generations, chunk_id TEXT NOT NULL REFERENCES chunks,
 content_hash TEXT NOT NULL, PRIMARY KEY(generation_id,chunk_id));
CREATE TABLE IF NOT EXISTS generation_revisions(
 generation_id TEXT NOT NULL REFERENCES index_generations, document_id TEXT NOT NULL REFERENCES documents,
 revision_id TEXT NOT NULL REFERENCES revisions, PRIMARY KEY(generation_id,document_id));
CREATE TABLE IF NOT EXISTS publications(
 publication_id TEXT PRIMARY KEY, generation_id TEXT NOT NULL REFERENCES index_generations,
 corpus_version INTEGER NOT NULL, identity TEXT NOT NULL, config TEXT NOT NULL,
 snapshot_digest TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS publication_revisions(
 publication_id TEXT NOT NULL REFERENCES publications, document_id TEXT NOT NULL REFERENCES documents,
 revision_id TEXT NOT NULL REFERENCES revisions, metadata TEXT NOT NULL,
 PRIMARY KEY(publication_id,document_id));
CREATE TABLE IF NOT EXISTS publication_chunks(
 publication_id TEXT NOT NULL REFERENCES publications, chunk_id TEXT NOT NULL REFERENCES chunks,
 content_hash TEXT NOT NULL, PRIMARY KEY(publication_id,chunk_id));
CREATE TABLE IF NOT EXISTS collection_registry(
 collection_name TEXT PRIMARY KEY, role TEXT NOT NULL, status TEXT NOT NULL,
 archive_path TEXT, archive_digest TEXT, remote_id TEXT);
CREATE TABLE IF NOT EXISTS wiki_pages(
 page_id TEXT PRIMARY KEY, kind TEXT NOT NULL, title TEXT NOT NULL, current_revision INTEGER NOT NULL,
 file_hash TEXT NOT NULL, stale INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS wiki_revisions(
 page_id TEXT NOT NULL REFERENCES wiki_pages, revision INTEGER NOT NULL, body TEXT NOT NULL,
 evidence TEXT NOT NULL, links TEXT NOT NULL, created_at TEXT NOT NULL, PRIMARY KEY(page_id,revision));
CREATE TABLE IF NOT EXISTS wiki_dependencies(
 page_id TEXT NOT NULL REFERENCES wiki_pages, document_id TEXT NOT NULL REFERENCES documents,
 revision_id TEXT NOT NULL REFERENCES revisions, PRIMARY KEY(page_id,document_id,revision_id));
CREATE TABLE IF NOT EXISTS wiki_links(
 page_id TEXT NOT NULL REFERENCES wiki_pages, target_id TEXT NOT NULL REFERENCES wiki_pages,
 target_revision INTEGER NOT NULL, PRIMARY KEY(page_id,target_id));
CREATE VIRTUAL TABLE IF NOT EXISTS wiki_fts USING fts5(page_id UNINDEXED,title,body);
CREATE TABLE IF NOT EXISTS audit(id INTEGER PRIMARY KEY, action TEXT NOT NULL, detail TEXT NOT NULL, created_at TEXT NOT NULL);
"""


class Catalog:
    def __init__(self, settings: LibrarySettings, *, writable: bool = False):
        self.settings = settings
        self.writable = writable
        if writable:
            settings.home.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(settings.database, timeout=30)
            self.db.execute("PRAGMA journal_mode=WAL")
        else:
            if not settings.database.exists():
                raise LibraryError(
                    "not_initialized", "Run ingest with migrate_legacy=true first"
                )
            self.db = sqlite3.connect(
                settings.database.as_uri() + "?mode=ro", uri=True, timeout=30
            )
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        if writable:
            existing = self.db.execute("PRAGMA user_version").fetchone()[0]
            if existing not in (0, 2, 3, SCHEMA_VERSION):
                raise LibraryError(
                    "schema_mismatch", "Unsupported catalog schema; do not overwrite it"
                )
            self.db.executescript(SCHEMA)
            if existing == 2:
                self.db.execute(
                    "INSERT OR IGNORE INTO chunk_fts_rows SELECT chunk_id,rowid FROM chunk_fts"
                )
            self.db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            self.db.commit()
            from .publications import bootstrap

            bootstrap(self)
        elif self.db.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
            raise LibraryError("schema_mismatch", "Unsupported catalog schema")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.db.close()

    @contextmanager
    def transaction(self):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise

    def rows(self, sql: str, params=()) -> list[dict]:
        return [dict(row) for row in self.db.execute(sql, params)]

    def one(self, sql: str, params=()) -> dict | None:
        row = self.db.execute(sql, params).fetchone()
        return dict(row) if row else None

    def get(self, key: str, default=None):
        row = self.one("SELECT value FROM meta WHERE key=?", (key,))
        return json.loads(row["value"]) if row else default

    def set(self, key: str, value) -> None:
        self.db.execute(
            "INSERT INTO meta VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, dumps(value)),
        )

    def log(self, action: str, detail) -> None:
        self.db.execute(
            "INSERT INTO audit(action,detail,created_at) VALUES(?,?,?)",
            (action, dumps(detail), now()),
        )

    def job(
        self,
        job_id: str,
        kind: str,
        document_id: str | None,
        revision_id: str | None,
        state: str,
        detail,
    ) -> None:
        self.db.execute(
            "INSERT INTO jobs VALUES(?,?,?,?,?,?,?) ON CONFLICT(job_id) DO UPDATE SET "
            "state=excluded.state,detail=excluded.detail,updated_at=excluded.updated_at",
            (job_id, kind, document_id, revision_id, state, dumps(detail), now()),
        )

    def snapshot(self) -> int:
        return self.get("corpus_version", 0)
