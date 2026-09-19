"""Explicit, resumable Milvus archive and collection lifecycle operations."""
from __future__ import annotations

import json
from pathlib import Path

from .common import LibraryError, atomic_text, digest, dumps, now
from .locking import single_writer
from .vectors import VectorStore

ARCHIVE_VERSION = 1
MAX_COLLECTIONS = 5


def _client(cat):
    store = VectorStore(cat.settings)
    return store, store.client


def _canonical(row):
    return dumps(row)


def _archive_dir(cat, name):
    return cat.settings.home / "milvus_archives" / name


def _manifest(cat, name, client, rows):
    desc = client.describe_collection(collection_name=name)
    try:
        schema = client.describe_collection(collection_name=name).get("schema", {})
    except Exception:
        schema = {}
    try:
        indexes = client.list_indexes(collection_name=name)
    except Exception:
        indexes = []
    aliases = []
    try:
        aliases = client.list_aliases(collection_name=name)
    except Exception:
        pass
    canonical = "".join(_canonical(r) + "\n" for r in rows)
    alias_rows = []
    try:
        store, client = _client(cat)
        for name in sorted(names):
            try:
                found = client.list_aliases(collection_name=name)
                for alias in (found.get("aliases", []) if isinstance(found, dict) else found):
                    alias_rows.append({"collection": name, "alias": alias})
            except Exception:
                pass
        store.close()
    except Exception:
        pass
    return {
        "archive_version": ARCHIVE_VERSION,
        "collection": name,
        "created_at": now(),
        "description": desc,
        "schema": schema,
        "schema_digest": digest(schema),
        "indexes": indexes,
        "aliases": aliases,
        "row_count": len(rows),
        "unique_primary_keys": len({r.get("chunk_id") for r in rows}),
        "vector_dimensions": len(rows[0].get("vector", [])) if rows else 0,
        "rows_digest": digest(canonical),
        "archive_digest": digest([name, schema, rows]),
    }


def inspect(cat):
    registry = cat.rows("SELECT * FROM collection_registry ORDER BY collection_name")
    generations = cat.rows("SELECT generation_id,collection_name,state,created_at FROM index_generations ORDER BY created_at")
    names = {r["collection_name"] for r in registry}
    names.update(r["collection_name"] for r in generations)
    try:
        store, client = _client(cat)
        remote = []
        for name in sorted(names):
            try:
                desc = client.describe_collection(collection_name=name)
                remote.append({"collection": name, "row_count": desc.get("num_entities"), "exists": True})
            except Exception:
                remote.append({"collection": name, "exists": False})
        store.close()
    except Exception as exc:
        remote = [{"warning": getattr(exc, "code", type(exc).__name__)}]
    protected = {"content_index", "paper_rag_chunks__staging_fbaab99f121f"}
    protected.update(r["collection_name"] for r in generations if r["state"] == "ready")
    return {
        "max_collections": MAX_COLLECTIONS,
        "used_collections": len(names),
        "available_slots": max(0, MAX_COLLECTIONS - len(names)),
        "protected_collections": sorted(protected),
        "current_generation": cat.get("active_generation"),
        "previous_generation": cat.get("previous_generation"),
        "building_generation": cat.one("SELECT generation_id FROM index_generations WHERE state='building'"),
        "retirable_collections": [r["collection_name"] for r in registry if r["status"] == "online" and not r.get("protected") and r["collection_name"] not in protected],
        "remote": remote,
        "aliases": alias_rows,
    }


@single_writer
def archive(cat, collection, *, force=False):
    collection = str(collection)
    directory = _archive_dir(cat, collection)
    temp = directory.with_name(directory.name + ".tmp")
    store, client = _client(cat)
    try:
        if directory.exists() and not force:
            manifest_path = directory / "manifest.json"
            if manifest_path.exists():
                return json.loads(manifest_path.read_text(encoding="utf-8"))
        temp.mkdir(parents=True, exist_ok=True)
        rows = []
        iterator = client.query_iterator(collection_name=collection, batch_size=500, limit=-1, output_fields=["*"])
        try:
            while True:
                batch = iterator.next()
                if not batch:
                    break
                rows.extend(batch)
        finally:
            try:
                iterator.close()
            except Exception:
                pass
        rows.sort(key=lambda r: str(r.get("chunk_id", "")))
        manifest = _manifest(cat, collection, client, rows)
        atomic_text(temp / "rows.jsonl", "".join(_canonical(r) + "\n" for r in rows))
        atomic_text(temp / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        if directory.exists():
            import shutil
            shutil.rmtree(directory)
        temp.replace(directory)
        with cat.transaction():
            cat.db.execute("INSERT INTO collection_registry(collection_name,role,status,archive_path,archive_digest) VALUES(?,?,?,?,?) ON CONFLICT(collection_name) DO UPDATE SET status='archived',archive_path=excluded.archive_path,archive_digest=excluded.archive_digest", (collection, "legacy", "archived", str(directory), manifest["archive_digest"]))
            cat.log("collection_archive", {"collection": collection, "archive_digest": manifest["archive_digest"], "row_count": len(rows)})
        return manifest
    finally:
        store.close()


def _load_archive(cat, collection, archive_digest):
    directory = _archive_dir(cat, collection)
    manifest_path = directory / "manifest.json"
    rows_path = directory / "rows.jsonl"
    if not manifest_path.exists() or not rows_path.exists():
        raise LibraryError("not_found", "Archive files not found")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["archive_digest"] != archive_digest:
        raise LibraryError("archive_digest_mismatch", "Archive digest does not match request")
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != manifest["row_count"] or digest([collection, manifest.get("schema", {}), rows]) != archive_digest:
        raise LibraryError("archive_corrupt", "Archive content failed digest validation")
    return manifest, rows


@single_writer
def restore(cat, collection, archive_digest, *, target=None, dry_run=False):
    manifest, rows = _load_archive(cat, collection, archive_digest)
    target = target or (cat.settings.collection_prefix + "_restore_" + digest([collection, archive_digest])[:12])
    names = set(r["collection_name"] for r in cat.rows("SELECT collection_name FROM collection_registry WHERE status!='retired'"))
    names.update(r["collection_name"] for r in cat.rows("SELECT collection_name FROM index_generations"))
    if target not in names and len(names) >= MAX_COLLECTIONS:
        raise LibraryError("collection_capacity", "Milvus collection limit reached; retire an archived collection first")
    if dry_run:
        return {"dry_run": True, "target": target, "rows": len(rows), "archive_digest": archive_digest}
    store = VectorStore(cat.settings)
    try:
        if target not in names:
            store.create(target)
        vectors = [r.pop("vector") for r in rows]
        store.upsert(target, rows, vectors)
        store.ready(target)
        store.verify(target, rows)
        with cat.transaction():
            cat.db.execute("INSERT OR REPLACE INTO collection_registry(collection_name,role,status,archive_path,archive_digest) VALUES(?,?,?,?,?)", (target, "restore", "online", str(_archive_dir(cat, collection)), archive_digest))
            cat.log("collection_restore", {"source": collection, "target": target, "archive_digest": archive_digest})
        return {"target": target, "rows": len(rows), "archive_digest": archive_digest, "verified": True}
    finally:
        store.close()


@single_writer
def retire(cat, collection, archive_digest, *, dry_run=False):
    row = cat.one("SELECT * FROM collection_registry WHERE collection_name=?", (collection,))
    if not row or row.get("archive_digest") != archive_digest:
        raise LibraryError("archive_digest_mismatch", "Collection is not bound to the supplied archive")
    protected = {"content_index", "paper_rag_chunks__staging_fbaab99f121f", cat.get("active_generation"), cat.get("previous_generation")}
    protected.update(r["collection_name"] for r in cat.rows("SELECT collection_name FROM index_generations WHERE state='ready'"))
    if collection in protected:
        raise LibraryError("protected_collection", "Protected collection cannot be retired")
    if dry_run:
        return {"dry_run": True, "collection": collection, "archive_digest": archive_digest}
    store = VectorStore(cat.settings)
    try:
        store.client.drop_collection(collection_name=collection)
        with cat.transaction():
            cat.db.execute("UPDATE collection_registry SET status='retired' WHERE collection_name=?", (collection,))
            cat.log("collection_retire", {"collection": collection, "archive_digest": archive_digest})
        return {"collection": collection, "retired": True, "archive_digest": archive_digest}
    finally:
        store.close()
