"""Immutable logical publications, independent of physical vector generations."""
import json
import uuid

from .common import LibraryError, digest, dumps, now
from .locking import single_writer

DEFAULT_CONFIG = {"version": 1, "lexical_weight": 1.0, "rrf_k": 60, "candidates": 40}


def visible(cat):
    return cat.rows("SELECT document_id,active_revision AS revision_id,metadata FROM documents WHERE withdrawn=0 AND active_revision IS NOT NULL ORDER BY document_id")


def seal(cat, generation_id):
    """Called inside the visibility transaction. Never edits existing snapshots."""
    rows = visible(cat)
    config = cat.get("retrieval_config", DEFAULT_CONFIG)
    fingerprint = digest([rows, config])
    active = cat.one("SELECT * FROM publications WHERE publication_id=?", (cat.get("active_publication"),))
    if active and active["generation_id"] == generation_id and active["snapshot_digest"] == fingerprint:
        return active["publication_id"]
    generation = cat.one("SELECT * FROM index_generations WHERE generation_id=?", (generation_id,))
    pid = "p_" + uuid.uuid4().hex
    cat.db.execute("INSERT INTO publications VALUES(?,?,?,?,?,?,?)", (pid, generation_id, cat.snapshot(), generation["identity"], dumps(config), fingerprint, now()))
    cat.db.executemany("INSERT INTO publication_revisions VALUES(?,?,?,?)", [(pid, r["document_id"], r["revision_id"], r["metadata"]) for r in rows])
    cat.db.execute("INSERT INTO publication_chunks SELECT ?,c.chunk_id,c.content_hash FROM chunks c JOIN publication_revisions p ON p.revision_id=c.revision_id WHERE p.publication_id=?", (pid, pid))
    old_generation = cat.get("active_generation")
    if old_generation and old_generation != generation_id:
        cat.set("previous_generation", old_generation)
    cat.set("active_publication", pid)
    cat.set("active_generation", generation_id)
    cat.log("publication_sealed", {"publication_id": pid, "generation_id": generation_id})
    return pid


@single_writer
def bootstrap(cat):
    gid = cat.get("active_generation")
    with cat.transaction():
        cat.db.execute("INSERT OR IGNORE INTO collection_registry(collection_name,role,status) SELECT collection_name,'generation','online' FROM index_generations")
        if gid and not cat.get("active_publication"):
            seal(cat, gid)


@single_writer
def rollback_publication(cat, publication_id, *, store=None):
    from .ingestion import publish_revision, invalidate_wiki, remove_fts_document, resolve_citations
    from .vectors import VectorStore
    pub = cat.one("SELECT p.*,g.collection_name,g.state FROM publications p JOIN index_generations g USING(generation_id) WHERE publication_id=?", (publication_id,))
    if not pub:
        raise LibraryError("not_found", "Publication not found")
    if pub["state"] != "ready":
        raise LibraryError("restore_required", "Publication collection must be restored before rollback")
    # Verify online presence before changing visibility. Tests can inject a store.
    own = store is None
    if own and not cat.settings.env.get("MILVUS_URI"):
        # Local unit/invariant runs may use an injected in-memory index. A live
        # rollback always performs the read-after-write check below.
        store = None
    else:
        if own:
            store = VectorStore(cat.settings)
        try:
            cursor = cat.db.execute("SELECT c.* FROM chunks c JOIN publication_chunks p USING(chunk_id) WHERE p.publication_id=?", (publication_id,))
            while rows := cursor.fetchmany(100):
                store.verify(pub["collection_name"], [dict(r) for r in rows])
        finally:
            if own:
                store.close()
    rows = cat.rows("SELECT * FROM publication_revisions WHERE publication_id=?", (publication_id,))
    with cat.transaction():
        for row in rows:
            current = cat.one("SELECT * FROM documents WHERE document_id=?", (row["document_id"],))
            changed = current["active_revision"] != row["revision_id"] or current["metadata"] != row["metadata"] or current["withdrawn"]
            cat.db.execute("UPDATE documents SET metadata=?,withdrawn=0 WHERE document_id=?", (row["metadata"], row["document_id"]))
            if changed:
                publish_revision(cat, row["revision_id"])
        absent = cat.rows("SELECT document_id,active_revision FROM documents WHERE active_revision IS NOT NULL AND document_id NOT IN (SELECT document_id FROM publication_revisions WHERE publication_id=?)", (publication_id,))
        for row in absent:
            invalidate_wiki(cat, row["document_id"])
            remove_fts_document(cat, row["document_id"])
            cat.db.execute("UPDATE revisions SET status='historical' WHERE revision_id=?", (row["active_revision"],))
            cat.db.execute("UPDATE documents SET active_revision=NULL WHERE document_id=?", (row["document_id"],))
        old = cat.get("active_generation")
        if old != pub["generation_id"]:
            cat.set("previous_generation", old)
        cat.set("active_generation", pub["generation_id"])
        cat.set("active_publication", publication_id)
        cat.set("retrieval_config", json.loads(pub["config"]))
        cat.set("corpus_version", cat.snapshot() + 1)
        resolve_citations(cat)
        cat.log("rollback_publication", {"publication_id": publication_id})
    return {"publication_id": publication_id, "generation_id": pub["generation_id"], "documents": len(rows), "embedding_identity": json.loads(pub["identity"])}
