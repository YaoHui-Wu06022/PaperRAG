"""Rehearse a live generation rebuild and rollback using cached embeddings."""

import json
from pathlib import Path

from paper_rag.library.catalog import Catalog
from paper_rag.library.settings import LibrarySettings
from paper_rag.library.vectors import index_library, rollback


def main():
    settings = LibrarySettings.load(Path.cwd())
    with Catalog(settings, writable=True) as cat:
        old = cat.get("active_generation")
        assert old, "A published generation is required"
        before = cat.rows(
            "SELECT document_id,active_revision FROM documents ORDER BY document_id"
        )
        cache_size = cat.one("SELECT count(*) n FROM embedding_cache")["n"]
        new = index_library(cat, rebuild=True)
        rollback(cat, old)
        assert (
            cat.rows(
                "SELECT document_id,active_revision FROM documents ORDER BY document_id"
            )
            == before
        )
        rollback(cat, new["generation_id"])
        assert cat.one("SELECT count(*) n FROM embedding_cache")["n"] == cache_size
        report = {
            "previous_generation": old,
            "new_generation": new,
            "rollback_verified": True,
            "cached_embeddings_reused": True,
            "active_generation": cat.get("active_generation"),
        }
        (settings.home / "rollback-verification.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report))


if __name__ == "__main__":
    main()
