"""Offline smoke checks against the migrated library; does not call model APIs."""

import json
from pathlib import Path

from paper_rag.library.catalog import Catalog
from paper_rag.library.settings import LibrarySettings
from paper_rag.library.vectors import index_library
from paper_rag.library.contracts import SearchRequest, ReadRequest, PapersRequest
from paper_rag.library.search import search
from paper_rag.library.reading import papers, read


def main():
    settings = LibrarySettings.load(Path.cwd())
    with Catalog(settings, writable=True) as cat:
        publication = index_library(cat, lexical_only=True)
        result = search(cat, SearchRequest(query="attention", mode="lexical"))
        assert result["items"], "No attention evidence found"
        ref = result["items"][0]["evidence_ids"][0]
        evidence = read(cat, ReadRequest(evidence_id=ref))
        assert evidence["fragments"]
        print(
            json.dumps(
                {
                    "publication": publication,
                    "count": papers(cat, PapersRequest(), count=True),
                    "search_hits": len(result["items"]),
                    "evidence_resolves": True,
                }
            )
        )


if __name__ == "__main__":
    main()
