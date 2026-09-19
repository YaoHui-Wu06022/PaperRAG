"""Verify an existing synthetic fixture after an interrupted capacity run."""

import json
from pathlib import Path
import statistics
import time
from paper_rag.library.catalog import Catalog
from paper_rag.library.settings import LibrarySettings
from paper_rag.library.contracts import SearchRequest, ReadRequest
from paper_rag.library.reading import read
from paper_rag.library.search import search
from paper_rag.library.vectors import index_library
from scripts.benchmark_library_scale import peak_memory


def main():
    paths = sorted(
        (Path.cwd() / "data/scale").glob("*/catalog.sqlite3"),
        key=lambda p: p.stat().st_mtime,
    )
    settings = LibrarySettings(Path.cwd(), paths[-1].parent, {})
    with Catalog(settings, writable=True) as cat:
        documents = cat.one("SELECT count(*) n FROM documents")["n"]
        chunks = cat.one("SELECT count(*) n FROM chunks")["n"]
        print(f"resume {documents} documents / {chunks} chunks", flush=True)
        start = time.perf_counter()
        index_library(cat, lexical_only=True)
        publish_seconds = time.perf_counter() - start
        print(f"published in {publish_seconds:.2f}s", flush=True)
        timings = []
        for number in range(20):
            result = search(
                cat, SearchRequest(query=f"topic{number*17}", mode="lexical")
            )
            assert result["items"]
            timings.append(result["timings"]["total_ms"])
        request = ReadRequest(document_id="d_0", max_chars=997)
        fragments = []
        while True:
            result = read(cat, request)
            fragments.extend(f["text"] for f in result["fragments"])
            if not result["next_cursor"]:
                break
            request = request.model_copy(update={"cursor": result["next_cursor"]})
        expected = "".join(
            r["text"]
            for r in cat.rows(
                "SELECT text FROM blocks WHERE revision_id='r_0' ORDER BY ordinal"
            )
        )
        assert "".join(fragments) == expected
        start = time.perf_counter()
        snapshot = cat.snapshot()
        index_library(cat, lexical_only=True)
        assert cat.snapshot() == snapshot
        report = {
            "documents": documents,
            "chunks": chunks,
            "fts_publish_seconds": publish_seconds,
            "unchanged_index_seconds": time.perf_counter() - start,
            "search_p50_ms": statistics.median(timings),
            "search_max_ms": max(timings),
            "peak_process_rss_bytes": peak_memory(),
            "paged_read_complete": True,
            "fixture_build": "existing synthetic fixture; original slow run interrupted before FTS publication",
            "scope": "SQLite/FTS and source pagination; excludes remote embedding and Milvus scale",
        }
    report["database_bytes"] = settings.database.stat().st_size
    destination = settings.home / "report.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report | {"path": str(destination)}))


if __name__ == "__main__":
    main()
