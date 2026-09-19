"""Synthetic SQLite/FTS capacity test. Never touches real sources or Milvus.

Run with --documents 10000 --chunks-per-document 50 for acceptance scale.
This measures storage/retrieval mechanics, not real-paper retrieval quality or
remote embedding throughput. Source text size is recorded in the output.
"""

import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import statistics
import time
import uuid

from paper_rag.library.catalog import Catalog
from paper_rag.library.common import dumps, digest, now
from paper_rag.library.contracts import SearchRequest, ReadRequest
from paper_rag.library.reading import read
from paper_rag.library.search import search
from paper_rag.library.settings import LibrarySettings
from paper_rag.library.vectors import index_library


def peak_memory():
    if os.name != "nt":
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t)
            for name in (
                "PeakWorkingSetSize",
                "WorkingSetSize",
                "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage",
                "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage",
                "PagefileUsage",
                "PeakPagefileUsage",
            )
        ]

    result = Counters()
    result.cb = ctypes.sizeof(result)
    ctypes.windll.psapi.GetProcessMemoryInfo(
        ctypes.c_void_p(-1), ctypes.byref(result), result.cb
    )
    return result.PeakWorkingSetSize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents", type=int, default=10000)
    parser.add_argument("--chunks-per-document", type=int, default=50)
    args = parser.parse_args()
    home = Path.cwd() / "data" / "scale" / uuid.uuid4().hex
    settings = LibrarySettings(Path.cwd(), home, {})
    start = time.perf_counter()
    text_chars = 0
    with Catalog(settings, writable=True) as cat:
        for number in range(args.documents):
            doc, rev = f"d_{number}", f"r_{number}"
            with cat.transaction():
                cat.db.execute(
                    "INSERT INTO documents VALUES(?, 'paper', NULL, ?, 0, ?)",
                    (
                        doc,
                        dumps(
                            {
                                "title": f"Synthetic document {number}",
                                "authors": [],
                                "year": {},
                            }
                        ),
                        now(),
                    ),
                )
                cat.db.execute(
                    "INSERT INTO revisions VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (
                        rev,
                        doc,
                        digest(doc),
                        "synthetic",
                        "synthetic",
                        "synthetic",
                        "synthetic.pdf",
                        None,
                        "staged",
                        now(),
                    ),
                )
                for offset in range(args.chunks_per_document):
                    block, chunk = f"b_{offset}", f"c_{number}_{offset}"
                    text = (
                        f"Synthetic research topic{number % 1000} evidence item{offset}. "
                        + "Residual learning attention retrieval evidence verification algorithm memory complexity. "
                        * 5
                    )
                    text_chars += len(text)
                    cat.db.execute(
                        "INSERT INTO blocks VALUES(?,?,NULL,?,'body','paragraph',?,?)",
                        (rev, block, offset, text, dumps({"page": offset})),
                    )
                    cat.db.execute(
                        "INSERT INTO chunks VALUES(?,?,?,NULL,'body',?,?,?,?)",
                        (
                            chunk,
                            doc,
                            rev,
                            text,
                            text,
                            digest(text),
                            dumps([{"block_id": block, "start": 0, "end": len(text)}]),
                        ),
                    )
            if (number + 1) % 1000 == 0:
                print(f"prepared {number+1} documents", flush=True)
        inserted = time.perf_counter()
        index_library(cat, lexical_only=True)
        indexed = time.perf_counter()
        timings = []
        for number in range(20):
            result = search(
                cat, SearchRequest(query=f"topic{number*17}", mode="lexical")
            )
            assert result["items"]
            timings.append(result["timings"]["total_ms"])
        request = ReadRequest(document_id="d_0", max_chars=997)
        retrieved = ""
        pages = 0
        while True:
            result = read(cat, request)
            retrieved += "".join(f["text"] for f in result["fragments"])
            pages += 1
            if not result["next_cursor"]:
                break
            request = request.model_copy(update={"cursor": result["next_cursor"]})
        expected = "".join(
            r["text"]
            for r in cat.rows(
                "SELECT text FROM blocks WHERE revision_id='r_0' ORDER BY ordinal"
            )
        )
        assert retrieved == expected
        unchanged_start = time.perf_counter()
        snapshot = cat.snapshot()
        index_library(cat, lexical_only=True)
        assert cat.snapshot() == snapshot
        report = {
            "documents": args.documents,
            "chunks": args.documents * args.chunks_per_document,
            "text_characters": text_chars,
            "build_seconds": inserted - start,
            "fts_publish_seconds": indexed - inserted,
            "unchanged_index_seconds": time.perf_counter() - unchanged_start,
            "search_p50_ms": statistics.median(timings),
            "search_max_ms": max(timings),
            "peak_process_rss_bytes": peak_memory(),
            "paged_read_complete": True,
            "read_pages": pages,
            "scope": "synthetic SQLite/FTS only; no quality or external service throughput claim",
        }
    report["database_bytes"] = settings.database.stat().st_size
    (home / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report | {"report": str(home / "report.json")}))


if __name__ == "__main__":
    main()
