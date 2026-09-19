"""Fetch a curated, reproducible set of public arXiv PDFs into data/pdf.

Only immutable source files and a UTF-8 manifest are written. Parsing is a separate step.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

PAPERS = [
    {
        "arxiv_id": "2608.12984v1",
        "title": "Reconcile Once, Write Anytime: A Trust-Tiered Librarian and a Multi-Agent Writer for Drift-Free, Point-in-Time Research",
    },
    {
        "arxiv_id": "2607.22157v1",
        "title": "Learning on the Job: Continual Learning from Deployment Feedback for Frozen-Weights Agents",
    },
    {
        "arxiv_id": "2606.22681v1",
        "title": "Only Ask What You Don't Know: Grounded Delta Planning for Efficient Multi-step RAG",
    },
    {
        "arxiv_id": "2603.14170v1",
        "title": "Citation-Enforced RAG for Fiscal Document Intelligence: Cited, Explainable Knowledge Retrieval in Tax Compliance",
    },
    {
        "arxiv_id": "2609.01780v1",
        "title": "KGVoyager: Knowledge Graph Agnostic Question Answering via Agentic Navigation",
    },
]


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    output = root / "data" / "pdf"
    manifest_path = root / "data" / "incoming_research_papers.json"
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for item in PAPERS:
        arxiv_id = item["arxiv_id"]
        safe_id = arxiv_id.replace("/", "_")
        path = (
            output
            / f"{safe_id}_{item['title'][:70].replace(':', '').replace('/', '_')}.pdf"
        )
        if not path.exists():
            request = urllib.request.Request(
                f"https://export.arxiv.org/pdf/{arxiv_id}",
                headers={"User-Agent": "paper-rag-research-library/0.2"},
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read()
            if not data.startswith(b"%PDF"):
                raise RuntimeError(f"Unexpected response for {arxiv_id}")
            path.write_bytes(data)
        manifest.append(
            {
                **item,
                "path": str(path.relative_to(root)).replace("\\", "/"),
                "status": "downloaded",
            }
        )
        print(f"{arxiv_id}\t{path.name}")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(manifest_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
