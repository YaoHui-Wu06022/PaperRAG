"""Resolve arXiv metadata by exact ID, preserving downloaded source identity."""

import json
import urllib.request
import xml.etree.ElementTree as ET
import time
from pathlib import Path

from paper_rag.library.catalog import Catalog
from paper_rag.library.ingestion import import_document
from paper_rag.library.settings import LibrarySettings


def main():
    root = Path.cwd()
    path = root / "data/incoming_research_papers.json"
    items = json.loads(path.read_text(encoding="utf-8"))
    ns = {"a": "http://www.w3.org/2005/Atom"}
    for item in items:
        time.sleep(3)
        req = urllib.request.Request(
            "https://export.arxiv.org/api/query?id_list=" + item["arxiv_id"],
            headers={"User-Agent": "paper-rag/0.2"},
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                feed = ET.fromstring(response.read())
        except Exception as exc:
            print(item["arxiv_id"], type(exc).__name__, flush=True)
            continue
        entry = feed.find("a:entry", ns)
        identifier = entry.findtext("a:id", namespaces=ns).rsplit("/", 1)[-1]
        if identifier != item["arxiv_id"]:
            raise ValueError("arXiv identity mismatch")
        title = " ".join(entry.findtext("a:title", namespaces=ns).split())
        item.update(
            title=title,
            authors=[e.text for e in entry.findall("a:author/a:name", ns)],
            year={
                "preprint_year": int(entry.findtext("a:published", namespaces=ns)[:4]),
                "publish_year": None,
            },
            abstract=" ".join(entry.findtext("a:summary", namespaces=ns).split()),
        )
        print(identifier, title, flush=True)
        path.write_text(
            json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    path.write_text(
        json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with Catalog(LibrarySettings.load(root), writable=True) as cat:
        for item in items:
            import_document(cat, root / item["path"], metadata=item)


if __name__ == "__main__":
    main()
