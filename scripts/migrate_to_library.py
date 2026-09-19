"""Migrate the current legacy corpus and the downloaded research papers."""

from __future__ import annotations

import json
from pathlib import Path

from paper_rag.library.catalog import Catalog
from paper_rag.library.ingestion import import_document, migrate_legacy
from paper_rag.library.settings import LibrarySettings
from paper_rag.library.common import json_lines


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    settings = LibrarySettings.load(root)
    with Catalog(settings, writable=True) as catalog:
        legacy = migrate_legacy(catalog)
        incoming = json.loads(
            (root / "data" / "incoming_research_papers.json").read_text(
                encoding="utf-8"
            )
        )
        new = []
        manifest = json_lines(root / "data" / "manifest.jsonl")
        for item in incoming:
            path = root / item["path"]
            # Legacy ingestion renames PDFs; use its explicit mapping, never a
            # fuzzy filename match that could attach metadata to the wrong paper.
            if not path.is_file():
                matches = [row for row in manifest if row.get("title") == item["title"]]
                if len(matches) == 1:
                    path = Path(matches[0]["pdf_path"])
                    item["path"] = path.relative_to(root).as_posix()
            try:
                metadata = {
                    k: item[k]
                    for k in ("title", "authors", "year", "abstract", "arxiv_id")
                    if k in item
                }
                new.append(import_document(catalog, path, metadata=metadata))
            except Exception as exc:
                new.append(
                    {
                        "path": item["path"],
                        "status": "error",
                        "error": type(exc).__name__,
                        "message": str(exc),
                    }
                )
        report = {"legacy": legacy, "new": new}
        (root / "data" / "incoming_research_papers.json").write_text(
            json.dumps(incoming, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (settings.home / "migration-report.json").parent.mkdir(
            parents=True, exist_ok=True
        )
        (settings.home / "migration-report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "legacy_ok": len(legacy["documents"]),
                    "legacy_failures": len(legacy["failures"]),
                    "new": new,
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
