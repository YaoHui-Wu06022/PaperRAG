"""Read-only live verification with a saved report and exact evidence resolution."""

import json
import subprocess
import sys
from pathlib import Path

from paper_rag.library.catalog import Catalog
from paper_rag.library.contracts import DoctorRequest, SearchRequest, ReadRequest
from paper_rag.library.service import doctor
from paper_rag.library.settings import LibrarySettings
from paper_rag.library.search import search
from paper_rag.library.reading import read


def main():
    settings = LibrarySettings.load(Path.cwd())
    report = {"doctor": doctor(settings, DoctorRequest(remote=True)), "queries": []}
    with Catalog(settings) as cat:
        for query in (
            "scaled dot product attention",
            "Grounded Delta Planning",
            "contrastive learning temperature",
        ):
            result = search(cat, SearchRequest(query=query))
            assert result["items"] and not result["degraded"], result["warnings"]
            for item in result["items"]:
                for ref in item["evidence_ids"]:
                    evidence = read(cat, ReadRequest(evidence_id=ref))
                    assert (
                        evidence["fragments"]
                        and not evidence["fragments"][0]["historical"]
                    )
            report["queries"].append({"query": query, "result": result})
    for args, request in (
        (["papers", "count"], {}),
        (["wiki", "lint"], {}),
        (["search"], {"query": "attention", "mode": "lexical"}),
    ):
        process = subprocess.run(
            [sys.executable, "-X", "utf8", "-m", "paper_rag", *args, "--json"],
            input=json.dumps(request),
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=True,
        )
        result = json.loads(process.stdout)
        assert result["schema_version"] == 1
    destination = settings.home / "live-verification.json"
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "doctor": report["doctor"],
                "queries_checked": len(report["queries"]),
                "cli_checked": 3,
                "report": str(destination),
            }
        )
    )


if __name__ == "__main__":
    main()
