"""Evaluate the reviewed benchmark without any answer-generation API."""

import json
import sys
from pathlib import Path
from paper_rag.library.catalog import Catalog
from paper_rag.library.contracts import EvalRequest
from paper_rag.library.evaluation import evaluate
from paper_rag.library.settings import LibrarySettings


def main():
    settings = LibrarySettings.load(Path.cwd())
    mode = sys.argv[1] if len(sys.argv) > 1 else "lexical"
    with Catalog(settings) as cat:
        report = evaluate(
            cat,
            EvalRequest(
                cases_path=str(settings.home / "eval/reviewed-facts.json"),
                output_path=str(settings.home / "eval" / f"{mode}.json"),
                mode=mode,
                top_k=10,
            ),
        )
        print(
            json.dumps(
                {
                    "mode": mode,
                    "cases": len(report["cases"]),
                    "summary": report["summary"],
                    "degraded_cases": report["degraded_cases"],
                    "first_warnings": next(
                        (r["warnings"] for r in report["cases"] if r["warnings"]), []
                    ),
                    "release_blocked": report["release_blocked"],
                }
            )
        )


if __name__ == "__main__":
    main()
