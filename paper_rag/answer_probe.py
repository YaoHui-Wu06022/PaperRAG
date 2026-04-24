from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def find_project_root() -> Path:
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / "pyproject.toml").exists() or (parent / ".env").exists():
            return parent
    return Path.cwd()


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper_rag.answer import run_ask
from paper_rag.config import Settings


DEFAULT_QUERIES = [
    # "BERT是谁写的",
    # "哪些论文引用了ResNet",
    # "Resnet之后在CVPR发表论文有哪些",
    "哪些论文引用resnet"
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe paper-rag ask outputs for custom questions.")
    parser.add_argument("queries", nargs="*", help="Optional questions. Defaults to representative ask examples.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT, help="Project root containing .env")
    parser.add_argument("--debug", action="store_true", help="Print raw plan JSON after the rendered answer")
    args = parser.parse_args()

    settings = Settings.load(args.project_root)
    queries = args.queries or DEFAULT_QUERIES
    for index, query in enumerate(queries, start=1):
        print(f"## {index}. {query}")
        try:
            result = run_ask(settings, query)
            print(result.answer)
            if result.provenance:
                print()
                print("证据:")
                for item in result.provenance:
                    print(f"- {item}")
            for warning in result.warnings:
                print(f"提示: {warning}")
            if args.debug:
                print()
                print(json.dumps(result.plan, ensure_ascii=False, indent=2))
        except Exception as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
