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

from paper_rag.config import Settings
from paper_rag.retrieval.planner import run_plan


DEFAULT_QUERIES = [
    "Resnet之后在CVPR发表论文有哪些",
    # "ResNet或transformer引用了哪些论文",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe paper-rag plan outputs for custom questions.")
    parser.add_argument("queries", nargs="*", help="Optional questions. Defaults to representative plan examples.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT, help="Project root containing .env")
    parser.add_argument("--compact", action="store_true", help="Print each plan JSON on one line")
    args = parser.parse_args()

    settings = Settings.load(args.project_root)
    queries = args.queries or DEFAULT_QUERIES
    for index, query in enumerate(queries, start=1):
        print(f"## {index}. {query}")
        try:
            plan = run_plan(settings, query)
        except Exception as exc:
            plan = {"error": str(exc)}
        indent = None if args.compact else 2
        print(json.dumps(plan, ensure_ascii=False, indent=indent))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
