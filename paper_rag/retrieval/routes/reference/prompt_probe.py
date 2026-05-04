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
from paper_rag.retrieval.routes.common.parser_client import PlanParserClient
from paper_rag.retrieval.routes.reference.prompt import reference_parser_prompt


DEFAULT_QUERIES = [
    "VIT之前，有哪些论文引用了Transformer"  ,
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe reference parser prompt outputs.")
    parser.add_argument("query", nargs="*", help="Optional single query. Defaults to the first DEFAULT_QUERIES item.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT, help="Project root containing .env")
    args = parser.parse_args()

    settings = Settings.load(args.project_root)
    client = PlanParserClient.from_settings(settings)
    query = " ".join(args.query).strip() if args.query else (DEFAULT_QUERIES[0] if DEFAULT_QUERIES else "")
    if not query:
        print(json.dumps({"error": "No query provided. Add an item to DEFAULT_QUERIES or pass one on the command line."}, ensure_ascii=False, indent=2))
        return 1
    try:
        print(pretty_json_or_raw(client.complete_json(reference_parser_prompt(), query)))
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2))
        return 1
    return 0


def pretty_json_or_raw(content: str) -> str:
    try:
        return json.dumps(json.loads(content), ensure_ascii=False, indent=2)
    except json.JSONDecodeError:
        return content


if __name__ == "__main__":
    sys.exit(main())
