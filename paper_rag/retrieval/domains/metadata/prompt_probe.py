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
from paper_rag.retrieval.domains.common.parser_client import PlanParserClient
from paper_rag.retrieval.domains.metadata.prompt import metadata_parser_system_prompt
from paper_rag.retrieval.domains.metadata.schema import validate_metadata_parse


DEFAULT_QUERIES = [
    "ResNet 和 Transformer 分别是哪一年发表的？",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe metadata route parser prompt outputs.")
    parser.add_argument("query", nargs="*", help="Optional single query. Defaults to the first DEFAULT_QUERIES item.")
    parser.add_argument("--project-root", type=Path, default=find_project_root(), help="Project root containing .env")
    parser.add_argument("--validated", action="store_true", help="Print validated parser payload instead of raw model output")
    args = parser.parse_args()

    settings = Settings.load(args.project_root)
    client = PlanParserClient.from_settings(settings)
    query = " ".join(args.query).strip() if args.query else (DEFAULT_QUERIES[0] if DEFAULT_QUERIES else "")
    if not query:
        print(json.dumps({"error": "No query provided. Add an item to DEFAULT_QUERIES or pass one on the command line."}, ensure_ascii=False, indent=2))
        return 1

    content = ""
    try:
        content = parse_once(client, query)
        if args.validated:
            print(json.dumps({"query": query, "metadata": validate_metadata_parse(strip_code_fence(content))}, ensure_ascii=False, indent=2))
        else:
            print(pretty_json_or_raw(content))
    except Exception as exc:
        payload: dict[str, str] = {"error": str(exc)}
        if content:
            payload["raw"] = content
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1
    return 0


def parse_once(client: PlanParserClient, query: str) -> str:
    return client.complete_json(metadata_parser_system_prompt(), query)


def pretty_json_or_raw(content: str) -> str:
    try:
        return json.dumps(json.loads(strip_code_fence(content)), ensure_ascii=False, indent=2)
    except json.JSONDecodeError:
        return content


def strip_code_fence(content: str) -> str:
    text = content.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


if __name__ == "__main__":
    sys.exit(main())
