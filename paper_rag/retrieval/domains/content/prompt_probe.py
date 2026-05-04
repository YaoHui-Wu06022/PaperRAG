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
from paper_rag.retrieval.domains.common.parser_client import PlanParserClient, chat_completion_content
from paper_rag.retrieval.domains.content.prompt import content_parser_system_prompt


DEFAULT_QUERIES = [
    " 作者包含 Smith 或 Lee 的论文是否报告了消融实验？"
]

def main() -> int:
    parser = argparse.ArgumentParser(description="Probe content parser prompt outputs.")
    parser.add_argument("queries", nargs="*", help="Optional queries. Defaults to representative content examples.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT, help="Project root containing .env")
    args = parser.parse_args()

    settings = Settings.load(args.project_root)
    client = PlanParserClient.from_settings(settings)
    queries = args.queries or DEFAULT_QUERIES
    if not queries:
        print(json.dumps({"error": "No queries provided. Add items to DEFAULT_QUERIES or pass queries on the command line."}, ensure_ascii=False, indent=2))
        return 1
    try:
        print(pretty_json_or_raw(parse_batch(client, queries)))
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2))
        return 1
    return 0


def parse_once(client: PlanParserClient, query: str) -> str:
    return client.complete_json(content_parser_system_prompt(), query)


def parse_batch(client: PlanParserClient, queries: list[str]) -> str:
    return complete_batch_json(client, batch_system_prompt(content_parser_system_prompt()), batch_query(queries))


def batch_system_prompt(system_prompt: str) -> str:
    return (
        system_prompt
    )


def batch_query(queries: list[str]) -> str:
    return (
        "\n".join(f"{index}. {query}" for index, query in enumerate(queries, start=1))
    )


def complete_batch_json(client: PlanParserClient, system_prompt: str, query: str) -> str:
    payload = {
        "model": client.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        "temperature": 0,
    }
    return chat_completion_content(client.chat_completion(payload))


def pretty_json_or_raw(content: str) -> str:
    try:
        return json.dumps(json.loads(content), ensure_ascii=False, indent=2)
    except json.JSONDecodeError:
        return content


if __name__ == "__main__":
    sys.exit(main())
