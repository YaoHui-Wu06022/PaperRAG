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
from paper_rag.retrieval.domains.reference.prompt import reference_parser_prompt


DEFAULT_QUERIES = [

]


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe reference parser prompt outputs.")
    parser.add_argument("queries", nargs="*", help="Optional queries. Defaults to representative reference examples.")
    parser.add_argument("--project-root", type=Path, default=find_project_root(), help="Project root containing .env")
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
    return client.complete_json(reference_parser_prompt(), query)


def parse_batch(client: PlanParserClient, queries: list[str]) -> str:
    return complete_batch_json(client, batch_system_prompt(reference_parser_prompt()), batch_query(queries))


def batch_system_prompt(system_prompt: str) -> str:
    return (
        system_prompt
        + "\n\n批量测试模式覆盖说明：\n"
        + "- 只返回一个 JSON 数组。\n"
        + "- 数组里的每一项必须严格符合上面定义的单问题 schema。\n"
    )


def batch_query(queries: list[str]) -> str:
    return (
        "请按顺序解析下面所有问题，并只返回一个 JSON 数组。\n"
        "数组里的每一项必须是一个单问题 schema 对象，只包含 intent、direction、anchors、anchor_mode、filters。\n"
        + "\n".join(f"{index}. {query}" for index, query in enumerate(queries, start=1))
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
