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
from paper_rag.retrieval.plan.domains.metadata.parser import PlanParserClient, chat_completion_content
from paper_rag.retrieval.plan.domains.metadata.prompt import metadata_parser_system_prompt


DEFAULT_QUERIES = [
    "在2016-2025年不在ACM发表的论文有哪些"
]
# DEFAULT_QUERIES = [
#     "Attention is All You Need之后有哪些不在2019年以前的论文",
#     "ResNet 是哪一年发表的",
#     "找一下标题里包含 Transformer 的论文",
#     "Word2Vec 之后、BERT 之前有哪些论文",
#     "哪些论文不是 He Kaiming 写的"
# ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe metadata parser prompt outputs for anchored year queries.")
    parser.add_argument("queries", nargs="*", help="Optional queries. Defaults to anchored year-range examples.")
    parser.add_argument("--project-root", type=Path, default=find_project_root(), help="Project root containing .env")
    args = parser.parse_args()

    settings = Settings.load(args.project_root)
    client = PlanParserClient.from_settings(settings)
    queries = args.queries or DEFAULT_QUERIES
    for index, query in enumerate(queries, start=1):
        print(f"## {index}. {query}")
        try:
            content = parse_once(client, query)
        except Exception as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2))
        else:
            print(pretty_json_or_raw(content))
        print()
    return 0


def parse_once(client: PlanParserClient, query: str) -> str:
    payload = {
        "model": client.model,
        "messages": [
            {"role": "system", "content": metadata_parser_system_prompt()},
            {"role": "user", "content": query},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    return chat_completion_content(client.chat_completion(payload))


def pretty_json_or_raw(content: str) -> str:
    try:
        return json.dumps(json.loads(content), ensure_ascii=False, indent=2)
    except json.JSONDecodeError:
        return content

if __name__ == "__main__":
    sys.exit(main())
