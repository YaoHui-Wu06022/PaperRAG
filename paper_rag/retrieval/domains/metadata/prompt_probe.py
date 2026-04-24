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


DEFAULT_QUERIES = [
    # "在2016-2025年不在ACM发表的论文有哪些",
    # "找出 2015 年之后的论文",
    # "列出不在 2015 年到 2018 年之间的论文",
    # "哪些论文不是 He Kaiming 写的",
    # "找一下标题是 Transformer 的论文",
    # "ResNet 是哪一年发表的",
    # "Attention is All You Need之后有哪些不在2019年以前的论文",
    # "Word2Vec之后、BERT 之前有哪些论文",
    "题目中包含 BERT 的论文有哪些"
]



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
    return client.complete_json(metadata_parser_system_prompt(), query)


def pretty_json_or_raw(content: str) -> str:
    try:
        return json.dumps(json.loads(content), ensure_ascii=False, indent=2)
    except json.JSONDecodeError:
        return content

if __name__ == "__main__":
    sys.exit(main())
