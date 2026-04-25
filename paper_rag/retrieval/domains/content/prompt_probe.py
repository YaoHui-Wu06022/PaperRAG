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
from paper_rag.retrieval.domains.content.prompt import content_parser_system_prompt


DEFAULT_QUERIES = [
    # "这篇论文用了什么损失函数？"
    # "《Attention Is All You Need》这篇论文的方法是什么？"
    # "ResNet和DenseNet有什么区别？"
    # "ResNet和BERT之间有哪些论文用了attention机制？"
    # "2018年以后CVPR的目标检测论文用了哪些数据集？"
    # "哪些ResNet之后的论文用了transformer方法？"
    # "Resnet 这篇论文讲了什么内容"
    "2018年以后CVPR的目标检测论文用了哪些数据集？",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe content parser prompt outputs.")
    parser.add_argument("queries", nargs="*", help="Optional queries. Defaults to representative content examples.")
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
    return client.complete_json(content_parser_system_prompt(), query)


def pretty_json_or_raw(content: str) -> str:
    try:
        return json.dumps(json.loads(content), ensure_ascii=False, indent=2)
    except json.JSONDecodeError:
        return content


if __name__ == "__main__":
    sys.exit(main())
