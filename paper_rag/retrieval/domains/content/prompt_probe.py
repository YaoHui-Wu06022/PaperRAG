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
        + "\n\n批量测试模式覆盖说明：\n"
        + "- 只返回一个 JSON 数组。\n"
        + "- 数组里的每一项必须严格符合上面定义的单问题 schema。\n"
    )


def batch_query(queries: list[str]) -> str:
    return (
        "请按顺序解析下面所有问题，并只返回一个 JSON 数组。\n"
        "数组里的每一项必须是一个单问题 schema 对象，只包含 intent、anchors、compare_objects、objects、filters。\n"
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
