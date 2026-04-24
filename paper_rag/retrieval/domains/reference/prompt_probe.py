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
from paper_rag.retrieval.domains.reference.prompt import reference_parser_prompt


DEFAULT_QUERIES = [
    # "哪些论文引用了ResNet",
    # "ResNet引用了哪些论文",
    # "哪些论文同时引用了ResNet和EfficientNet",
    # "哪些2019年之后的论文引用了BERT",
    # "ResNet引用的论文里哪些和ImageNet有关",
    # "ResNet 和 Resnxt 的参考文献分别有多少"
    # "哪些论文被Resnet或Transformer引用"
    "transformer引用了哪些CVPR的论文"
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe reference parser prompt outputs.")
    parser.add_argument("queries", nargs="*", help="Optional queries. Defaults to representative reference examples.")
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
    return client.complete_json(reference_parser_prompt(), query)


def pretty_json_or_raw(content: str) -> str:
    try:
        return json.dumps(json.loads(content), ensure_ascii=False, indent=2)
    except json.JSONDecodeError:
        return content


if __name__ == "__main__":
    sys.exit(main())
