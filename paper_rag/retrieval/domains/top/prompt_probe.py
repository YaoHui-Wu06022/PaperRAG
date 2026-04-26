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
from paper_rag.retrieval.domains.top.prompt import top_router_prompt
from paper_rag.retrieval.domains.top.schema import validate_top_parse


DEFAULT_QUERIES = [
    # content：论文名/模型名保留在 extract_query，不做 title filter
    # "列出 Transformer 论文里的关键模块",
    # "总结 BERT 的预训练任务",
    # "比较 ResNet 和 DenseNet 的连接方式",
    # "解释 ImageNet 在视觉模型训练中的作用",
    # "Transformer 为什么不需要 RNN？",
    # "列举 EfficientNet 的缩放策略",
    # "总结 GPT 和 BERT 的区别",
    # "分析 ResNet 的残差连接为什么有效",
    "标题是 Attention Is All You Need 的论文结构是什么？",
    # "题目中包含 BERT 的论文贡献是什么？",
    # "Transformer 相关论文的实验结果怎么样？",
    # "关于 diffusion 的论文有哪些采样方法？",

    # content：author/year/venue 仍然可以作为公共 filters
    # "作者为 He 的论文中，ResNet 的结构有什么特点？",
    # "作者不是 He Kaiming 的论文中，ResNet 有什么不足？",
    # "2017年以后发表的论文中，Transformer 的注意力机制有什么变化？",
    # "2015到2020年之间的论文中，BERT 的训练策略有什么变化？",
    # "CVPR 上的论文里，ResNet 方法有什么改进？",
    # "不在 CVPR 发表的论文中，ResNet 有什么不足？",
    # "发表在 ACL 或 EMNLP 的论文中，BERT 的预训练任务有什么区别？",
    # "作者为 Vaswani 且发表在 NeurIPS 的论文中，Transformer 的结构是什么？",

    # content：相对年份，论文名只作为 year interval 边界
    # "ResNet 以后，CNN 结构有什么发展？",
    # "ResNet 之前，CNN 是怎么解决梯度消失问题的？",
    # "ResNet 和 BERT 之间的论文里，模型结构有什么变化？",
    # "Transformer 之后，注意力机制有什么发展？",
    # "BERT 以前，语言模型的预训练方式有哪些？",
    # "ImageNet 之后，视觉模型的训练范式有什么变化？",

    # reference：anchor 保留在 extract_query，不做 title filter
    # "ImageNet 被哪些论文引用过？",
    # "哪些论文引用了 ImageNet？",
    # "ResNet 引用了哪些论文？",
    # "ResNet 和 EfficientNet 分别引用了哪些论文？",
    # "哪些论文同时引用了 ResNet 和 DenseNet？",
    # "BERT 有没有引用 Transformer？",
    # "AlexNet 和 ImageNet 分别被引用了多少次？",
    # "SENet 和 ResNet 分别被哪些论文引用？",
    # "Center Loss 被哪些论文引用？",
    # "ResNet 的参考文献有哪些？",

    # reference：author/year/venue filters + reference target
    # "2018年以后哪些论文引用了 BERT？",
    # "2015到2020年之间哪些论文引用了 Transformer？",
    # "ACL 上哪些论文引用了 Transformer？",
    # "CVPR 论文里哪些引用了 Transformer 或 BERT？",
    # "作者为 Vaswani 的论文引用了哪些文献？",
    # "不是 CVPR 发表的论文里，哪些引用了 ResNet？",
    # "发表在 ICCV 或 ECCV 的论文中，哪些引用了 ImageNet？",
    # "作者不是 He Kaiming 的论文里，哪些引用了 ResNet？",

    # "题目中带有 ResNet 的论文有哪些？",

]


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe top route parser prompt outputs.")
    parser.add_argument("queries", nargs="*", help="Optional queries. Defaults to representative top-route examples.")
    parser.add_argument("--project-root", type=Path, default=find_project_root(), help="Project root containing .env")
    parser.add_argument("--validated", action="store_true", help="Print schema-validated payload instead of raw model output")
    args = parser.parse_args()

    settings = Settings.load(args.project_root)
    client = PlanParserClient.from_settings(settings)
    queries = args.queries or DEFAULT_QUERIES
    if not queries:
        print(json.dumps({"error": "No queries provided. Add items to DEFAULT_QUERIES or pass queries on the command line."}, ensure_ascii=False, indent=2))
        return 1
    try:
        content = parse_batch(client, queries)
        if args.validated:
            print(json.dumps(validate_batch(content, queries), ensure_ascii=False, indent=2))
        else:
            print(pretty_json_or_raw(content))
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2))
        return 1
    return 0


def parse_once(client: PlanParserClient, query: str) -> str:
    return client.complete_json(top_router_prompt(), query)


def parse_batch(client: PlanParserClient, queries: list[str]) -> str:
    return complete_batch_json(client, batch_system_prompt(top_router_prompt()), batch_query(queries))


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
        "数组里的每一项必须是一个单问题 schema 对象，只包含 router、extract_query、filters。\n"
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


def validate_batch(content: str, queries: list[str]) -> dict:
    results = json.loads(content)
    if not isinstance(results, list):
        raise ValueError("Batch response must be a JSON array")
    validated = []
    for index, item in enumerate(results, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Batch item {index} is not an object")
        query = queries[index - 1]
        validated.append({
            "index": index,
            "query": query,
            "result": validate_top_parse(item, query),
        })
    return {"results": validated}


def pretty_json_or_raw(content: str) -> str:
    try:
        return json.dumps(json.loads(content), ensure_ascii=False, indent=2)
    except json.JSONDecodeError:
        return content


if __name__ == "__main__":
    sys.exit(main())
