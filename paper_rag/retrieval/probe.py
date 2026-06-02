"""统一的 planner/prompt/retrieval 手动调试入口。

这个文件把原来分散的 *_probe.py 收到一个 CLI 里，默认行为尽量贴近
“直接运行旧 probe 文件”：不传 query 时使用内置示例，只保留必要的 route/debug。

常用命令：

```powershell
python -m paper_rag probe --help
python -m paper_rag probe evidence
python -m paper_rag probe evidence --debug "ResNet 的结构是什么？"
python -m paper_rag probe planner --route content
python -m paper_rag probe prompt --route content
python -m paper_rag probe retrieval
```

子命令说明：

- `evidence`：执行完整 planner，查看最终 evidence。
- `planner`：绕过 top route，直接运行某个 domain router + planner。
- `prompt`：查看 parser LLM 输出；content route 会顺手写入 retrieval case。
- `retrieval`：固定读取 data/probe_cases/retrieval_probe_cases.json，复测 Dense/BM25/fused 召回。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from paper_rag.config import Settings
from paper_rag.corpus.context import CorpusContext
from paper_rag.corpus.records import paper_record_keys
from paper_rag.corpus.scope import resolve_scope_records
from paper_rag.retrieval.chunk_fusion import fuse_chunk_hits
from paper_rag.retrieval.dense.service import search_dense_chunks
from paper_rag.retrieval.plan import run_plan
from paper_rag.retrieval.route import RouteDecision
from paper_rag.retrieval.routes.common.parser_client import PlanParserClient
from paper_rag.retrieval.routes.content.context import context_unit
from paper_rag.retrieval.routes.content.planner import plan_body
from paper_rag.retrieval.routes.content.prompt import content_parser_system_prompt
from paper_rag.retrieval.routes.content.retrieval_query import build_content_retrieval_query
from paper_rag.retrieval.routes.content.router import build_content_decision
from paper_rag.retrieval.routes.content.schema import validate_content_parse
from paper_rag.retrieval.routes.metadata.planner import plan_metadata
from paper_rag.retrieval.routes.metadata.prompt import metadata_parser_system_prompt
from paper_rag.retrieval.routes.metadata.router import build_metadata_decision
from paper_rag.retrieval.routes.reference.planner import plan_reference
from paper_rag.retrieval.routes.reference.prompt import reference_parser_prompt
from paper_rag.retrieval.routes.reference.router import build_reference_decision
from paper_rag.retrieval.routes.top.prompt import top_route_prompt


EVIDENCE_DEFAULT_QUERIES = {
    "metadata": "BERT 是谁写的？",
    "reference": "哪些论文引用了 ResNet？",
    "content": "ResNet 的模型结构是什么？",
}

PLANNER_DEFAULT_QUERIES = {
    "metadata": "ResNet 和 Transformer 分别是哪一年发表的？",
    "reference": "哪些论文引用了 ResNet？",
    "content": "ResNet 的模型结构是什么？",
}

PROMPT_DEFAULT_QUERIES = {
    "top": "ResNet 和 Transformer 分别是哪一年发表的？",
    "metadata": "ResNet后续有哪些论文标题里带Resnet",
    "reference": "VIT之前，有哪些论文引用了Transformer",
    "content": "作者包含 Smith 或 Lee 的论文是否报告了消融实验？",
}

RETRIEVAL_TOP_K = 5


def main(argv: list[str] | None = None) -> int:
    parser = new_probe_parser(description="调试 Paper_RAG 的 parser、planner 和正文召回链路。")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="项目根目录，需包含 .env 或 data 目录")
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_probe_subcommands(subparsers)
    args = parser.parse_args(argv)
    return args.handler(args)


def add_probe_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("probe", help="调试 parser、planner 和正文召回内部结果", add_help=False)
    add_chinese_help(parser)
    probe_subparsers = parser.add_subparsers(dest="probe_command", required=True)
    add_probe_subcommands(probe_subparsers)


def new_probe_parser(*, description: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description, add_help=False)
    add_chinese_help(parser)
    return parser


def add_chinese_help(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-h", "--help", action="help", default=argparse.SUPPRESS, help="显示帮助信息并退出")


def add_probe_subcommands(subparsers: argparse._SubParsersAction) -> None:
    add_evidence_parser(subparsers)
    add_planner_parser(subparsers)
    add_prompt_parser(subparsers)
    add_retrieval_parser(subparsers)


def add_query(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("query", nargs="*", help="可选问题；不传时按 route 使用默认示例")


def add_evidence_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("evidence", help="查看最终 planner evidence 输出", add_help=False)
    add_chinese_help(parser)
    add_query(parser)
    parser.add_argument("--route", choices=["auto", "metadata", "reference", "content"], default="auto", help="默认 auto 走 top route；指定 route 时绕过 top")
    parser.add_argument("--debug", action="store_true", help="输出 planner 调试信息")
    parser.set_defaults(handler=handle_evidence)


def add_planner_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("planner", help="绕过 top router，直接调试某条 domain router + planner", add_help=False)
    add_chinese_help(parser)
    add_query(parser)
    parser.add_argument("--route", choices=["metadata", "reference", "content"], required=True, help="选择要直跑的 domain route")
    parser.add_argument("--debug", action="store_true", help="输出 planner 调试信息")
    parser.set_defaults(handler=handle_planner)


def add_prompt_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("prompt", help="查看某条 parser prompt 输出", add_help=False)
    add_chinese_help(parser)
    add_query(parser)
    parser.add_argument("--route", choices=["top", "metadata", "reference", "content"], required=True, help="选择要查看的 parser prompt")
    parser.set_defaults(handler=handle_prompt)


def add_retrieval_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("retrieval", help="用手写 content parser JSON 调试 Dense/BM25/fused 召回", add_help=False)
    add_chinese_help(parser)
    parser.set_defaults(handler=handle_retrieval)


def handle_evidence(args: argparse.Namespace) -> int:
    settings = Settings.load(args.project_root)
    query = selected_query(args.query, EVIDENCE_DEFAULT_QUERIES["content" if args.route == "auto" else args.route])
    payload = (
        run_plan(settings, query, debug=args.debug)
        if args.route == "auto"
        else run_domain_probe(settings, args.route, query, debug=args.debug)
    )
    print_json(payload)
    return 0


def handle_planner(args: argparse.Namespace) -> int:
    settings = Settings.load(args.project_root)
    query = selected_query(args.query, PLANNER_DEFAULT_QUERIES[args.route])
    print_json(run_domain_probe(settings, args.route, query, debug=args.debug))
    return 0


def handle_prompt(args: argparse.Namespace) -> int:
    settings = Settings.load(args.project_root)
    client = PlanParserClient.from_settings(settings)
    query = selected_query(args.query, PROMPT_DEFAULT_QUERIES[args.route])
    content = ""
    try:
        content = client.complete_json(prompt_for_route(args.route), query)
        if args.route != "content":
            print(pretty_json_or_raw(content))
            return 0

        # 保留旧 content prompt probe 的核心效果：校验 parser_result，并沉淀为 retrieval case。
        parser_result = validate_content_parse(strip_code_fence(content), query)
        case_path = default_probe_cases_path(settings)
        saved = upsert_cases(case_path, [{"query": query, "parser_result": parser_result}])
        print_json({"results": [{"query": query, "status": "ok", "parser_result": parser_result}], "cases_path": str(case_path), "saved": saved})
    except Exception as exc:
        payload: dict[str, Any] = {"error": str(exc)}
        if content:
            payload["raw"] = content
        print_json(payload)
        return 1
    return 0


def handle_retrieval(args: argparse.Namespace) -> int:
    settings = Settings.load(args.project_root)
    cases_path = default_probe_cases_path(settings)
    cases = load_cases(cases_path)
    if not cases:
        print_json({"error": "没有找到匹配的 retrieval case", "cases": str(cases_path)})
        return 1
    payload = [run_retrieval_case(settings, case, top_k=RETRIEVAL_TOP_K) for case in cases]
    print_json(payload)
    return 0


def run_domain_probe(
    settings: Settings,
    route_name: str,
    query: str,
    *,
    debug: bool,
) -> dict[str, Any]:
    """绕过 top router，直接检查某条 domain 的 parser 归一化和 planner 输出。"""
    warnings: list[str] = []
    try:
        route = build_domain_decision(settings, route_name, query, warnings)
        return plan_domain(settings, route, warnings, debug=debug)
    except Exception as exc:
        return {"query": query, "route": route_name, "status": "probe_failed", "error": str(exc), "warnings": warnings}


def build_domain_decision(settings: Settings, route_name: str, query: str, warnings: list[str]) -> RouteDecision:
    base = RouteDecision(route=route_name, query=query, parse_status="ok")
    if route_name == "metadata":
        return build_metadata_decision(settings, base, warnings)
    if route_name == "reference":
        return build_reference_decision(settings, base, warnings)
    if route_name == "content":
        return build_content_decision(settings, base, warnings)
    raise ValueError(f"不支持的 route：{route_name}")


def plan_domain(settings: Settings, route: RouteDecision, warnings: list[str], *, debug: bool) -> dict[str, Any]:
    if route.route == "metadata":
        return plan_metadata(settings, route, warnings, debug=debug)
    if route.route == "reference":
        return plan_reference(settings, route, warnings, debug=debug)
    if route.route == "content":
        return plan_body(settings, route, warnings, debug=debug)
    raise ValueError(f"不支持的 route：{route.route}")


def prompt_for_route(route: str) -> str:
    if route == "top":
        return top_route_prompt()
    if route == "metadata":
        return metadata_parser_system_prompt()
    if route == "reference":
        return reference_parser_prompt()
    if route == "content":
        return content_parser_system_prompt()
    raise ValueError(f"不支持的 prompt route：{route}")


class StaticContentParser:
    """把手写 JSON 伪装成 content parser，绕过 prompt/LLM"""

    def __init__(self, parser_result: dict[str, Any]) -> None:
        self.parser_result = parser_result

    def parse_content(self, query: str) -> dict[str, Any]:
        return validate_content_parse(self.parser_result, query)


def run_retrieval_case(settings: Settings, case: dict[str, Any], *, top_k: int) -> dict[str, Any]:
    """用静态 parser_result 复现 content 召回，避免每次调试都调用 parser LLM。"""
    warnings: list[str] = []
    query = str(case.get("query") or "").strip()
    parser_result = case.get("parser_result")
    if not query or not isinstance(parser_result, dict):
        return {"status": "invalid_case", "warnings": ["case 需要 query 和 parser_result"]}
    try:
        corpus = CorpusContext(settings)
        route = build_content_decision(
            settings,
            RouteDecision(route="content", query=query, parse_status="ok"),
            warnings,
            plan_parser=StaticContentParser(parser_result),
            corpus=corpus,
        )
        retrieval_query = build_content_retrieval_query(settings, route, warnings)
        scope_records, _ = resolve_scope_records(
            settings,
            route.paper_semantic,
            route.filters,
            route.paper_groups,
            route.group_mode,
            corpus=corpus,
        )
        chunk_documents = corpus.content_chunks_for_records(scope_records)
        chunk_documents_by_id = {chunk.chunk_id: chunk for chunk in chunk_documents}
        dense_hits = dense_results(settings, retrieval_query["dense_query"], paper_record_keys(scope_records), warnings)
        bm25_hits = corpus.bm25_index.search_many(
            retrieval_query["bm25_queries"],
            settings.plan_bm25_top_k,
            allowed_chunk_ids=[chunk.chunk_id for chunk in chunk_documents],
        )
        fused_hits = fuse_chunk_hits(chunk_documents_by_id, dense_hits, bm25_hits)
        contexts = [context_unit(settings, hit, settings.plan_block_window) for hit in fused_hits[:top_k]]
        return {
            "query": query,
            "status": "ok",
            "parser_result": parser_result,
            "resolved_scope": {
                "paper_semantic": route.paper_semantic,
                "filters": route.filters,
                "paper_groups": route.paper_groups,
                "papers": [record.get("title") for record in scope_records],
                "chunk_count": len(chunk_documents),
            },
            "retrieval_query": {
                "dense_query": retrieval_query["dense_query"],
                "bm25_queries": retrieval_query["bm25_queries"],
                "source_terms": retrieval_query.get("source_terms") or {},
            },
            "dense_hits": [compact_dense_hit(hit) for hit in dense_hits[:top_k]],
            "bm25_hits": [compact_bm25_hit(hit) for hit in bm25_hits[:top_k]],
            "fused_contexts": [compact_context(context) for context in contexts],
            "expected_terms": case.get("expected_terms") or [],
            "warnings": warnings,
        }
    except Exception as exc:
        return {"query": query, "status": "probe_failed", "error": str(exc), "warnings": warnings}


def dense_results(settings: Settings, dense_query: str, paper_ids: list[str], warnings: list[str]) -> list[Any]:
    try:
        return search_dense_chunks(settings, dense_query, paper_ids=paper_ids)
    except Exception as exc:
        warnings.append(f"Dense 召回失败：{exc}；继续展示 BM25 结果")
        return []


def compact_dense_hit(hit: Any) -> dict[str, Any]:
    return {
        "score": getattr(hit, "score", None),
        "chunk_id": getattr(hit, "chunk_id", ""),
        "title": getattr(hit, "title", ""),
        "section": getattr(hit, "section_path_text", ""),
        "pages": getattr(hit, "pages_text", ""),
        "text": snippet(getattr(hit, "text", "")),
    }


def compact_bm25_hit(hit: Any) -> dict[str, Any]:
    chunk_document = (hit.payload or {}).get("chunk_document")
    return {
        "score": hit.score,
        "chunk_id": hit.doc_id,
        "title": getattr(chunk_document, "title", ""),
        "section": getattr(chunk_document, "section_path_text", ""),
        "pages": getattr(chunk_document, "pages_text", ""),
        "text": snippet(hit.text),
    }


def compact_context(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": context.get("chunk_id"),
        "title": context.get("title"),
        "section_path": context.get("section_path"),
        "pages": context.get("pages"),
        "text": snippet(context.get("chunk_text") or context.get("text") or "", limit=700),
    }


def selected_query(parts: list[str], default: str) -> str:
    return " ".join(parts).strip() if parts else default


def default_probe_cases_path(settings: Settings) -> Path:
    return settings.data_dir / "probe_cases" / "retrieval_probe_cases.json"


def load_cases(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict):
        payload = payload.get("cases") or []
    if not isinstance(payload, list):
        raise ValueError("case 文件必须是数组，或形如 {'cases': [...]} 的对象")
    return [case for case in payload if isinstance(case, dict)]


def upsert_cases(path: Path, new_cases: list[dict[str, Any]]) -> int:
    if not new_cases:
        return 0
    cases = load_cases(path) if path.exists() and path.read_text(encoding="utf-8-sig").strip() else []
    by_query = {str(case.get("query") or ""): case for case in cases if case.get("query")}
    saved = 0
    for case in new_cases:
        query = str(case.get("query") or "")
        if not query:
            continue
        by_query[query] = {"query": query, "parser_result": case["parser_result"]}
        saved += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(by_query.values()), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return saved


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


def snippet(text: str, limit: int = 360) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + "..."


def print_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
