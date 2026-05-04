from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


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
from paper_rag.retrieval.domains.content.schema import validate_content_parse


DEFAULT_QUERIES = [
    " 作者包含 Smith 或 Lee 的论文是否报告了消融实验？"
]
DEFAULT_CASES_PATH = Path(__file__).with_name("retrieval_probe_cases.json")


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe content parser prompt outputs.")
    parser.add_argument("queries", nargs="*", help="Optional queries. Each quoted argument is treated as one query.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT, help="Project root containing .env")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH, help="Where to append retrieval probe cases")
    parser.add_argument("--no-save", action="store_true", help="Only print parser outputs; do not update cases JSON")
    args = parser.parse_args()

    settings = Settings.load(args.project_root)
    client = PlanParserClient.from_settings(settings)
    queries = args.queries or DEFAULT_QUERIES
    if not queries:
        print(json.dumps({"error": "No queries provided. Add items to DEFAULT_QUERIES or pass queries on the command line."}, ensure_ascii=False, indent=2))
        return 1

    results = [parse_query(client, query) for query in queries]
    saved = 0
    if not args.no_save:
        saved = upsert_cases(args.cases, [
            {"query": result["query"], "parser_result": result["parser_result"]}
            for result in results
            if result.get("status") == "ok"
        ])
    print(json.dumps({
        "results": results,
        "cases_path": str(args.cases),
        "saved": saved,
    }, ensure_ascii=False, indent=2))
    return 0 if all(result.get("status") == "ok" for result in results) else 1


def parse_query(client: PlanParserClient, query: str) -> dict[str, Any]:
    """调用 content prompt，并把输出按 schema 校验成 retrieval_probe 可用的 parser_result。"""
    try:
        content = client.complete_json(content_parser_system_prompt(), query)
        parser_result = validate_content_parse(content, query)
        return {"query": query, "status": "ok", "parser_result": parser_result}
    except Exception as exc:
        return {"query": query, "status": "parse_failed", "error": str(exc)}


def upsert_cases(path: Path, new_cases: list[dict[str, Any]]) -> int:
    """按 query 去重写入 retrieval_probe_cases.json；同 query 再跑会更新结果。"""
    if not new_cases:
        return 0
    cases = load_cases(path)
    by_query = {str(case.get("query") or ""): case for case in cases if case.get("query")}
    saved = 0
    for case in new_cases:
        query = str(case.get("query") or "")
        if not query:
            continue
        by_query[query] = {
            "query": query,
            "parser_result": case["parser_result"],
        }
        saved += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(by_query.values()), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return saved


def load_cases(path: Path) -> list[dict[str, Any]]:
    """读取已有 cases；不存在或为空时返回空列表。"""
    if not path.exists():
        return []
    content = path.read_text(encoding="utf-8-sig")
    if not content.strip():
        return []
    payload = json.loads(content)
    if isinstance(payload, dict):
        payload = payload.get("cases") or []
    if not isinstance(payload, list):
        raise ValueError("retrieval probe cases must be a JSON list")
    return [case for case in payload if isinstance(case, dict)]


if __name__ == "__main__":
    sys.exit(main())
