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
from paper_rag.retrieval.domains.common.errors import PlanParseError
from paper_rag.retrieval.domains.common.parser_client import PlanParserClient
from paper_rag.retrieval.domains.common.schema import validate_paper_filters
from paper_rag.retrieval.domains.top.prompt import top_router_prompt


TOP_ROUTERS = {"metadata", "reference", "content", "unclear"}
TOP_FILTER_OPS_BY_FIELD = {
    "author": {"contains"},
    "year": {"=", "interval"},
    "venue": {"=", "in"},
    "title": {"contains"},
    "paper": {"=", "in"},
}

DEFAULT_QUERIES = [
    "哪些论文提到图像处理",
]

def main() -> int:
    parser = argparse.ArgumentParser(description="Probe top route parser prompt outputs.")
    parser.add_argument("query", nargs="*", help="Optional single query. Defaults to the first DEFAULT_QUERIES item.")
    parser.add_argument("--project-root", type=Path, default=find_project_root(), help="Project root containing .env")
    parser.add_argument("--validated", action="store_true", help="Print probe-validated payload instead of raw model output")
    args = parser.parse_args()

    settings = Settings.load(args.project_root)
    client = PlanParserClient.from_settings(settings)
    query = " ".join(args.query).strip() if args.query else (DEFAULT_QUERIES[0] if DEFAULT_QUERIES else "")
    if not query:
        print(json.dumps({"error": "No query provided. Add an item to DEFAULT_QUERIES or pass one on the command line."}, ensure_ascii=False, indent=2))
        return 1

    content = ""
    try:
        content = parse_once(client, query)
        if args.validated:
            print(json.dumps({"query": query, "scheme": validate_probe_top_parse(content)}, ensure_ascii=False, indent=2))
        else:
            print(pretty_json_or_raw(content))
    except Exception as exc:
        payload: dict[str, str] = {"error": str(exc)}
        if content:
            payload["raw"] = content
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1
    return 0


def parse_once(client: PlanParserClient, query: str) -> str:
    return client.complete_json(top_router_prompt(), query)


def validate_probe_top_parse(content: str | dict[str, Any]) -> dict[str, Any]:
    payload = load_probe_payload(content)
    router = payload.get("router")
    if router not in TOP_ROUTERS:
        raise PlanParseError(f"Invalid top router: {router}")
    extract_query = payload.get("extract_query")
    if not isinstance(extract_query, str):
        raise PlanParseError("Top parser extract_query must be a string")
    filters = validate_top_filters(payload.get("filters", []), "Top")
    filter_groups = validate_filter_groups(payload.get("filter_groups", []))
    warnings = validate_placeholder_warnings(extract_query, filter_groups)
    return {
        "router": router,
        "extract_query": extract_query.strip(),
        "filters": filters,
        "filter_groups": filter_groups,
        "warnings": warnings,
    }


def load_probe_payload(content: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(content, str):
        payload = json.loads(strip_code_fence(content))
    else:
        payload = dict(content)
    if isinstance(payload, list):
        raise PlanParseError("Top probe expected one JSON object, got a JSON array")
    if not isinstance(payload, dict):
        raise PlanParseError("Top probe JSON root must be an object")
    return payload


def validate_filter_groups(value: Any) -> list[dict[str, Any]]:
    if value is None:
        value = []
    if not isinstance(value, list):
        raise PlanParseError("Top parser filter_groups must be a list")
    groups: list[dict[str, Any]] = []
    for group in value:
        if not isinstance(group, dict):
            raise PlanParseError("Top parser filter_group must be an object")
        subject = group.get("subject")
        if not isinstance(subject, str):
            raise PlanParseError("Top parser filter_group subject must be a string")
        groups.append({
            "subject": subject.strip(),
            "filters": validate_top_filters(group.get("filters", []), "Top filter_group"),
        })
    return groups


def validate_top_filters(value: Any, name: str) -> list[dict[str, Any]]:
    filters = validate_paper_filters(value, name)
    for filter_item in filters:
        field = filter_item.get("field")
        allowed_ops = TOP_FILTER_OPS_BY_FIELD.get(str(field))
        if allowed_ops is None:
            raise PlanParseError(f"Invalid top filter field: {field}")
        op = filter_item.get("op")
        if op not in allowed_ops:
            raise PlanParseError(f"Invalid top filter op for {field}: {op}")
    return filters


def validate_placeholder_warnings(extract_query: str, filter_groups: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    has_placeholder = "{subject" in extract_query
    if not filter_groups and has_placeholder:
        warnings.append("extract_query contains subject placeholder but filter_groups is empty")
    if filter_groups and not has_placeholder:
        warnings.append("filter_groups is non-empty but extract_query has no subject placeholder")
    return warnings


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


if __name__ == "__main__":
    sys.exit(main())
