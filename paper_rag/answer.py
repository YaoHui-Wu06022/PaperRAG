from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import Settings
from .dataprocess.manifest import effective_year
from .retrieval.data.venues import canonicalize_venue
from .retrieval.top_planner import run_plan

MAX_DISPLAY_ITEMS = 10


@dataclass(frozen=True)
class AskResult:
    original_query: str
    route: str
    answer: str
    provenance: list[str]
    warnings: list[str]
    plan: dict[str, Any]


def run_ask(
    settings: Settings,
    query: str,
    *,
    plan_parser=None,
    embedder=None,
    store=None,
) -> AskResult:
    plan_pack = run_plan(
        settings,
        query,
        plan_parser=plan_parser,
        embedder=embedder,
        store=store,
    )
    answer, provenance = render_ask_answer(settings, plan_pack)
    return AskResult(
        original_query=str(plan_pack.get("original_query") or ""),
        route=str(plan_pack.get("route") or ""),
        answer=answer,
        provenance=provenance,
        warnings=display_warnings(list(plan_pack.get("warnings") or [])),
        plan=plan_pack,
    )


def display_warnings(warnings: list[str]) -> list[str]:
    return [warning for warning in warnings if not warning.startswith("metadata_relative_year_fallback:")]


def render_ask_answer(settings: Settings, plan_pack: dict[str, Any]) -> tuple[str, list[str]]:
    route = str(plan_pack.get("route") or "")
    evidence = plan_pack.get("evidence") or {}

    if route == "error":
        return "证据规划失败，无法回答。", build_provenance(route, plan_pack)
    if route == "unclear":
        return "问题语义不明确：请补充是要查正文内容、论文元数据，还是引用关系。", build_provenance(route, plan_pack)
    if route == "reference":
        return render_reference_answer(plan_pack, evidence), build_provenance(route, plan_pack)
    if route != "metadata":
        return "当前 ask 只支持 metadata/reference 问题，content 还未接入最终回答。", build_provenance(route, plan_pack)

    parse_status = str(evidence.get("parse_status") or "")
    if parse_status == "parse_failed":
        return "这个 metadata 问题暂时无法解析。", build_provenance(route, plan_pack)
    if parse_status == "unknown":
        return "暂时无法确定这是一个支持的 metadata 问题。", build_provenance(route, plan_pack)

    records = list(evidence.get("records") or [])
    if not records:
        return "没有找到匹配的论文。", build_provenance(route, plan_pack)

    intent = str(plan_pack.get("intent") or "lookup")
    if intent == "count":
        return render_count_answer(settings, records, evidence.get("count")), build_provenance(route, plan_pack)
    if intent == "list":
        return render_list_answer(settings, records), build_provenance(route, plan_pack)
    return render_lookup_answer(settings, records, plan_pack.get("return_field")), build_provenance(route, plan_pack)


def build_provenance(route: str, plan_pack: dict[str, Any]) -> list[str]:
    evidence = plan_pack.get("evidence") or {}
    count = evidence.get("count")
    record_count = len(evidence.get("records") or [])
    parts = [f"plan({route})"]
    if route == "metadata":
        parts.append(f"records={count if count is not None else record_count}")
    if route == "reference":
        result_count = len(reference_results_for_answer(evidence))
        parts.append(f"references={count if count is not None else result_count}")
    return ["; ".join(parts)]


def render_reference_answer(plan_pack: dict[str, Any], evidence: dict[str, Any]) -> str:
    parse_status = str(evidence.get("parse_status") or "")
    direction = str(evidence.get("direction") or "")
    if parse_status == "parse_failed":
        return "这个 reference 问题暂时无法解析。"
    if parse_status == "unknown_direction":
        return "暂时无法判断引用方向。"
    references = reference_results_for_answer(evidence)
    if not references:
        return "没有找到匹配的引用证据。"
    if str(plan_pack.get("intent") or "list") == "count":
        total = evidence.get("count") if isinstance(evidence.get("count"), int) else len(references)
        head = f"共找到 {total} 条引用证据。"
    else:
        head = f"共找到 {len(references)} 条引用证据："
    lines = [head]
    for index, reference in enumerate(references[:MAX_DISPLAY_ITEMS], start=1):
        lines.append(f"{index}. {reference_summary(reference, direction)}")
    if len(references) > MAX_DISPLAY_ITEMS:
        lines.append("...")
    return "\n".join(lines)


def reference_results_for_answer(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    if evidence.get("direction") == "cited_by":
        return list(evidence.get("citing_papers") or [])
    return list(evidence.get("reference_items") or [])


def reference_summary(reference: dict[str, Any], direction: str) -> str:
    ref = reference.get("reference") or {}
    raw_text = str(ref.get("raw_text") or "").strip()
    if len(raw_text) > 220:
        raw_text = f"{raw_text[:217]}..."
    if direction == "cited_by":
        paper = reference.get("citing_paper") or {}
        title = str(paper.get("title") or "未知论文")
        return title if not raw_text else f"{title} -> {raw_text}"
    paper = reference.get("anchor_paper") or {}
    title = str(paper.get("title") or "未知论文")
    return f"{title} 引用了 -> {raw_text}"


def render_lookup_answer(settings: Settings, records: list[dict[str, Any]], return_field: Any) -> str:
    field = str(return_field or "title")
    if len(records) == 1:
        return render_lookup_line(settings, records[0], field)
    lines = [render_lookup_line(settings, record, field, index=index) for index, record in enumerate(records, start=1)]
    return "\n".join(lines)


def render_lookup_line(settings: Settings, record: dict[str, Any], field: str, *, index: int | None = None) -> str:
    title = str(record.get("title") or "").strip() or "未命名论文"
    value = record.get("value", record.get(field))
    value_text = render_value(settings, value, field)
    if field == "title":
        text = f"{title} 的标题是 {value_text}。"
    elif field == "author":
        text = f"{title} 的作者是 {value_text}。"
    elif field == "year":
        if isinstance(value, dict):
            text = f"{title} 的年份信息是 {value_text}。"
        else:
            text = f"{title} 发表于 {value_text} 年。"
    elif field == "venue":
        text = f"{title} 发表在 {value_text}。"
    else:
        text = f"{title} 的 {field} 是 {value_text}。"
    if index is not None:
        return f"{index}. {text}"
    return text


def render_list_answer(settings: Settings, records: list[dict[str, Any]]) -> str:
    head = f"共找到 {len(records)} 篇论文："
    lines = [head]
    for index, record in enumerate(records[:MAX_DISPLAY_ITEMS], start=1):
        lines.append(f"{index}. {paper_summary(settings, record)}")
    if len(records) > MAX_DISPLAY_ITEMS:
        lines.append("...")
    return "\n".join(lines)


def render_count_answer(settings: Settings, records: list[dict[str, Any]], count: Any) -> str:
    total = count if isinstance(count, int) else len(records)
    head = f"共找到 {total} 篇论文。"
    if not records:
        return head
    lines = [head]
    for index, record in enumerate(records[:MAX_DISPLAY_ITEMS], start=1):
        lines.append(f"{index}. {paper_summary(settings, record)}")
    if len(records) > MAX_DISPLAY_ITEMS:
        lines.append("...")
    return "\n".join(lines)


def paper_summary(settings: Settings, record: dict[str, Any]) -> str:
    title = str(record.get("title") or "").strip() or "未命名论文"
    year = record.get("year")
    venue = canonicalize_venue(settings, record.get("venue"))
    effective = effective_year(year)
    meta_bits = [str(value) for value in [effective, venue] if value not in (None, "")]
    if meta_bits:
        return f"{title} ({'，'.join(meta_bits)})"
    return title


def render_value(settings: Settings, value: Any, field: str) -> str:
    if isinstance(value, list):
        values = [str(item).strip() for item in value if str(item).strip()]
        return "、".join(values) if values else "未找到"
    if value is None or value == "":
        return "未找到"
    if field == "year":
        if isinstance(value, dict):
            return render_year_value(value)
        try:
            return str(int(value))
        except (TypeError, ValueError):
            return str(value)
    if field == "venue":
        return canonicalize_venue(settings, value)
    return str(value)


def render_year_value(value: dict[str, Any]) -> str:
    preprint_year = value.get("preprint_year")
    publish_year = value.get("publish_year")
    if preprint_year and publish_year:
        return f"预印本年份 {preprint_year}，正式发表年份 {publish_year}"
    if preprint_year:
        return f"预印本年份 {preprint_year}，未找到正式发表年份"
    if publish_year:
        return f"正式发表年份 {publish_year}"
    return "未找到"
