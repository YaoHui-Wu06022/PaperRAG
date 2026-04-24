from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import Settings
from ..dataprocess.manifest import effective_year
from .data.venues import canonicalize_venue
from .plan.planner import run_plan
from .plan.translation import contains_chinese

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
    original_query = str(plan_pack.get("original_query") or "")
    chinese = contains_chinese(original_query)
    route = str(plan_pack.get("route") or "")
    evidence = plan_pack.get("evidence") or {}

    if route == "error":
        return localized_message(chinese, "证据规划失败，无法回答。", "Planning failed, unable to answer."), build_provenance(route, plan_pack)
    if route == "reference":
        return render_reference_answer(plan_pack, evidence, chinese), build_provenance(route, plan_pack)
    if route != "metadata":
        return localized_message(
            chinese,
            "当前 ask 只支持 metadata/reference 问题；content 还未接入最终回答。",
            "The current ask command only supports metadata/reference questions; content is not implemented yet.",
        ), build_provenance(route, plan_pack)

    parse_status = str(evidence.get("parse_status") or "")
    if parse_status == "parse_failed":
        return localized_message(chinese, "这个 metadata 问题暂时无法解析。", "This metadata question could not be parsed."), build_provenance(route, plan_pack)
    if parse_status == "unknown":
        return localized_message(chinese, "暂时无法确定这是一个支持的 metadata 问题。", "This does not look like a supported metadata question."), build_provenance(route, plan_pack)

    records = list(evidence.get("records") or [])
    if not records:
        return localized_message(chinese, "没有找到匹配的论文。", "No matching papers were found."), build_provenance(route, plan_pack)

    intent = str(plan_pack.get("intent") or "lookup")
    if intent == "count":
        return render_count_answer(settings, records, evidence.get("count"), chinese), build_provenance(route, plan_pack)
    if intent == "list":
        return render_list_answer(settings, records, chinese), build_provenance(route, plan_pack)
    return render_lookup_answer(settings, records, plan_pack.get("return_field"), chinese), build_provenance(route, plan_pack)


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


def render_reference_answer(plan_pack: dict[str, Any], evidence: dict[str, Any], chinese: bool) -> str:
    parse_status = str(evidence.get("parse_status") or "")
    if parse_status == "parse_failed":
        return localized_message(chinese, "这个 reference 问题暂时无法解析。", "This reference question could not be parsed.")
    if parse_status == "unknown_direction":
        return localized_message(chinese, "暂时无法判断引用方向。", "Unable to determine the reference direction.")
    references = reference_results_for_answer(evidence)
    if not references:
        return localized_message(chinese, "没有找到匹配的引用证据。", "No matching reference evidence was found.")
    if str(plan_pack.get("intent") or "list") == "count":
        total = evidence.get("count") if isinstance(evidence.get("count"), int) else len(references)
        head = localized_message(chinese, f"共找到 {total} 条引用证据。", f"Found {total} reference match(es).")
    else:
        head = localized_message(chinese, f"共找到 {len(references)} 条引用证据：", f"Found {len(references)} reference match(es):")
    lines = [head]
    for index, reference in enumerate(references[:MAX_DISPLAY_ITEMS], start=1):
        lines.append(f"{index}. {reference_summary(reference)}")
    if len(references) > MAX_DISPLAY_ITEMS:
        lines.append("...")
    return "\n".join(lines)


def reference_results_for_answer(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    if evidence.get("direction") == "incoming":
        return list(evidence.get("citing_papers") or [])
    return list(evidence.get("reference_items") or [])


def reference_summary(reference: dict[str, Any]) -> str:
    direction = reference.get("direction")
    ref = reference.get("reference") or {}
    raw_text = str(ref.get("raw_text") or "").strip()
    if len(raw_text) > 220:
        raw_text = f"{raw_text[:217]}..."
    if direction == "incoming":
        paper = reference.get("citing_paper") or {}
        title = str(paper.get("title") or "Unknown paper")
        return title if not raw_text else f"{title} -> {raw_text}"
    paper = reference.get("anchor_paper") or {}
    title = str(paper.get("title") or "Unknown paper")
    return f"{title} cites -> {raw_text}"


def render_lookup_answer(settings: Settings, records: list[dict[str, Any]], return_field: Any, chinese: bool) -> str:
    field = str(return_field or "title")
    if len(records) == 1:
        return render_lookup_line(settings, records[0], field, chinese)
    lines = [render_lookup_line(settings, record, field, chinese, index=index) for index, record in enumerate(records, start=1)]
    return "\n".join(lines)


def render_lookup_line(settings: Settings, record: dict[str, Any], field: str, chinese: bool, *, index: int | None = None) -> str:
    title = str(record.get("title") or "").strip() or localized_message(chinese, "未命名论文", "Untitled paper")
    value = record.get("value", record.get(field))
    value_text = render_value(settings, value, field, chinese)
    if field == "title":
        text = localized_message(chinese, f"{title} 的标题是 {value_text}。", f"{title} is titled {value_text}.")
    elif field == "author":
        text = localized_message(chinese, f"{title} 的作者是 {value_text}。", f"{title} was written by {value_text}.")
    elif field == "year":
        if isinstance(value, dict):
            text = localized_message(chinese, f"{title} 的年份信息是 {value_text}。", f"{title} year information: {value_text}.")
        else:
            text = localized_message(chinese, f"{title} 发表于 {value_text} 年。", f"{title} was published in {value_text}.")
    elif field == "venue":
        text = localized_message(chinese, f"{title} 发表在 {value_text}。", f"{title} was published in {value_text}.")
    else:
        text = localized_message(chinese, f"{title} 的 {field} 是 {value_text}。", f"The {field} of {title} is {value_text}.")
    if index is not None:
        return f"{index}. {text}"
    return text


def render_list_answer(settings: Settings, records: list[dict[str, Any]], chinese: bool) -> str:
    head = localized_message(chinese, f"共找到 {len(records)} 篇论文：", f"Found {len(records)} paper(s):")
    lines = [head]
    for index, record in enumerate(records[:MAX_DISPLAY_ITEMS], start=1):
        lines.append(f"{index}. {paper_summary(settings, record, chinese)}")
    if len(records) > MAX_DISPLAY_ITEMS:
        lines.append("...")
    return "\n".join(lines)


def render_count_answer(settings: Settings, records: list[dict[str, Any]], count: Any, chinese: bool) -> str:
    total = count if isinstance(count, int) else len(records)
    head = localized_message(chinese, f"共找到 {total} 篇论文。", f"Found {total} paper(s).")
    if not records:
        return head
    lines = [head]
    for index, record in enumerate(records[:MAX_DISPLAY_ITEMS], start=1):
        lines.append(f"{index}. {paper_summary(settings, record, chinese)}")
    if len(records) > MAX_DISPLAY_ITEMS:
        lines.append("...")
    return "\n".join(lines)


def paper_summary(settings: Settings, record: dict[str, Any], chinese: bool) -> str:
    title = str(record.get("title") or "").strip() or localized_message(chinese, "未命名论文", "Untitled paper")
    year = record.get("year")
    venue = canonicalize_venue(settings, record.get("venue"))
    effective = effective_year(year)
    meta_bits = [str(value) for value in [effective, venue] if value not in (None, "")]
    if meta_bits:
        joiner = "，" if chinese else ", "
        return f"{title} ({joiner.join(meta_bits)})"
    return title


def render_value(settings: Settings, value: Any, field: str, chinese: bool) -> str:
    if isinstance(value, list):
        values = [str(item).strip() for item in value if str(item).strip()]
        joiner = "、" if chinese else ", "
        return joiner.join(values) if values else localized_message(chinese, "未找到", "not found")
    if value is None or value == "":
        return localized_message(chinese, "未找到", "not found")
    if field == "year":
        if isinstance(value, dict):
            return render_year_value(value, chinese)
        try:
            return str(int(value))
        except (TypeError, ValueError):
            return str(value)
    if field == "venue":
        return canonicalize_venue(settings, value)
    return str(value)


def render_year_value(value: dict[str, Any], chinese: bool) -> str:
    preprint_year = value.get("preprint_year")
    publish_year = value.get("publish_year")
    if chinese:
        if preprint_year and publish_year:
            return f"预印本年份 {preprint_year}，正式发表年份 {publish_year}"
        if preprint_year:
            return f"预印本年份 {preprint_year}，未找到正式发表年份"
        if publish_year:
            return f"正式发表年份 {publish_year}"
        return "未找到"
    if preprint_year and publish_year:
        return f"preprint year {preprint_year}, publication year {publish_year}"
    if preprint_year:
        return f"preprint year {preprint_year}; publication year not found"
    if publish_year:
        return f"publication year {publish_year}"
    return "not found"


def localized_message(chinese: bool, zh: str, en: str) -> str:
    return zh if chinese else en
