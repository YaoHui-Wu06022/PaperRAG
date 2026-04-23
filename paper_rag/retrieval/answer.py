from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import Settings
from .plan.planner import run_plan
from .plan.translation import BaiduTranslator, contains_chinese

MAX_DISPLAY_ITEMS = 10


@dataclass(frozen=True)
class AskResult:
    original_query: str
    retrieval_query: str
    route: str
    answer: str
    provenance: list[str]
    warnings: list[str]
    plan: dict[str, Any]


def run_ask(
    settings: Settings,
    query: str,
    *,
    translator: BaiduTranslator | None = None,
    plan_parser=None,
    embedder=None,
    store=None,
) -> AskResult:
    plan_pack = run_plan(
        settings,
        query,
        translator=translator,
        plan_parser=plan_parser,
        embedder=embedder,
        store=store,
    )
    answer, provenance = render_ask_answer(plan_pack)
    return AskResult(
        original_query=str(plan_pack.get("original_query") or ""),
        retrieval_query=str(plan_pack.get("retrieval_query") or ""),
        route=str(plan_pack.get("route") or ""),
        answer=answer,
        provenance=provenance,
        warnings=list(plan_pack.get("warnings") or []),
        plan=plan_pack,
    )


def render_ask_answer(plan_pack: dict[str, Any]) -> tuple[str, list[str]]:
    original_query = str(plan_pack.get("original_query") or "")
    chinese = contains_chinese(original_query)
    route = str(plan_pack.get("route") or "")
    evidence = plan_pack.get("evidence") or {}

    if route == "error":
        return localized_message(chinese, "中文翻译失败，无法回答。", "Translation failed, unable to answer."), build_provenance(route, plan_pack)
    if route != "metadata":
        return localized_message(
            chinese,
            "当前 ask 只支持 metadata 问题，reference/content 还未接入。",
            "The current ask command only supports metadata questions; reference/content are not implemented yet.",
        ), build_provenance(route, plan_pack)

    parse_status = str(evidence.get("parse_status") or "")
    if parse_status == "parse_failed":
        return localized_message(chinese, "这个 metadata 问题暂时无法解析。", "This metadata question could not be parsed."), build_provenance(route, plan_pack)
    if parse_status == "unknown":
        return localized_message(chinese, "暂时无法确定这是一个 metadata 问题。", "This does not look like a supported metadata question."), build_provenance(route, plan_pack)

    records = list(evidence.get("records") or [])
    if not records:
        return localized_message(chinese, "没有找到匹配的论文。", "No matching papers were found."), build_provenance(route, plan_pack)

    intent = str(plan_pack.get("intent") or "lookup")
    if intent == "count":
        return render_count_answer(records, evidence.get("count"), chinese), build_provenance(route, plan_pack)
    if intent == "list":
        return render_list_answer(records, chinese), build_provenance(route, plan_pack)
    return render_lookup_answer(records, plan_pack.get("return_field"), chinese), build_provenance(route, plan_pack)


def build_provenance(route: str, plan_pack: dict[str, Any]) -> list[str]:
    evidence = plan_pack.get("evidence") or {}
    count = evidence.get("count")
    record_count = len(evidence.get("records") or [])
    parts = [f"plan({route})"]
    if route == "metadata":
        parts.append(f"records={count if count is not None else record_count}")
    translation_provider = plan_pack.get("translation_provider")
    if translation_provider:
        parts.append(f"translation={translation_provider}")
    return ["，".join(parts)]


def render_lookup_answer(records: list[dict[str, Any]], return_field: Any, chinese: bool) -> str:
    field = str(return_field or "title")
    if len(records) == 1:
        return render_lookup_line(records[0], field, chinese)
    lines = [render_lookup_line(record, field, chinese, index=index) for index, record in enumerate(records, start=1)]
    return "\n".join(lines)


def render_lookup_line(record: dict[str, Any], field: str, chinese: bool, *, index: int | None = None) -> str:
    title = str(record.get("title") or "").strip() or localized_message(chinese, "未命名论文", "Untitled paper")
    value = record.get("value", record.get(field))
    value_text = render_value(value, field, chinese)
    paper_text = title
    if field == "title":
        text = localized_message(chinese, f"{paper_text} 的标题是 {value_text}。", f"{paper_text} is titled {value_text}.")
    elif field == "author":
        text = localized_message(chinese, f"{paper_text} 的作者是 {value_text}。", f"{paper_text} was written by {value_text}.")
    elif field == "year":
        text = localized_message(chinese, f"{paper_text} 发表于 {value_text}。", f"{paper_text} was published in {value_text}.")
    elif field == "venue":
        text = localized_message(chinese, f"{paper_text} 发表于 {value_text}。", f"{paper_text} was published in {value_text}.")
    else:
        text = localized_message(chinese, f"{paper_text} 的 {field} 是 {value_text}。", f"The {field} of {paper_text} is {value_text}.")
    if index is not None:
        return f"{index}. {text}"
    return text


def render_list_answer(records: list[dict[str, Any]], chinese: bool) -> str:
    head = localized_message(chinese, f"共找到 {len(records)} 篇论文：", f"Found {len(records)} paper(s):")
    lines = [head]
    for index, record in enumerate(records[:MAX_DISPLAY_ITEMS], start=1):
        lines.append(f"{index}. {paper_summary(record, chinese)}")
    if len(records) > MAX_DISPLAY_ITEMS:
        lines.append("..." if chinese else "...")
    return "\n".join(lines)


def render_count_answer(records: list[dict[str, Any]], count: Any, chinese: bool) -> str:
    total = count if isinstance(count, int) else len(records)
    head = localized_message(chinese, f"共找到 {total} 篇论文。", f"Found {total} paper(s).")
    if not records:
        return head
    lines = [head]
    for index, record in enumerate(records[:MAX_DISPLAY_ITEMS], start=1):
        lines.append(f"{index}. {paper_summary(record, chinese)}")
    if len(records) > MAX_DISPLAY_ITEMS:
        lines.append("..." if chinese else "...")
    return "\n".join(lines)


def paper_summary(record: dict[str, Any], chinese: bool) -> str:
    title = str(record.get("title") or "").strip() or localized_message(chinese, "未命名论文", "Untitled paper")
    year = record.get("year")
    venue = record.get("venue")
    summary = title
    meta_bits = [str(value) for value in [year, venue] if value not in (None, "")]
    if meta_bits:
        joiner = "，" if chinese else ", "
        summary = f"{summary} ({joiner.join(meta_bits)})"
    return summary


def render_value(value: Any, field: str, chinese: bool) -> str:
    if isinstance(value, list):
        values = [str(item).strip() for item in value if str(item).strip()]
        joiner = "、" if chinese else ", "
        return joiner.join(values) if values else localized_message(chinese, "未找到", "not found")
    if value is None or value == "":
        return localized_message(chinese, "未找到", "not found")
    if field == "year":
        try:
            return str(int(value))
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def localized_message(chinese: bool, zh: str, en: str) -> str:
    return zh if chinese else en
