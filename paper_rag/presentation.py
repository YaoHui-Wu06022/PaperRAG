"""为 MCP 和 CLI 生成确定性的论文库展示文本。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from paper_rag.catalog.references import normalize_arxiv_id


TEMPLATE_VERSION = "library-answer-v1"


def attach_presentation(payload: Mapping[str, Any], presentation: dict[str, Any]) -> dict[str, Any]:
    """在不修改原始结构化字段的前提下附加展示信息。"""

    data = dict(payload.get("data") or {})
    data["presentation"] = presentation
    result = dict(payload)
    result["data"] = data
    return result


def search_presentation(data: Mapping[str, Any]) -> dict[str, Any]:
    """生成论文元数据列表的确定性答案。"""

    items = [item for item in data.get("items", []) if isinstance(item, Mapping)]
    count = int(data.get("count", len(items)) or 0)
    lines = [f"共找到 {count} 篇论文："]
    if not items:
        lines = ["未找到符合条件的论文。"]
    else:
        lines.append("")
        for number, item in enumerate(items, start=1):
            lines.extend(
                [
                    f"{number}. {_value(item.get('title'))}",
                    f"   arXiv: {_value(item.get('paper_id'))}",
                    f"   作者: {_join_values(item.get('authors'))}",
                    f"   年份: {_year(item.get('published_at'))}",
                    f"   分类: {_join_values(item.get('categories'))}",
                ]
            )
    return _presentation("metadata_list", "verbatim", "\n".join(lines))


def citation_presentation(
    data: Mapping[str, Any],
    mode: str,
    *,
    title_or_paper_id: str | None = None,
    title_lookup: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """生成三种本地引用查询的精简确定性答案。"""

    if mode == "references":
        items = [item for item in data.get("items", []) if isinstance(item, Mapping)]
        local_items = [
            item
            for item in items
            if item.get("resolution") == "local" and item.get("matched_paper_id")
        ]
        lines = [
            f"论文《{_value(title_or_paper_id or data.get('paper_id'))}》共有 {len(items)} 条参考文献。",
            f"本地匹配：{len(local_items)}",
        ]
        if local_items:
            lines.append("")
            lines.extend(
                f"{number}. {_title(item.get('matched_paper_id'), title_lookup)}"
                for number, item in enumerate(local_items[:10], start=1)
            )
        return _presentation("references_list", "verbatim", "\n".join(lines))

    if mode == "citations":
        items = [item for item in data.get("items", []) if isinstance(item, Mapping)]
        lines = [
            f"在当前本地论文库中，共有 {len(items)} 篇论文引用了 {_title(data.get('paper_id'), title_lookup)}：",
        ]
        if items:
            lines.append("")
            lines.extend(
                f"{number}. {_title(item.get('source_paper_id'), title_lookup)}"
                for number, item in enumerate(items[:10], start=1)
            )
        return _presentation("citations_list", "verbatim", "\n".join(lines))

    if mode == "graph":
        edges = [edge for edge in data.get("edges", []) if isinstance(edge, Mapping)]
        root_id = normalize_arxiv_id(str(data.get("paper_id") or ""))
        outgoing = [
            edge
            for edge in edges
            if normalize_arxiv_id(str(edge.get("source_paper_id") or "")) == root_id
        ]
        incoming = [
            edge
            for edge in edges
            if normalize_arxiv_id(str(edge.get("target_arxiv_id") or "")) == root_id
        ]
        graph_depth = int(data.get("depth") or 1)
        lines = [f"目标论文: {_title(data.get('paper_id'), title_lookup)}"]
        if graph_depth > 1:
            lines.extend(
                [
                    f"直接引用: {len(outgoing)} 篇",
                    f"直接被引用: {len(incoming)} 篇",
                    f"查询深度: {graph_depth}",
                    f"多跳范围: {len(data.get('nodes') or [])} 个节点，{len(edges)} 条边",
                ]
            )
        else:
            lines.extend([f"引用: {len(outgoing)} 篇", f"被引用: {len(incoming)} 篇"])
        if data.get("direction") == "both":
            _append_graph_section(lines, "引用（前10条）：", outgoing, "target_arxiv_id", title_lookup)
            _append_graph_section(lines, "被引用（前10条）：", incoming, "source_paper_id", title_lookup)
        elif edges:
            lines.extend(["", "引用关系（前10条）："])
            lines.extend(
                f"{number}. {_title(edge.get('source_paper_id'), title_lookup)} 引用 {_title(edge.get('target_arxiv_id'), title_lookup)}"
                for number, edge in enumerate(edges[:10], start=1)
            )
        return _presentation("citation_graph", "verbatim", "\n".join(lines))

    raise ValueError(f"unsupported citation mode: {mode}")


def retrieve_presentation(data: Mapping[str, Any], mode: str) -> dict[str, Any]:
    """标记正文证据的客户端处理策略。"""

    policy = "compose" if str(mode).casefold() == "hybrid" else "verbatim"
    answer_text = "" if policy == "compose" else str(data.get("context_text") or "")
    return _presentation("rag_evidence", policy, answer_text)


def _presentation(answer_type: str, render_policy: str, answer_text: str) -> dict[str, Any]:
    return {
        "template_version": TEMPLATE_VERSION,
        "answer_type": answer_type,
        "render_policy": render_policy,
        "answer_text": answer_text,
    }


def _value(value: Any) -> str:
    if value is None or value == "":
        return "-"
    return str(value)


def _title(value: Any, title_lookup: Mapping[str, str] | None) -> str:
    """优先使用本地论文题目，元数据缺失时回退到稳定 ID。"""

    identifier = _value(value)
    if title_lookup:
        return _value(title_lookup.get(identifier.casefold(), identifier))
    return identifier


def _join_values(value: Any) -> str:
    if isinstance(value, str):
        return value or "-"
    if isinstance(value, Iterable):
        values = [_value(item) for item in value if item not in (None, "")]
        return "、".join(values) or "-"
    return _value(value)


def _year(value: Any) -> str:
    text = _value(value)
    return text[:4] if text != "-" else text


def _append_graph_section(
    lines: list[str],
    heading: str,
    edges: list[Mapping[str, Any]],
    field: str,
    title_lookup: Mapping[str, str] | None,
) -> None:
    """按引用方向分别追加最多十条论文题目。"""

    lines.extend(["", heading])
    lines.extend(
        f"{number}. {_title(edge.get(field), title_lookup)}"
        for number, edge in enumerate(edges[:10], start=1)
    )


__all__ = [
    "TEMPLATE_VERSION",
    "attach_presentation",
    "citation_presentation",
    "retrieve_presentation",
    "search_presentation",
]
