"""metadata/reference/失败状态的本地确定性回答生成。"""

from __future__ import annotations

from typing import Any


def should_use_answer_llm(evidence: dict[str, Any]) -> bool:
    """只有 content 正文证据需要交给 LLM 组织自然语言答案。"""
    if evidence.get("route") != "content" or evidence.get("status") != "ok":
        return False
    results = evidence.get("results")
    return isinstance(results, dict) and bool(results.get("contexts"))


def compose_local_answer(evidence: dict[str, Any]) -> str:
    """本地组织 metadata/reference/失败状态的答案。"""
    route = evidence.get("route")
    status = evidence.get("status")
    if status != "ok":
        return local_failure_answer(evidence)
    if route == "metadata":
        return compose_metadata_answer(evidence)
    if route == "reference":
        return compose_reference_answer(evidence)
    if route == "content":
        return "没有找到足够的正文证据，暂时无法可靠回答这个问题。"
    return local_failure_answer(evidence)


def compose_answer_failure_answer() -> str:
    """回答模型调用失败时返回明确的本地降级说明。"""
    return "已经找到正文证据，但回答模型调用失败，暂时无法生成答案。"


def local_failure_answer(evidence: dict[str, Any]) -> str:
    """把 parse failed / unclear / graph missing 转成直接可读的回答。"""
    status = evidence.get("status")
    if status == "unclear":
        return "这个问题的检索语义还不明确，暂时无法确定应该查元数据、引用关系还是正文内容。"
    if status == "graph_missing":
        return "本地 citation graph 不存在，暂时无法回答引用关系问题；请先运行 paper-rag ingest 重建引用图。"
    if status == "parse_failed":
        return "问题解析失败，暂时无法回答。"
    return "没有足够证据回答这个问题。"


def compose_metadata_answer(evidence: dict[str, Any]) -> str:
    """从 metadata evidence 本地生成确定性答案。"""
    results = evidence.get("results") if isinstance(evidence.get("results"), dict) else {}
    if "count" in results:
        sections = [f"共找到 {results['count']} 篇符合条件的论文。"]
        title_aliases = evidence_title_aliases(evidence)
        items = results.get("items") or []
        if items:
            sections.append(format_answer_lines([format_metadata_item(item, title_aliases) for item in items], numbered=True))
        if results.get("groups"):
            sections.append(compose_metadata_groups(results["groups"], title_aliases, intent="count"))
        return "\n".join(section for section in sections if section)
    if "exists" in results:
        if results["exists"]:
            return "是的"
        actual = results.get("actual") or []
        if actual:
            title_aliases = evidence_title_aliases(evidence)
            lines = [format_metadata_item(item, title_aliases, compact_lookup=False) for item in actual]
            return "不是，" + format_answer_lines(lines, numbered=False)
        if results.get("groups"):
            return "不是\n" + compose_metadata_groups(results["groups"], evidence_title_aliases(evidence), intent="exists")
        return "不是"
    if results.get("groups"):
        return compose_metadata_groups(results["groups"], evidence_title_aliases(evidence), intent=str(evidence.get("intent") or ""))
    items = results.get("items") or []
    if not items:
        return "没有找到符合条件的论文元数据。"
    title_aliases = evidence_title_aliases(evidence)
    lines = [format_metadata_item(item, title_aliases, compact_lookup=False) for item in items]
    return format_answer_lines(lines, numbered=evidence.get("intent") == "list")


def compose_metadata_groups(
    groups: list[dict[str, Any]],
    title_aliases: dict[str, str] | None = None,
    *,
    intent: str = "",
) -> str:
    """本地组织 metadata 分组结果。"""
    lines: list[str] = []
    for group in groups:
        scope = format_scope(group.get("scope"))
        items = group.get("items") or []
        if items:
            item_text = format_answer_lines(
                [format_metadata_item(item, title_aliases, compact_lookup=intent == "lookup") for item in items],
                numbered=intent in {"list", "count"},
            )
            if intent == "count" and "count" in group:
                lines.append(f"{scope}：{group['count']} 篇\n{item_text}")
            elif intent == "list":
                lines.append(f"{scope}：\n{item_text}")
            else:
                lines.append(item_text)
        elif "count" in group:
            lines.append(f"{scope}：{group['count']} 篇")
        elif "exists" in group:
            lines.append(f"{scope}：{'有' if group['exists'] else '没有'}符合条件的论文")
        else:
            lines.append(f"{scope}：没有找到对应论文。")
    return "\n".join(lines)


def format_metadata_item(
    item: dict[str, Any],
    title_aliases: dict[str, str] | None = None,
    *,
    compact_lookup: bool = False,
) -> str:
    """格式化单条 metadata item。"""
    title = display_title(item.get("title"), title_aliases)
    values = item.get("values") if isinstance(item.get("values"), dict) else {}
    if not values:
        return title
    if compact_lookup and len(values) == 1:
        field, value = next(iter(values.items()))
        return f"{title}：{format_lookup_value(field, value)}。"
    return f"{title} 的{format_metadata_values(values)}。"


def format_lookup_value(field: str, value: Any) -> str:
    """lookup 单字段时省掉“年份是”等机械连接词。"""
    text = format_value(value)
    if text:
        return text
    labels = {
        "author": "作者未知",
        "year": "年份未知",
        "venue": "发表 venue 未知",
        "title": "标题未知",
    }
    return labels.get(field, "信息未知")


def format_metadata_values(values: dict[str, Any]) -> str:
    """把 author/year/venue 等字段转成中文短文本。"""
    labels = {
        "author": "作者是",
        "year": "年份是",
        "venue": "发表 venue 是",
        "title": "标题是",
    }
    parts = []
    for field, value in values.items():
        label = labels.get(field, f"{field} 是")
        parts.append(f"{label} {format_value(value)}")
    return "；".join(parts)


def compose_reference_answer(evidence: dict[str, Any]) -> str:
    """从 reference evidence 本地生成确定性答案。"""
    results = evidence.get("results") if isinstance(evidence.get("results"), dict) else {}
    title_aliases = evidence_title_aliases(evidence)
    if "count" in results:
        sections = [f"共找到 {results['count']} 篇符合引用关系的论文。"]
        papers = results.get("papers") or []
        edges = results.get("edges") or []
        if papers:
            sections.append(format_answer_lines([display_title(paper, title_aliases) for paper in papers], numbered=True))
        if edges:
            edge_lines = "\n".join(f"- {format_reference_edge(edge, title_aliases)}" for edge in edges)
            sections.append(f"引用证据：\n{edge_lines}")
        return "\n".join(section for section in sections if section)
    if "exists" in results:
        answer = "是，存在符合条件的引用关系。" if results["exists"] else "否，没有找到符合条件的引用关系。"
        edges = results.get("edges") or []
        if edges:
            answer += f"\n证据：{format_reference_edge(edges[0], title_aliases)}"
        return answer
    if results.get("groups") and reference_uses_per_groups(evidence):
        return compose_reference_groups(results["groups"], intent=str(evidence.get("intent") or ""))
    papers = results.get("papers") or []
    edges = results.get("edges") or []
    if not papers and not edges:
        return "没有找到符合条件的本地引用关系。"
    sections: list[str] = []
    if papers:
        sections.append(format_answer_lines([display_title(paper, title_aliases) for paper in papers], numbered=evidence.get("intent") == "list"))
    if edges:
        edge_lines = "\n".join(f"- {format_reference_edge(edge, title_aliases)}" for edge in edges)
        sections.append(f"引用证据：\n{edge_lines}")
    return "\n".join(section for section in sections if section)


def reference_uses_per_groups(evidence: dict[str, Any]) -> bool:
    """reference 只有 per 模式直接逐组回答；and/or 使用聚合后的 papers。"""
    plan = evidence.get("plan") if isinstance(evidence.get("plan"), dict) else {}
    return plan.get("source_mode") == "per" or plan.get("object_mode") == "per"


def compose_reference_groups(groups: list[dict[str, Any]], *, intent: str = "") -> str:
    """本地组织 reference 分组结果。"""
    lines: list[str] = []
    for group in groups:
        scope = format_scope(group.get("scope"))
        papers = group.get("papers") or []
        if papers:
            lines.append(f"{scope}：\n{format_answer_lines(papers, numbered=intent in {'list', 'count'})}")
        elif "count" in group:
            lines.append(f"{scope}：{group['count']} 篇")
        elif "exists" in group:
            lines.append(f"{scope}：{'存在' if group['exists'] else '没有'}引用关系")
        else:
            lines.append(f"{scope}：没有命中")
    return "\n".join(lines)


def format_reference_edge(edge: dict[str, Any], title_aliases: dict[str, str] | None = None) -> str:
    """格式化一条 compact reference edge。"""
    source = display_title(edge.get("source"), title_aliases) if edge.get("source") else "未知论文"
    obj = display_title(edge.get("object"), title_aliases) if edge.get("object") else "未知论文"
    page = f"，页码 {edge['page']}" if edge.get("page") else ""
    block = f"，block {edge['block']}" if edge.get("block") else ""
    return f"{source} -> {obj}{page}{block}"


def format_answer_lines(lines: list[str], *, numbered: bool) -> str:
    """多行结果按需加 [1]/[2] 编号。"""
    cleaned = [line for line in lines if line]
    if not cleaned:
        return ""
    if numbered:
        return "\n".join(f"[{index}] {line}" for index, line in enumerate(cleaned, start=1))
    return "\n".join(cleaned)


def evidence_title_aliases(evidence: dict[str, Any]) -> dict[str, str]:
    """从 evidence.resolved.aliases 中构造 canonical title -> 用户别名的展示映射。"""
    resolved = evidence.get("resolved") if isinstance(evidence.get("resolved"), dict) else {}
    aliases = resolved.get("aliases") if isinstance(resolved, dict) else []
    mapping: dict[str, str] = {}
    for match in aliases if isinstance(aliases, list) else []:
        if not isinstance(match, dict):
            continue
        canonical = str(match.get("canonical") or "").strip()
        alias = str(match.get("alias") or "").strip()
        if canonical and alias:
            mapping.setdefault(canonical, alias)
    return mapping


def display_title(value: Any, title_aliases: dict[str, str] | None = None) -> str:
    """有用户别名时优先用别名展示论文，否则返回完整标题。"""
    title = str(value or "论文").strip()
    return (title_aliases or {}).get(title, title)


def format_scope(scope: Any) -> str:
    """把 compact scope list 转成文本。"""
    if isinstance(scope, list) and scope:
        return "、".join(str(item) for item in scope)
    return "默认范围"


def format_list(values: list[Any]) -> str:
    """中文顿号连接列表。"""
    return "、".join(str(value) for value in values)


def format_value(value: Any) -> str:
    """把 metadata value 转成中文展示文本。"""
    if isinstance(value, list):
        return format_list(value)
    if isinstance(value, dict):
        year_parts = []
        if value.get("preprint_year"):
            year_parts.append(f"预印本 {value['preprint_year']}")
        if value.get("publish_year"):
            year_parts.append(f"正式发表 {value['publish_year']}")
        if year_parts:
            return "，".join(year_parts)
        return "，".join(f"{key}: {item}" for key, item in value.items())
    return str(value)
