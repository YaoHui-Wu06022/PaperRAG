"""Jev 决策层的稳定策略。

Jev 只负责有限集合分类；开放式 scope 和正文对象由结构化抽取器识别，再由本地代码校验。
"""

from __future__ import annotations

POLICY_VERSION = "v1"
ROUTE_CHOICES = {
    "metadata": "最终答案只需要作者、年份、venue、标题或论文数量",
    "reference": "最终答案需要查询引用发出方、被引论文或引用数量",
    "content": "最终答案需要阅读论文正文中的方法、结构、实验或解释",
    "unclear": "无法确定查询对象或无法判断所需数据源",
}
METADATA_INTENT_CHOICES = {
    "lookup": "查询论文的指定元数据字段",
    "list": "列出满足条件的论文或元数据",
    "count": "只需要论文数量",
    "exists": "只需要判断论文或元数据条件是否存在",
    "null": "不是 metadata 问题",
}
REFERENCE_INTENT_CHOICES = {
    "list": "列出引用发出方或被引论文",
    "count": "统计引用关系数量",
    "exists": "判断引用关系是否存在",
    "null": "不是 reference 问题",
}
REFERENCE_SIDE_CHOICES = {
    "source": "返回引用别人/发出引用的论文",
    "object": "返回被引用的论文",
    "null": "不需要引用关系方向",
}
CONTENT_INTENT_CHOICES = {
    "lookup": "从正文查找一个事实或局部信息",
    "reason": "解释方法、原因、机制或实验结论",
    "compare": "比较两篇论文、两个结构或两个方法",
    "summary": "总结正文内容",
    "list": "列出正文中出现的对象或内容",
    "count": "统计正文对象数量",
    "exists": "判断正文中是否存在某个对象或事实",
    "null": "不是 content 问题",
}


def decision_questions() -> dict[str, dict[str, object]]:
    """返回一次 Jev 请求所需的全部固定 choice 问题。"""
    return {
        "route": {"type": "choice", "instructions": "选择最终答案所需的数据源 route。", "criteria": ROUTE_CHOICES},
        "metadata_intent": {"type": "choice", "instructions": "仅在 metadata route 下选择元数据任务；否则选择 null。", "criteria": METADATA_INTENT_CHOICES},
        "reference_intent": {"type": "choice", "instructions": "仅在 reference route 下选择引用关系任务；否则选择 null。", "criteria": REFERENCE_INTENT_CHOICES},
        "reference_side": {"type": "choice", "instructions": "判断引用关系答案返回 source 还是 object；非 reference 选择 null。", "criteria": REFERENCE_SIDE_CHOICES},
        "content_intent": {"type": "choice", "instructions": "仅在 content route 下选择正文任务；否则选择 null。", "criteria": CONTENT_INTENT_CHOICES},
        "needs_synthesis": {
            "type": "choice",
            "instructions": "只有需要把多个正文证据组织成解释、比较或总结时选择 true；metadata/reference 通常选择 false。",
            "criteria": {
                "true": "需要综合多个正文证据，或解释、比较、总结方法与实验。",
                "false": "单字段 metadata、citation graph，或单条正文事实查询。",
            },
        },
        "complexity": {
            "type": "choice",
            "instructions": "选择问题复杂度等级；1 是单字段查询，5 是多论文正文综合。",
            "criteria": {
                "1": "单个元数据字段、数量或存在性判断。",
                "2": "单篇论文的单个正文事实。",
                "3": "单篇论文的解释或两个字段的比较。",
                "4": "跨章节、跨论文的综合或比较。",
                "5": "需要多个论文、多个证据来源的完整综合。",
            },
        },
    }


def rule_fallback(query: str) -> dict[str, object]:
    """Jev 不可用或置信度不足时的保守规则 fallback。"""
    text = "".join(str(query or "").split()).casefold()
    if any(token in text for token in ("引用", "被引", "参考文献", "citedby", "citation")):
        route = "reference"
    elif any(token in text for token in ("作者", "年份", "哪年", "venue", "会议", "标题", "发表", "发布", "多少篇论文", "论文数量")):
        route = "metadata"
    elif any(token in text for token in ("方法", "结构", "实验", "数据集", "使用", "损失函数", "消融", "指标", "为什么", "如何", "比较", "差异", "优缺点", "原理")):
        route = "content"
    else:
        route = "unclear"
    if route != "metadata":
        metadata_intent = "null"
    elif any(x in text for x in ("多少", "数量", "几篇")):
        metadata_intent = "count"
    elif any(x in text for x in ("哪些", "有哪些", "列出", "列表")):
        metadata_intent = "list"
    else:
        metadata_intent = "lookup"
    reference_intent = "list" if route == "reference" else "null"
    if route == "reference" and any(x in text for x in ("哪些论文引用", "被哪些论文引用")):
        reference_side = "source"
    elif route == "reference" and any(x in text for x in ("引用了哪些", "参考了哪些")):
        reference_side = "object"
    else:
        reference_side = "object" if route == "reference" and any(x in text for x in ("被引用", "引用的")) else ("source" if route == "reference" else "null")
    if route != "content":
        content_intent = "null"
    elif any(x in text for x in ("比较", "差异", "区别")):
        content_intent = "compare"
    elif any(x in text for x in ("为什么", "如何", "原理", "机制", "优缺点")):
        content_intent = "reason"
    elif any(x in text for x in ("总结", "概括")):
        content_intent = "summary"
    elif any(x in text for x in ("哪些", "有哪些", "列出", "列表")):
        content_intent = "list"
    elif any(x in text for x in ("多少", "数量", "几种")):
        content_intent = "count"
    else:
        content_intent = "lookup"
    return {
        "route": route,
        "metadata_intent": metadata_intent,
        "reference_intent": reference_intent,
        "reference_side": reference_side,
        "content_intent": content_intent,
        "needs_synthesis": route == "content" and content_intent in {"reason", "compare", "summary"},
        "complexity": 2 if route != "unclear" else 1,
    }
