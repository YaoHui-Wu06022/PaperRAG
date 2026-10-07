"""JEV 不可用时的正文任务规则回退。"""

from __future__ import annotations

import re

from paper_rag.routing.schemas import RetrieveDecision, RetrieveRequest, RetrieveTask, RouteIntent


_RULES = (
    (RetrieveTask.COMPARISON, re.compile(r"比较|对比|差异|优劣|异同")),
    (RetrieveTask.SUMMARY, re.compile(r"总结|概括|主要贡献|摘要|讲了什么")),
    (RetrieveTask.REASON, re.compile(r"概念|含义|定义|解释|什么是|如何理解|为什么|原理|如何实现|怎么实现")),
)


def classify_by_rules(request: RetrieveRequest) -> RetrieveDecision:
    """正文请求默认按事实证据处理，特征命中时切换任务。"""

    # 读取单篇论文表/图中的比较数据是事实任务，不自动扩大为跨论文比较。
    if re.search(r"(?:表|图)\s*\d+", request.query) and re.search(r"展示|报告|列出|显示|给出", request.query):
        return RetrieveDecision(RouteIntent.RETRIEVE, RetrieveTask.FACT, provider="rules", fallback_used=True, confidence=0.8)
    for task, pattern in _RULES:
        if pattern.search(request.query):
            return RetrieveDecision(RouteIntent.RETRIEVE, task, provider="rules", fallback_used=True, confidence=0.8)
    return RetrieveDecision(RouteIntent.RETRIEVE, RetrieveTask.FACT, provider="rules", fallback_used=True, confidence=0.7)


__all__ = ["classify_by_rules"]
