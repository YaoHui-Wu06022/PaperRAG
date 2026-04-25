from __future__ import annotations

from ...top_router import RouteDecision


CONTENT_ACTION_TERMS = {
    # fact
    "流程",
    "内容",
    "步骤",
    "怎么",
    "如何",
    "方法",
    "做法",
    "设计",
    "实现",
    "训练",
    "微调",
    "构建",
    "建模",
    "实验",
    "验证",
    "评估",
    "评价",
    "消融",
    "结果",
    "性能",
    # method
    "使用",
    "用了",
    "采用",
    "引入",
    "提出",
    "利用",
    "结合",
    "融合",
    "替换",
    "改进",
    "优化",
    "扩展",
    "加入",
    "去掉",
    # reason
    "为什么",
    "为何",
    "原因",
    "动机",
    "机制",
    "原理",
    "作用",
    "影响",
    "问题",
    # compare
    "比",
    "区别",
    "差异",
    "不同",
    "相同",
    "共同",
    "优",
    "缺",
    # summary
    "总结",
    "概括",
    "归纳",
    "综述",
    "贡献",
    "局限",
    "不足",
    "趋势",
    "发展",
}


def content_route(query: str, tokens: list[str] | None = None) -> RouteDecision | None:
    _ = tokens
    term = first_content_action_term(query)
    if not term:
        return None
    return RouteDecision(
        route="content",
        reason=f"matched content action clue: {term}",
        intent=None,
        query=query,
    )


def first_content_action_term(query: str) -> str | None:
    for term in sorted(CONTENT_ACTION_TERMS, key=len, reverse=True):
        if term in query:
            return term
    return None
