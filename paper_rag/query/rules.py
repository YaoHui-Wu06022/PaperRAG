"""Jev 不可用时的高精度本地意图回退规则。"""

from __future__ import annotations

import re

from paper_rag.query.schemas import QueryIntent, QueryIntentDecision


# 仅保留需要上下文的常见短句，不使用单个宽泛关键词。
_INTENT_RULES = (
    (
        QueryIntent.METADATA_LOOKUP,
        re.compile(r"作者是谁|发表时间|发布时间|论文版本|元数据|DOI", re.IGNORECASE),
    ),
    (
        QueryIntent.PAPER_SUMMARY,
        re.compile(r"(?:总结|概括).{0,12}(?:论文|文章|主要贡献|方法|内容)|主要贡献(?:是什么|有哪些)"),
    ),
    (
        QueryIntent.PAPER_COMPARISON,
        re.compile(r"(?:比较|对比).{0,12}(?:论文|方法|模型|差异|优劣)|差异.{0,12}(?:论文|方法|模型)"),
    ),
    (
        QueryIntent.PAPER_CONTENT,
        re.compile(r"如何实现|怎么实现|原理是什么|实验结果是什么|为什么.{0,8}(?:方法|模型)"),
    ),
    (
        QueryIntent.PAPER_DISCOVERY,
        re.compile(r"(?:找|搜索|查找|推荐).{0,8}(?:论文|文章|文献|研究)|有哪些.{0,8}(?:论文|研究|相关工作)"),
    ),
)


def classify_by_rules(query: str) -> QueryIntentDecision | None:
    """返回唯一的高置信度回退结果，多意图和模糊问题返回 ``None``。"""

    matches = match_rule_intents(query)
    if len(matches) != 1:
        return None
    intent = matches[0]
    return QueryIntentDecision(
        intent=intent,
        candidate_handlers=(intent.value,),
        matched_intents=(intent,),
    )


def match_rule_intents(query: str) -> tuple[QueryIntent, ...]:
    """返回所有命中的规则，供 Jev 失败时诊断多意图情况。"""

    text = str(query or "").strip()
    if not text:
        return ()
    return tuple(intent for intent, pattern in _INTENT_RULES if pattern.search(text))


def contains_arxiv_reference(value: str) -> bool:
    """判断文本中是否含有合法形态的 ArXiv ID 或论文 URL 片段。"""

    return ARXIV_REFERENCE.search(str(value or "")) is not None


ARXIV_REFERENCE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?P<id>(?:\d{4}\.\d{4,5}|[A-Za-z][A-Za-z0-9.-]*/\d{7})"
    r"(?:v[1-9]\d*)?)(?:\.pdf)?"
    r"(?![A-Za-z0-9_.])"
)


__all__ = ["classify_by_rules", "contains_arxiv_reference", "match_rule_intents"]
