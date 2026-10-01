"""不依赖 LLM 的 scope/content object 解析器。"""

from __future__ import annotations

import re
from typing import Any

from paper_rag.retrieval.routes.common.errors import PlanParseError
from paper_rag.retrieval.routes.common.jev_client import JevDecision
from paper_rag.retrieval.routes.content.schema import validate_content_parse
from paper_rag.retrieval.routes.metadata.schema import validate_metadata_parse
from paper_rag.retrieval.routes.reference.schema import validate_reference_parse


VENUES = ("CVPR", "ICCV", "ECCV", "NeurIPS", "NIPS", "ICML", "ACL", "AAAI", "IJCAI")
YEAR_RE = re.compile(r"(?<!\d)(19\d{2}|20\d{2})(?:\s*[年-]\s*(19\d{2}|20\d{2}))?(?!\d)")


class LocalRouteParser:
    def __init__(self, query: str, decision: JevDecision | None = None):
        self.query = str(query or "").strip()
        self.decision = decision

    def parse_metadata(self, query: str) -> dict[str, Any]:
        text = str(query or self.query).strip()
        intent = (self.decision.metadata_intent if self.decision else None) or self._metadata_intent(text)
        if intent == "null":
            intent = self._metadata_intent(text)
        fields = []
        for label, field in (("作者", "author"), ("年份", "year"), ("哪年", "year"), ("venue", "venue"), ("会议", "venue"), ("标题", "title")):
            if label.casefold() in text.casefold() and field not in fields:
                fields.append(field)
        if ("发表" in text or "发布" in text) and any(token in text for token in ("哪里", "哪个", "何处")):
            if "venue" not in fields:
                fields.append("venue")
        if intent == "lookup" and not fields:
            fields = ["title"]
        groups = self._paper_groups(text)
        payload = {"intent": intent, "return_fields": fields, "paper_semantic": "", "filters": [] if groups else self._scope_filters(text), "paper_groups": groups, "group_mode": "per" if groups else "single"}
        return validate_metadata_parse(payload, text)

    def parse_reference(self, query: str) -> dict[str, Any]:
        text = str(query or self.query).strip()
        intent = (self.decision.reference_intent if self.decision else None) or ("count" if any(x in text for x in ("多少", "数量", "几篇")) else "list")
        side = (self.decision.reference_side if self.decision else None)
        if not side:
            side = "source" if any(x in text for x in ("哪些论文引用", "被哪些论文引用")) else ("object" if any(x in text for x in ("引用了哪些", "参考了哪些")) else "source")
        source_filters: list[dict[str, Any]] = []
        object_filters: list[dict[str, Any]] = []
        paper_filters = self._scope_filters(text)
        if any(x in text for x in ("哪些论文引用", "被哪些论文引用")):
            object_filters = paper_filters
        else:
            source_filters = paper_filters
        payload = {"intent": intent, "return_side": side, "source_semantic": "", "source_filters": source_filters, "source_groups": [], "source_mode": "single", "object_semantic": "", "object_filters": object_filters, "object_groups": [], "object_mode": "single"}
        return validate_reference_parse(payload, text)

    def parse_content(self, query: str) -> dict[str, Any]:
        text = str(query or self.query).strip()
        intent = (self.decision.content_intent if self.decision else None) or self._content_intent(text)
        if intent == "null":
            intent = self._content_intent(text)
        scope_filters = self._scope_filters(text)
        objects = self._content_objects(text, scope_filters=scope_filters)
        compare_objects: list[str] = []
        if intent == "compare":
            compare_objects = self._compare_objects(text)
            if len(compare_objects) < 2:
                raise PlanParseError("无法从比较问题中解析出两个 compare_objects")
            objects = [part for part in objects if part.casefold() not in {x.casefold() for x in compare_objects}]
        if intent in {"count", "exists"} and not objects:
            raise PlanParseError("content count/exists 缺少 content_objects")
        groups = self._paper_groups(text)
        payload = {"intent": intent, "paper_semantic": "", "filters": [] if groups else scope_filters, "paper_groups": groups, "group_mode": "per" if groups else "single", "content_objects": objects, "compare_objects": compare_objects}
        return validate_content_parse(payload, text)

    @staticmethod
    def _scope_values(filters: list[dict[str, Any]]) -> list[str]:
        return [
            str(item.get("value") or "").strip()
            for item in filters
            if item.get("field") == "paper" and str(item.get("value") or "").strip()
        ]

    @staticmethod
    def _strip_scope_mentions(text: str, scope_values: list[str]) -> str:
        cleaned = text
        for value in sorted(scope_values, key=len, reverse=True):
            cleaned = re.sub(re.escape(value), " ", cleaned, flags=re.IGNORECASE)
        # Remove only query framing; preserve technical objects such as
        # BasicBlock/Bottleneck and their surrounding comparison relation.
        cleaned = re.sub(r"(?i)\b(?:in|from)\s+(?:the\s+)?paper\b", " ", cleaned)
        cleaned = re.sub(r"(?i)\b(?:compare|comparison|difference between|how do|what is)\b", " ", cleaned)
        cleaned = re.sub(r"(?:在|关于|针对)\s*\S{0,30}\s*(?:论文|本文)(?:中|里|内)?", " ", cleaned)
        return cleaned

    @staticmethod
    def _metadata_intent(text: str) -> str:
        compact = "".join(text.split())
        if any(x in compact for x in ("多少", "数量", "几篇")):
            return "count"
        if any(x in compact for x in ("是否有", "有没有", "存在")):
            return "exists"
        if any(x in compact for x in ("哪些", "有哪些", "列出", "列表")):
            return "list"
        return "lookup" if any(x in compact for x in ("作者", "年份", "哪年", "venue", "会议", "标题", "发表", "发布")) else "list"

    @staticmethod
    def _content_intent(text: str) -> str:
        compact = "".join(text.split())
        if any(x in compact for x in ("比较", "区别", "差异", "vs")):
            return "compare"
        if any(x in compact for x in ("为什么", "如何", "原理", "机制", "优缺点")):
            return "reason"
        if any(x in compact for x in ("总结", "概括")):
            return "summary"
        if any(x in compact for x in ("哪些", "有哪些", "列出")):
            return "list"
        if any(x in compact for x in ("多少", "数量", "几种")):
            return "count"
        return "lookup"

    @staticmethod
    def _scope_filters(text: str) -> list[dict[str, Any]]:
        filters: list[dict[str, Any]] = []
        relation = re.search(r"(.{2,80}?)\s*(后续|之后|以后|前期|之前)", text)
        if relation:
            candidate = relation.group(1).strip()
            op = "prior" if relation.group(2) in {"前期", "之前"} else "follow"
            filters.append({"field": "paper", "op": op, "value": candidate, "negated": False})
        years = YEAR_RE.findall(text)
        if years:
            start, end = years[0]
            if not end and any(token in text for token in ("以后", "之后", "起", "及以后")):
                interval = [int(start), "inf"]
            elif not end and any(token in text for token in ("以前", "之前")):
                interval = ["-inf", int(start)]
            else:
                interval = [int(start), int(end or start)]
            filters.append({"field": "year", "op": "interval", "value": interval, "negated": False})
        for venue in VENUES:
            if venue.casefold() in text.casefold():
                filters.append({"field": "venue", "op": "=", "value": venue, "negated": False})
        # Handle the common bilingual form "在 ResNet 论文中" / "in the
        # ResNet paper" before the broader title/paper regex below.
        scoped_match = re.search(
            r"(?i)(?:在|关于|针对)\s*([A-Za-z0-9][^，,。！？!?]{1,80}?)\s*论文(?:中|里|内)?|\b(?:in|from)\s+(?:the\s+)?([A-Za-z0-9][^,.;:!?]{1,100}?)\s+paper\b",
            text,
        )
        if scoped_match:
            candidate = next((group.strip() for group in scoped_match.groups() if group and group.strip()), "")
            candidate = re.split(r"(?i)\b(?:compare|comparison|how|what|which)\b|[，,。！？!?]", candidate, maxsplit=1)[0].strip()
            if candidate and len(candidate) > 1:
                filters.append({"field": "paper", "op": "=", "value": candidate, "negated": False})
        if not any(item.get("field") == "paper" for item in filters):
            match = re.search(r"(?:论文|paper|article)\s*[《\"']?([^》\"'，。?？]+)", text, flags=re.I)
            if match:
                candidate = match.group(1).strip()
                candidate = re.split(r"(?:中|里|的|发表|引用|有哪些|有多少)", candidate, maxsplit=1)[0].strip()
                if candidate and len(candidate) > 1:
                    filters.append({"field": "paper", "op": "=", "value": candidate, "negated": False})
        if not any(item.get("field") == "paper" for item in filters):
            target = re.search(r"(?:引用了|参考了|引用的)\s*([A-Za-z0-9][^，。?？]*)", text)
            candidate = target.group(1).strip() if target else ""
            if not candidate:
                source = re.search(r"^\s*([A-Za-z0-9][^，。?？]*)\s*(?:引用了|参考了)\s*哪些", text)
                candidate = source.group(1).strip() if source else ""
            if candidate and len(candidate) > 1:
                filters.append({"field": "paper", "op": "=", "value": candidate, "negated": False})
        return filters

    @staticmethod
    def _content_objects(text: str, *, scope_filters: list[dict[str, Any]] | None = None) -> list[str]:
        cleaned = LocalRouteParser._strip_scope_mentions(text, LocalRouteParser._scope_values(scope_filters or []))
        cleaned = re.sub(r"[？?。！!,，]", " ", cleaned)
        cleaned = re.sub(r"(?:请问|请|论文中|文中|原文中|有哪些|是什么|如何|为什么|比较|区别|差异|告诉我)", " ", cleaned)
        cleaned = re.sub(r"(?:在)?(?:结构|适用深度|性能|效果)(?:与|和)?(?:结构|适用深度|性能|效果)?(?:上|方面)?(?:有什么|有何|的)?(?:差异|区别|不同)?", " ", cleaned)
        cleaned = re.sub(r"(?:上|方面)?(?:有什么|有何)(?:差异|区别|不同)", " ", cleaned)
        chunks = [part.strip(" ：:;；") for part in re.split(r"\s+|以及|和|与|and", cleaned, flags=re.I) if part.strip(" ：:;；")]
        return [part for part in chunks if len(part) >= 2 and not re.fullmatch(r"[A-Za-z]{1,3}", part)][:8]

    @staticmethod
    def _paper_groups(text: str) -> list[dict[str, Any]]:
        if not any(token in text for token in ("分别", "各自", "每篇")):
            return []
        titles = re.findall(r"《([^》]+)》", text)
        return [{"semantic": "", "filters": [{"field": "paper", "op": "=", "value": title.strip(), "negated": False}]} for title in titles if title.strip()][:8]

    @staticmethod
    def _compare_objects(text: str) -> list[str]:
        cleaned = re.sub(r"(?:结构与适用深度上)?(?:有什么|有何|的)?(?:差异|区别|不同).*$", "", text)
        match = re.search(
            r"([A-Za-z][A-Za-z0-9_-]{1,40})\s*(?:和|与|及|vs\.?|versus|and)\s*([A-Za-z][A-Za-z0-9_-]{1,40})",
            cleaned,
            flags=re.I,
        )
        if match:
            return [match.group(1), match.group(2)]
        parts = re.split(r"(?:和|与|及|vs\.?| versus |\band\b|对比|比较)", cleaned, flags=re.I)
        values = [part.strip(" ：:，,。?？") for part in parts if len(part.strip(" ：:，,。?？")) >= 2]
        return values[-2:]
