"""查询结构化抽取结果的 schema 校验。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any


class ExtractionSchemaError(ValueError):
    """抽取响应不符合本地 schema。"""


@dataclass(frozen=True)
class QueryExtraction:
    """大模型只负责语义实体抽取，本地代码负责后续校验和检索。"""

    paper_mentions: list[str]
    content_objects: list[str]
    compare_objects: list[str]
    reference_mentions: list[str]
    confidence: float
    paper_groups: list[list[str]] | None = None
    author_mentions: list[str] | None = None
    year_intervals: list[list[int | str]] | None = None
    venue_mentions: list[str] | None = None
    source_paper_mentions: list[str] | None = None
    object_paper_mentions: list[str] | None = None
    reference_side: str | None = None
    group_mode: str = "single"
    metadata_fields: list[str] | None = None
    scope_required: bool = False
    cache_hit: bool = False

    def __post_init__(self) -> None:
        for name in (
            "paper_groups",
            "author_mentions",
            "year_intervals",
            "venue_mentions",
            "source_paper_mentions",
            "object_paper_mentions",
            "metadata_fields",
        ):
            if getattr(self, name) is None:
                object.__setattr__(self, name, [])

    @classmethod
    def from_payload(cls, payload: Any) -> "QueryExtraction":
        if not isinstance(payload, dict):
            raise ExtractionSchemaError("抽取结果必须是 object")
        allowed = {
            "paper_mentions",
            "paper_groups",
            "author_mentions",
            "year_intervals",
            "venue_mentions",
            "source_paper_mentions",
            "object_paper_mentions",
            "content_objects",
            "compare_objects",
            "reference_mentions",
            "reference_side",
            "group_mode",
            "metadata_fields",
            "scope_required",
            "confidence",
        }
        extra = set(payload) - allowed
        if extra:
            raise ExtractionSchemaError(f"抽取结果包含不支持的字段：{', '.join(sorted(extra))}")

        def string_list(name: str) -> list[str]:
            value = payload.get(name, [])
            if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
                raise ExtractionSchemaError(f"{name} 必须是字符串列表")
            return list(dict.fromkeys(item.strip() for item in value if item.strip()))

        def nested_string_list(name: str) -> list[list[str]]:
            value = payload.get(name, [])
            if not isinstance(value, list):
                raise ExtractionSchemaError(f"{name} 必须是二维字符串列表")
            groups: list[list[str]] = []
            for group in value:
                if not isinstance(group, list) or any(not isinstance(item, str) for item in group):
                    raise ExtractionSchemaError(f"{name} 必须是二维字符串列表")
                values = list(dict.fromkeys(item.strip() for item in group if item.strip()))
                if values:
                    groups.append(values)
            return groups

        def intervals(name: str) -> list[list[int | str]]:
            value = payload.get(name, [])
            if not isinstance(value, list):
                raise ExtractionSchemaError(f"{name} 必须是区间列表")
            output: list[list[int | str]] = []
            for item in value:
                if not isinstance(item, list) or len(item) != 2:
                    raise ExtractionSchemaError(f"{name} 的每个区间必须有两个边界")
                bounds: list[int | str] = []
                for bound in item:
                    if isinstance(bound, bool) or not isinstance(bound, (int, str)):
                        raise ExtractionSchemaError(f"{name} 的边界必须是整数或字符串")
                    bounds.append(int(bound) if isinstance(bound, int) else bound.strip())
                output.append(bounds)
            return output

        try:
            confidence = float(payload.get("confidence", 0.0))
        except (TypeError, ValueError) as exc:
            raise ExtractionSchemaError("confidence 必须是数字") from exc
        if not 0.0 <= confidence <= 1.0:
            raise ExtractionSchemaError("confidence 必须位于 0 到 1 之间")

        reference_side = payload.get("reference_side")
        if reference_side == "null":
            reference_side = None
        if reference_side not in {None, "source", "object"}:
            raise ExtractionSchemaError("reference_side 必须是 source、object 或 null")
        group_mode = payload.get("group_mode", "single")
        if group_mode not in {"single", "per", "or", "and"}:
            raise ExtractionSchemaError("group_mode 不受支持")
        metadata_fields = string_list("metadata_fields")
        if any(field not in {"author", "year", "venue", "title"} for field in metadata_fields):
            raise ExtractionSchemaError("metadata_fields 包含不支持的字段")
        scope_required = payload.get("scope_required", False)
        if not isinstance(scope_required, bool):
            raise ExtractionSchemaError("scope_required 必须是布尔值")

        result = cls(
            paper_mentions=string_list("paper_mentions"),
            content_objects=string_list("content_objects"),
            compare_objects=string_list("compare_objects"),
            reference_mentions=string_list("reference_mentions"),
            confidence=confidence,
            paper_groups=nested_string_list("paper_groups"),
            author_mentions=string_list("author_mentions"),
            year_intervals=intervals("year_intervals"),
            venue_mentions=string_list("venue_mentions"),
            source_paper_mentions=string_list("source_paper_mentions"),
            object_paper_mentions=string_list("object_paper_mentions"),
            reference_side=reference_side,
            group_mode=group_mode,
            metadata_fields=metadata_fields,
            scope_required=scope_required,
        )
        if len(result.compare_objects) == 1:
            raise ExtractionSchemaError("compare_objects 至少需要两个对象或为空")
        return result

    def to_payload(self) -> dict[str, Any]:
        return {
            "paper_mentions": self.paper_mentions,
            "paper_groups": self.paper_groups,
            "author_mentions": self.author_mentions,
            "year_intervals": self.year_intervals,
            "venue_mentions": self.venue_mentions,
            "source_paper_mentions": self.source_paper_mentions,
            "object_paper_mentions": self.object_paper_mentions,
            "content_objects": self.content_objects,
            "compare_objects": self.compare_objects,
            "reference_mentions": self.reference_mentions,
            "reference_side": self.reference_side,
            "group_mode": self.group_mode,
            "metadata_fields": self.metadata_fields,
            "scope_required": self.scope_required,
            "confidence": self.confidence,
        }

    def with_cache_hit(self, cache_hit: bool) -> "QueryExtraction":
        return replace(self, cache_hit=cache_hit)
