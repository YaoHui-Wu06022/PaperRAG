"""RouteDecision 保存 parser 归一化后的路由状态，供 planner 和 evidence 共用。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .data.aliases import AliasMatch


@dataclass(frozen=True)
class RouteDecision:
    """三条 domain 共享的不可变决策对象。

    metadata/content 使用 paper_* 字段，reference 使用 source_* / object_* 字段。
    """

    route: str
    query: str = ""
    intent: str | None = None
    resolved_papers: list[dict[str, Any]] = field(default_factory=list)
    alias_matches: list[AliasMatch] = field(default_factory=list)
    parser_result: dict[str, Any] | None = None
    parse_status: str = "not_parsed"
    parser_error: str | None = None
    return_fields: list[str] = field(default_factory=list)
    paper_semantic: str = ""
    filters: list[dict[str, Any]] = field(default_factory=list)
    paper_groups: list[dict[str, Any]] = field(default_factory=list)
    group_mode: str = "single"
    return_side: str | None = None
    source_semantic: str = ""
    source_filters: list[dict[str, Any]] = field(default_factory=list)
    source_groups: list[dict[str, Any]] = field(default_factory=list)
    source_mode: str = "single"
    object_semantic: str = ""
    object_filters: list[dict[str, Any]] = field(default_factory=list)
    object_groups: list[dict[str, Any]] = field(default_factory=list)
    object_mode: str = "single"
