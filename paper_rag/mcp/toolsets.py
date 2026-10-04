"""MCP 工具组解析、启用和注册漂移校验。"""

from __future__ import annotations

import os
from collections.abc import Iterable


TOOLSETS_ENV_VAR = "PAPER_RAG_TOOLSETS"

CORE_TOOLS: frozenset[str] = frozenset(
    {
        "paper_query",
        "paper_list",
        "paper_search",
        "paper_get_metadata",
        "paper_get_assets",
        "paper_asset_status",
        "paper_job_status",
        "paper_search_chunks",
        "paper_get_chunk",
        "paper_get_fulltext",
        "paper_get_references",
        "paper_get_citations",
        "paper_citation_graph",
    }
)

TOOLSETS: dict[str, frozenset[str]] = {
    "acquisition": frozenset({"paper_arxiv_download"}),
    "ingestion": frozenset({"paper_arxiv_ingest"}),
    "search-admin": frozenset({"paper_catalog_sync", "paper_embedding_status", "paper_embedding_rebuild"}),
    # 管理工具尚未注册，先保留组名以便配置文件稳定演进。
    "management": frozenset(),
}

DEFAULT_ON: frozenset[str] = frozenset({"acquisition", "ingestion"})


class UnknownToolsetError(ValueError):
    """环境变量包含未定义的工具组。"""


def _split(raw: str) -> list[str]:
    return [token.casefold() for token in raw.replace(",", " ").split() if token]


def resolve_enabled(raw: str | None = None) -> set[str]:
    """解析 ``PAPER_RAG_TOOLSETS``，返回启用的可选工具组。"""

    spec = _split(os.environ.get(TOOLSETS_ENV_VAR, "") if raw is None else raw)
    if not spec:
        return set(DEFAULT_ON)

    enabled: set[str] = set()
    valid = ", ".join(sorted(TOOLSETS) + ["all", "none"])
    for token in spec:
        negated = token.startswith("-")
        name = token[1:] if negated else token
        if name == "all":
            candidates = set(TOOLSETS)
        elif name == "none":
            candidates = set(TOOLSETS)
        elif name in TOOLSETS:
            candidates = {name}
        else:
            raise UnknownToolsetError(
                f"Unknown toolset {name!r} in {TOOLSETS_ENV_VAR}. Valid values: {valid}"
            )
        if name == "none" and not negated:
            enabled.clear()
        elif negated:
            enabled.difference_update(candidates)
        else:
            enabled.update(candidates)
    return enabled


def apply_toolsets(mcp: object, raw: str | None = None) -> set[str]:
    """将工具组可见性应用到 FastMCP 实例。"""

    enabled = resolve_enabled(raw)
    on = set(CORE_TOOLS)
    off: set[str] = set()
    for name, tools in TOOLSETS.items():
        if name in enabled:
            on.update(tools)
        else:
            off.update(tools)
    if off:
        mcp.disable(names=off)
    if on:
        mcp.enable(names=on)
    return enabled


def validate_toolsets(registered_names: Iterable[str]) -> None:
    """确保核心和可选工具组中的名称都已注册。"""

    registered = set(registered_names)
    groups = {"core": CORE_TOOLS, **TOOLSETS}
    missing = {
        name: sorted(tool for tool in tools if tool not in registered)
        for name, tools in groups.items()
        if any(tool not in registered for tool in tools)
    }
    if missing:
        details = "; ".join(f"{group}: {', '.join(names)}" for group, names in missing.items())
        raise RuntimeError(f"MCP 工具组包含未注册工具：{details}")


__all__ = [
    "CORE_TOOLS",
    "DEFAULT_ON",
    "TOOLSETS",
    "TOOLSETS_ENV_VAR",
    "UnknownToolsetError",
    "apply_toolsets",
    "resolve_enabled",
    "validate_toolsets",
]
