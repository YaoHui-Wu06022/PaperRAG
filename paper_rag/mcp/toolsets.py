"""论文库 MCP 工具组解析和注册校验。"""

from __future__ import annotations

import os
from collections.abc import Iterable

TOOLSETS_ENV_VAR = "PAPER_RAG_TOOLSETS"
CORE_TOOLS: frozenset[str] = frozenset({
    "library_search", "library_retrieve", "library_get_metadata",
    "library_job_status", "library_validate_answer",
})
TOOLSETS: dict[str, frozenset[str]] = {
    "acquisition": frozenset({"library_acquire_arxiv"}),
    "ingestion": frozenset({"library_ingest_mineru", "library_catalog_sync"}),
    "index-admin": frozenset({"library_index_status", "library_index_rebuild"}),
    "citation": frozenset({"library_citation"}),
    "fulltext": frozenset({"library_read", "library_get_chunk"}),
}
DEFAULT_ON: frozenset[str] = frozenset({"citation", "fulltext"})


class UnknownToolsetError(ValueError):
    """环境变量包含未定义的工具组。"""


def _split(raw: str) -> list[str]:
    return [token.casefold() for token in raw.replace(",", " ").split() if token]


def resolve_enabled(raw: str | None = None) -> set[str]:
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
            raise UnknownToolsetError(f"Unknown toolset {name!r} in {TOOLSETS_ENV_VAR}. Valid values: {valid}")
        if name == "none" and not negated:
            enabled.clear()
        elif negated:
            enabled.difference_update(candidates)
        else:
            enabled.update(candidates)
    return enabled


def apply_toolsets(mcp: object, raw: str | None = None) -> set[str]:
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


__all__ = ["CORE_TOOLS", "DEFAULT_ON", "TOOLSETS", "TOOLSETS_ENV_VAR", "UnknownToolsetError", "apply_toolsets", "resolve_enabled", "validate_toolsets"]
