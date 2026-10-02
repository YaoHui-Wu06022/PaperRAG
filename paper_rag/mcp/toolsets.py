"""MCP 工具组定义与注册校验。"""

from __future__ import annotations

from collections.abc import Iterable


TOOLSETS: dict[str, frozenset[str]] = {
    "acquisition": frozenset({"paper_arxiv_download", "paper_job_status"}),
}


def validate_toolsets(registered_names: Iterable[str]) -> None:
    """确保工具组声明的名称已经由工具模块注册。"""

    registered = set(registered_names)
    missing = {
        name: sorted(tool for tool in tools if tool not in registered)
        for name, tools in TOOLSETS.items()
        if any(tool not in registered for tool in tools)
    }
    if missing:
        details = "; ".join(f"{group}: {', '.join(names)}" for group, names in missing.items())
        raise RuntimeError(f"MCP 工具组包含未注册工具：{details}")

