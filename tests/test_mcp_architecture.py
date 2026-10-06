from __future__ import annotations

import asyncio


def test_mcp_server_registers_acquisition_tools():
    from paper_rag.mcp import server

    assert {
        "library_acquire_arxiv",
        "library_ingest_mineru",
        "library_job_status",
        "library_search",
        "library_get_metadata",
        "library_citation",
        "library_retrieve",
    } <= server._registered_tool_names()


def test_acquisition_toolset_matches_registered_tools():
    from paper_rag.mcp.server import TOOLSETS, _registered_tool_names

    from paper_rag.mcp.toolsets import CORE_TOOLS

    assert CORE_TOOLS <= _registered_tool_names()
    assert TOOLSETS["acquisition"] <= _registered_tool_names()
    assert TOOLSETS["ingestion"] <= _registered_tool_names()
    assert TOOLSETS["index-admin"] <= _registered_tool_names()


def test_default_fastmcp_surface_excludes_optional_index_admin():
    from paper_rag.mcp import server

    visible = {tool.name for tool in asyncio.run(server.mcp.list_tools())}
    assert "library_acquire_arxiv" not in visible
    assert "library_ingest_mineru" not in visible
    assert "library_catalog_sync" not in visible
    assert "library_read" in visible
    assert "library_retrieve" in visible
    assert "library_get_chunk" in visible
    assert "library_citation" in visible
    assert "library_get_assets" not in visible
    assert "library_get_asset_status" not in visible


def test_core_tool_descriptions_keep_only_agent_facing_distinctions():
    from paper_rag.mcp import server

    tools = {tool.name: tool for tool in asyncio.run(server.mcp.list_tools())}

    assert "元数据" in tools["library_search"].description
    assert "不读取正文" in tools["library_search"].description
    assert "引用关系图" in tools["library_citation"].description
    assert "answer_text" in tools["library_citation"].description
    assert tools["library_retrieve"].description == "正文检索工具，默认 hybrid 检索，返回证据片段信息。"

