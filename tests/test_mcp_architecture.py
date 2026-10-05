from __future__ import annotations

import asyncio


def test_mcp_server_registers_acquisition_tools():
    from paper_rag.mcp import server

    assert {
        "library_acquire_arxiv",
        "library_ingest_mineru",
        "library_job_status",
        "library_get_asset_status",
        "library_search",
        "library_get_metadata",
        "library_get_assets",
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

