from __future__ import annotations

import asyncio


def test_mcp_server_registers_acquisition_tools():
    from paper_rag.mcp import server

    assert {
        "paper_arxiv_download",
        "paper_arxiv_ingest",
        "paper_job_status",
        "paper_asset_status",
        "paper_list",
        "paper_search",
        "paper_get_metadata",
        "paper_get_assets",
        "paper_query",
    } <= server._registered_tool_names()


def test_acquisition_toolset_matches_registered_tools():
    from paper_rag.mcp.server import TOOLSETS, _registered_tool_names

    from paper_rag.mcp.toolsets import CORE_TOOLS

    assert CORE_TOOLS <= _registered_tool_names()
    assert TOOLSETS["acquisition"] <= _registered_tool_names()
    assert TOOLSETS["ingestion"] <= _registered_tool_names()
    assert TOOLSETS["search-admin"] <= _registered_tool_names()


def test_default_fastmcp_surface_excludes_optional_index_admin():
    from paper_rag.mcp import server

    visible = {tool.name for tool in asyncio.run(server.mcp.list_tools())}
    assert "paper_arxiv_download" in visible
    assert "paper_arxiv_ingest" in visible
    assert "paper_catalog_sync" not in visible
    assert "paper_get_fulltext" in visible
    assert "paper_search_chunks" in visible
    assert "paper_get_chunk" in visible

