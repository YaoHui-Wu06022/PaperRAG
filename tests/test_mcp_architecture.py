from __future__ import annotations


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

    assert TOOLSETS["core"] <= _registered_tool_names()
    assert TOOLSETS["acquisition"] <= _registered_tool_names()
    assert TOOLSETS["ingestion"] <= _registered_tool_names()
    assert TOOLSETS["search-admin"] <= _registered_tool_names()

