from __future__ import annotations


def test_mcp_server_registers_acquisition_tools():
    from paper_rag.mcp import server

    assert {"paper_arxiv_download", "paper_job_status"} <= server._registered_tool_names()


def test_acquisition_toolset_matches_registered_tools():
    from paper_rag.mcp.server import TOOLSETS, _registered_tool_names

    assert TOOLSETS["acquisition"] <= _registered_tool_names()

