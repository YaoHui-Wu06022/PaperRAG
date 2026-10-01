from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from paper_rag.mcp import server as mcp_server


def test_mcp_exposes_layered_tools_without_answer(monkeypatch):
    names = [tool.name for tool in mcp_server.server._tool_manager.list_tools()]
    assert names == [
        "paper_metadata",
        "paper_references",
        "paper_content",
        "paper_library_status",
        "paper_ingest",
        "paper_index",
        "paper_full_sync",
        "paper_job_status",
    ]

    monkeypatch.setattr(mcp_server, "_settings", lambda: type("Settings", (), {"plan_final_top_k": 8})())
    monkeypatch.setattr(
        mcp_server,
        "run_plan",
        lambda *_args, **_kwargs: {
            "route": "content",
            "status": "ok",
            "intent": "lookup",
            "results": {"contexts": [{"chunk_id": "c1"}]},
        },
    )
    result = mcp_server.paper_content("query", top_k=1)
    assert result["results"]["contexts"] == [{"chunk_id": "c1"}]
    assert "answer" not in result
    assert "deepseek_answer" not in result


def test_mcp_content_rejects_top_k_above_configuration(monkeypatch):
    monkeypatch.setattr(mcp_server, "_settings", lambda: type("Settings", (), {"plan_final_top_k": 8})())
    with pytest.raises(ValueError):
        mcp_server.paper_content("query", top_k=9)


def test_mcp_returns_stable_evidence_envelope(monkeypatch):
    monkeypatch.setattr(mcp_server, "_settings", lambda: object())
    monkeypatch.setattr(mcp_server, "run_plan", lambda *_args, **_kwargs: {"route": "metadata", "status": "ok"})
    result = mcp_server.paper_metadata("query")
    assert set(result) >= {"route", "status", "intent", "plan", "resolved", "decision", "retrieval", "results", "warnings"}
    assert "answer" not in result


def test_mcp_maps_content_retrieval_failure_to_insufficient(monkeypatch):
    monkeypatch.setattr(mcp_server, "_settings", lambda: object())
    monkeypatch.setattr(
        mcp_server,
        "run_plan",
        lambda *_args, **_kwargs: {
            "route": "content",
            "status": "ok",
            "retrieval": {"status": "insufficient"},
            "results": {},
        },
    )
    assert mcp_server.paper_content("query")["status"] == "insufficient"


def test_mcp_stdio_discovers_tools():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def discover():
        parameters = StdioServerParameters(
            command="C:/Users/Fallinty/anaconda3/envs/RAG_project/python.exe",
            args=["-m", "paper_rag.mcp.server"],
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        async with stdio_client(parameters) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                return [tool.name for tool in (await session.list_tools()).tools]

    assert asyncio.run(discover()) == [
        "paper_metadata",
        "paper_references",
        "paper_content",
        "paper_library_status",
        "paper_ingest",
        "paper_index",
        "paper_full_sync",
        "paper_job_status",
    ]


def test_mcp_stdio_discovers_tools_in_three_instances():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def discover_once():
        parameters = StdioServerParameters(
            command="C:/Users/Fallinty/anaconda3/envs/RAG_project/python.exe",
            args=["-m", "paper_rag.mcp.server"],
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        async with stdio_client(parameters) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                return [tool.name for tool in (await session.list_tools()).tools]

    async def discover_all():
        return await asyncio.gather(discover_once(), discover_once(), discover_once())

    results = asyncio.run(discover_all())
    assert all(len(result) == 8 for result in results)
