from __future__ import annotations

import json
from pathlib import Path
import time

import pytest

from paper_rag.config import Settings
from paper_rag.mcp.jobs import JobManager
from paper_rag.mcp.sources import SourceValidationError, normalize_arxiv_url, preview_sources, stage_sources


def make_pdf(path: Path, body: bytes = b"%PDF-1.7\ncontent") -> Path:
    path.write_bytes(body)
    return path


def test_source_preview_and_stage_deduplicate(tmp_path):
    settings = Settings.load(tmp_path)
    source = make_pdf(tmp_path / "input.pdf")
    preview = preview_sources(settings, [str(source)])
    assert preview[0]["type"] == "local_pdf"
    assert preview[0]["duplicate"] is False
    staged = stage_sources(settings, [str(source)], "job_test")
    assert staged[0]["duplicate"] is False
    assert Path(staged[0]["path"]).exists()
    second = stage_sources(settings, [str(source)], "job_test_2")
    assert second[0]["duplicate"] is True


def test_source_path_and_url_validation(tmp_path):
    settings = Settings.load(tmp_path)
    source = make_pdf(tmp_path / "input.pdf")
    with pytest.raises(SourceValidationError):
        preview_sources(settings, [str(tmp_path / "input.txt")])
    with pytest.raises(SourceValidationError):
        preview_sources(settings, ["http://example.com/paper.pdf"])
    with pytest.raises(SourceValidationError):
        preview_sources(settings, ["https://127.0.0.1/paper.pdf"])
    assert normalize_arxiv_url("https://arxiv.org/abs/1706.03762") == "https://arxiv.org/pdf/1706.03762.pdf"
    assert source.exists()


def test_job_manager_persists_and_marks_interrupted(tmp_path):
    settings = Settings.load(tmp_path)
    manager = JobManager(settings)
    record = manager.submit("test", lambda report: {"ok": True})
    for _ in range(50):
        status = manager.status(record["job_id"])
        if status["status"] == "succeeded":
            break
        time.sleep(0.01)
    assert manager.status(record["job_id"])["result"] == {"ok": True}
    queued = {
        "job_id": "job_old",
        "kind": "index",
        "status": "running",
        "created_at": "now",
        "started_at": "now",
        "finished_at": None,
        "progress": {},
        "result": {},
        "error": None,
    }
    settings.mcp_job_log_path.parent.mkdir(parents=True, exist_ok=True)
    settings.mcp_job_log_path.write_text(json.dumps(queued) + "\n", encoding="utf-8")
    restarted = JobManager(settings)
    assert restarted.status("job_old")["status"] == "interrupted"

    def fail(_report):
        raise RuntimeError("Authorization: Bearer sk-secret-value")

    failed = restarted.submit("failed", fail)
    for _ in range(50):
        if restarted.status(failed["job_id"])["status"] == "failed":
            break
        time.sleep(0.01)
    failed_status = restarted.status(failed["job_id"])
    assert failed_status["status"] == "failed"
    assert "sk-secret-value" not in failed_status["error"]


def test_mcp_ingest_confirmation_and_async_job(monkeypatch, tmp_path):
    from paper_rag.mcp import server as mcp_server

    settings = Settings.load(tmp_path)
    source = make_pdf(tmp_path / "input.pdf")
    monkeypatch.setattr(mcp_server, "_settings", lambda: settings)
    mcp_server._jobs.cache_clear()
    monkeypatch.setattr(mcp_server, "run_ingest", lambda *_args, **_kwargs: type("Summary", (), {"processed": ["ok"]})())
    preview = mcp_server.paper_ingest([str(source)], confirm=False)
    assert preview["status"] == "confirmation_required"
    assert not list(settings.pdf_dir.glob("*.pdf"))
    queued = mcp_server.paper_ingest([str(source)], confirm=True)
    job_id = queued["job"]["job_id"]
    for _ in range(100):
        status = mcp_server.paper_job_status(job_id)
        if status["status"] in {"succeeded", "failed"}:
            break
        time.sleep(0.01)
    assert mcp_server.paper_job_status(job_id)["status"] == "succeeded"
