from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import zipfile

import pytest

from paper_rag.config import Settings
from paper_rag.ingest import mineru
from paper_rag.ingest.mineru import MinerUError, ingest_arxiv_inputs, preview_arxiv_ingest, safe_zip_member_target


class FakeResponse:
    def __init__(self, payload: bytes, *, status: int = 200, headers: dict[str, str] | None = None):
        self.payload = payload
        self.status = status
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _size: int = -1) -> bytes:
        if _size < 0:
            payload, self.payload = self.payload, b""
            return payload
        payload, self.payload = self.payload[:_size], self.payload[_size:]
        return payload

    def getcode(self) -> int:
        return self.status


class FakeConnection:
    instances: list["FakeConnection"] = []

    def __init__(self, *_args, **_kwargs):
        self.requests: list[tuple[str, str, dict[str, str]]] = []
        self.__class__.instances.append(self)

    def request(self, method, path, body=None, headers=None, **_kwargs):
        self.requests.append((method, path, dict(headers or {})))
        assert body is not None
        body.read()

    def getresponse(self):
        return FakeResponse(b"", status=200)

    def close(self):
        pass


def make_settings(tmp_path: Path) -> Settings:
    (tmp_path / ".env").write_text(
        "MINERU_API_KEY=test-secret\n"
        "MINERU_API_BASE_URL=https://mineru.test/api/v4\n"
        "MINERU_POLL_INTERVAL_SECONDS=0\n"
        "MINERU_POLL_TIMEOUT_SECONDS=10\n",
        encoding="utf-8",
    )
    return Settings.load(tmp_path)


def make_source(settings: Settings, base_id: str = "1706.03762") -> Path:
    source = settings.arxiv_data_dir / base_id
    source.mkdir(parents=True)
    (source / "paper.pdf").write_bytes(b"%PDF-1.7\nexample")
    (source / "metadata.json").write_text(
        json.dumps({"base_id": base_id, "canonical_id": f"{base_id}v1"}),
        encoding="utf-8",
    )
    return source


def zip_result() -> bytes:
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("result/full.md", "# Parsed paper\n")
        archive.writestr("result/content_list.json", "[]")
    return stream.getvalue()


def test_settings_load_mineru_values_without_exposing_key(tmp_path: Path):
    settings = make_settings(tmp_path)

    assert settings.mineru_api_base_url == "https://mineru.test/api/v4"
    assert settings.mineru_model_version == "vlm"
    assert settings.mineru_language == "en"
    assert settings.mineru_api_key == "test-secret"


def test_ingest_signed_upload_poll_and_atomic_result(tmp_path: Path, monkeypatch):
    settings = make_settings(tmp_path)
    source = make_source(settings)
    poll_count = 0

    def opener(request, timeout):
        nonlocal poll_count
        assert timeout == settings.mineru_request_timeout_seconds
        if request.full_url.endswith("/file-urls/batch"):
            body = json.loads(request.data.decode("utf-8"))
            assert body["files"][0]["data_id"] == "1706.03762v1"
            assert request.headers["Authorization"] == "Bearer test-secret"
            return FakeResponse(json.dumps({"code": 0, "data": {"batch_id": "batch-1", "file_urls": ["https://upload.test/signed"]}}).encode())
        if request.full_url.endswith("/extract-results/batch/batch-1"):
            poll_count += 1
            if poll_count == 1:
                payload = {"code": 0, "data": {"extract_result": [{"data_id": "1706.03762v1", "state": "pending"}]}}
            else:
                payload = {"code": 0, "data": {"extract_result": [{"data_id": "1706.03762v1", "state": "done", "full_zip_url": "https://cdn.test/result.zip"}]}}
            return FakeResponse(json.dumps(payload).encode())
        if request.full_url == "https://cdn.test/result.zip":
            return FakeResponse(zip_result(), headers={"Content-Length": str(len(zip_result()))})
        raise AssertionError(request.full_url)

    FakeConnection.instances.clear()
    monkeypatch.setattr(mineru.http.client, "HTTPSConnection", FakeConnection)
    client = mineru.MinerUClient(settings, opener=opener, sleeper=lambda _seconds: None)
    monkeypatch.setattr(mineru, "MinerUClient", lambda _settings: client)

    result = ingest_arxiv_inputs(settings, ["1706.03762"])

    assert result.succeeded == 1
    output = source / "mineru"
    assert (output / "full.md").read_text(encoding="utf-8") == "# Parsed paper\n"
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["batch_id"] == "batch-1"
    assert manifest["canonical_id"] == "1706.03762v1"
    assert "Authorization" not in FakeConnection.instances[0].requests[0][2]


def test_ingest_is_idempotent_after_success(tmp_path: Path, monkeypatch):
    settings = make_settings(tmp_path)
    source = make_source(settings)
    output = source / "mineru"
    output.mkdir()
    (output / "full.md").write_text("old", encoding="utf-8")
    from paper_rag.ingest.mineru import _sha256

    (output / "manifest.json").write_text(
        json.dumps({
            "canonical_id": "1706.03762v1",
            "source_sha256": _sha256(source / "paper.pdf"),
            "model_version": settings.mineru_model_version,
            "language": settings.mineru_language,
        }),
        encoding="utf-8",
    )
    previews = preview_arxiv_ingest(settings, ["1706.03762"])
    assert previews[0]["status"] == "already_parsed"
    result = ingest_arxiv_inputs(settings, ["1706.03762"])
    assert result.skipped == 1
    assert (output / "full.md").read_text(encoding="utf-8") == "old"


def test_failed_parse_preserves_existing_output(tmp_path: Path, monkeypatch):
    settings = make_settings(tmp_path)
    source = make_source(settings)
    output = source / "mineru"
    output.mkdir()
    (output / "full.md").write_text("keep", encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps({"canonical_id": "old"}), encoding="utf-8")

    class FailedClient:
        def __init__(self, _settings):
            pass

        def parse_pdf(self, *_args, **_kwargs):
            raise MinerUError("remote failed")

    monkeypatch.setattr(mineru, "MinerUClient", FailedClient)
    result = ingest_arxiv_inputs(settings, ["1706.03762"])

    assert result.failed == 1
    assert (output / "full.md").read_text(encoding="utf-8") == "keep"


def test_zip_path_escape_is_rejected(tmp_path: Path):
    with pytest.raises(MinerUError):
        safe_zip_member_target(tmp_path, "../escape.txt")
    with pytest.raises(MinerUError):
        safe_zip_member_target(tmp_path, "C:/escape.txt")


def test_mcp_ingest_preview_does_not_submit_job(tmp_path: Path, monkeypatch):
    from paper_rag.mcp.tools import ingestion

    settings = make_settings(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(ingestion, "get_settings", lambda: settings)
    monkeypatch.setattr(
        ingestion,
        "preview_arxiv_ingest",
        lambda *_args: [{"input": "1706.03762", "status": "ready"}],
    )
    monkeypatch.setattr(ingestion, "get_jobs", lambda: calls.append("submit"))

    result = ingestion.paper_arxiv_ingest(["1706.03762"], confirm=False)

    assert result["status"] == "confirmation_required"
    assert calls == []


def test_mcp_ingest_confirm_queues_job(tmp_path: Path, monkeypatch):
    from paper_rag.mcp.tools import ingestion

    settings = make_settings(tmp_path)
    monkeypatch.setattr(ingestion, "get_settings", lambda: settings)
    monkeypatch.setattr(
        ingestion,
        "preview_arxiv_ingest",
        lambda *_args: [{"input": "1706.03762", "status": "ready"}],
    )

    class Jobs:
        def submit(self, kind, worker):
            assert kind == "arxiv_ingest"
            assert callable(worker)
            return {"job_id": "job_test", "status": "queued"}

    monkeypatch.setattr(ingestion, "get_jobs", lambda: Jobs())
    result = ingestion.paper_arxiv_ingest(["1706.03762"], confirm=True)

    assert result["status"] == "queued"
    assert result["job"]["job_id"] == "job_test"

