from __future__ import annotations

import json
from pathlib import Path
import time

import pytest

from paper_rag.acquisition.arxiv import (
    ArxivError,
    ArxivMetadata,
    ArxivStore,
    normalize_arxiv_input,
    parse_latest_metadata,
)
from paper_rag.config import Settings


def metadata(base_id: str = "1706.03762", version: int = 1) -> ArxivMetadata:
    canonical = f"{base_id}v{version}"
    return ArxivMetadata(
        base_id=base_id,
        version=version,
        canonical_id=canonical,
        title="A Test Paper",
        authors=["Alice Smith"],
        abstract="An abstract.",
        categories=["cs.CV"],
        published_at="2017-06-12T00:00:00Z",
        updated_at="2017-06-12T00:00:00Z",
        abs_url=f"https://arxiv.org/abs/{canonical}",
        pdf_url=f"https://arxiv.org/pdf/{canonical}.pdf",
    )


class FakeClient:
    def __init__(self, current: ArxivMetadata | None = None, *, fail_download: bool = False):
        self.current = current or metadata()
        self.fail_download = fail_download
        self.downloads = 0

    def resolve_latest(self, ref):
        return self.current

    def download_pdf(self, paper, target: Path):
        self.downloads += 1
        if self.fail_download:
            raise ArxivError("fake download failed")
        content = b"%PDF-1.7\nA test PDF"
        target.write_bytes(content)
        import hashlib

        return hashlib.sha256(content).hexdigest(), len(content)


def test_normalize_modern_legacy_and_urls():
    assert normalize_arxiv_input("1706.03762").canonical_input == "1706.03762"
    assert normalize_arxiv_input("1706.03762v2").requested_version == 2
    assert normalize_arxiv_input("https://arxiv.org/abs/1706.03762").base_id == "1706.03762"
    assert normalize_arxiv_input("https://arxiv.org/pdf/hep-th/9901001.pdf").base_id == "hep-th/9901001"
    with pytest.raises(ValueError):
        normalize_arxiv_input("https://example.com/paper.pdf")
    with pytest.raises(ValueError):
        normalize_arxiv_input("http://arxiv.org/abs/1706.03762")


def test_parse_atom_metadata_reads_version_from_pdf_link():
    xml = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>http://arxiv.org/abs/1706.03762</id>
        <published>2017-06-12T00:00:00Z</published>
        <updated>2017-06-12T00:00:00Z</updated>
        <title> A Test Paper </title>
        <summary> An abstract. </summary>
        <author><name>Alice Smith</name></author>
        <category term="cs.CV" />
        <link title="pdf" href="http://export.arxiv.org/pdf/1706.03762v2" />
      </entry>
    </feed>"""
    result = parse_latest_metadata(xml, "1706.03762")
    assert result.canonical_id == "1706.03762v2"
    assert result.title == "A Test Paper"
    assert result.categories == ["cs.CV"]
    assert result.pdf_url.endswith("1706.03762v2.pdf")


def test_download_is_idempotent_and_writes_manifest(tmp_path: Path):
    settings = Settings.load(tmp_path)
    client = FakeClient()
    store = ArxivStore(settings, client)

    first = store.download(["1706.03762"])
    second = store.download(["https://arxiv.org/abs/1706.03762"])

    assert first.downloaded == 1
    assert second.skipped == 1
    assert client.downloads == 1
    target = settings.arxiv_data_dir / "1706.03762"
    assert (target / "paper.pdf").read_bytes().startswith(b"%PDF-")
    saved = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
    assert saved["canonical_id"] == "1706.03762v1"
    manifest = (settings.arxiv_data_dir / "manifest.jsonl").read_text(encoding="utf-8")
    assert manifest.count("1706.03762") >= 1


def test_new_version_replaces_old_and_old_request_does_not_downgrade(tmp_path: Path):
    settings = Settings.load(tmp_path)
    client = FakeClient(metadata(version=1))
    store = ArxivStore(settings, client)
    assert store.download(["1706.03762"]).downloaded == 1

    client.current = metadata(version=2)
    assert store.download(["1706.03762"]).downloaded == 1
    saved = json.loads(
        (settings.arxiv_data_dir / "1706.03762" / "metadata.json").read_text(encoding="utf-8")
    )
    assert saved["version"] == 2

    result = store.download(["1706.03762v1"])
    assert result.skipped == 1
    assert result.items[0].error == "拒绝覆盖本地较新版本"
    assert client.downloads == 2


def test_batch_continues_after_invalid_input_and_duplicate_is_skipped(tmp_path: Path):
    settings = Settings.load(tmp_path)
    store = ArxivStore(settings, FakeClient())
    result = store.download(["1706.03762", "not-an-id", "1706.03762"])
    assert result.downloaded == 1
    assert result.failed == 1
    assert result.skipped == 1
    assert result.items[2].error == "批次内重复输入"


def test_failed_download_preserves_existing_asset(tmp_path: Path):
    settings = Settings.load(tmp_path)
    good = ArxivStore(settings, FakeClient())
    good.download(["1706.03762"])
    target = settings.arxiv_data_dir / "1706.03762" / "paper.pdf"
    before = target.read_bytes()

    failed = ArxivStore(settings, FakeClient(metadata(version=2), fail_download=True))
    result = failed.download(["1706.03762"])
    assert result.failed == 1
    assert target.read_bytes() == before


def test_mcp_confirmation_does_not_write(tmp_path: Path, monkeypatch):
    from paper_rag.mcp import server

    settings = Settings.load(tmp_path)
    monkeypatch.setattr(server, "_settings", lambda: settings)
    monkeypatch.setattr(server, "preview_arxiv_inputs", lambda *_args: [{"input": "1706.03762", "status": "new"}])
    monkeypatch.setattr(server, "download_arxiv_inputs", lambda *_args, **_kwargs: None)
    server._jobs.cache_clear()

    preview = server.paper_arxiv_download(["1706.03762"], confirm=False)
    assert preview["status"] == "confirmation_required"
    assert not settings.arxiv_data_dir.exists()


def test_mcp_confirmation_queues_download_job(tmp_path: Path, monkeypatch):
    from paper_rag.mcp import server

    settings = Settings.load(tmp_path)
    monkeypatch.setattr(server, "_settings", lambda: settings)
    monkeypatch.setattr(server, "preview_arxiv_inputs", lambda *_args: [{"input": "1706.03762", "status": "new"}])

    class Result:
        def to_dict(self):
            return {"items": [], "downloaded": 1, "skipped": 0, "failed": 0}

    monkeypatch.setattr(server, "download_arxiv_inputs", lambda *_args, **_kwargs: Result())
    server._jobs.cache_clear()
    queued = server.paper_arxiv_download(["1706.03762"], confirm=True)
    job_id = queued["job"]["job_id"]
    for _ in range(100):
        status = server.paper_job_status(job_id)
        if status["status"] in {"succeeded", "failed"}:
            break
        time.sleep(0.01)
    status = server.paper_job_status(job_id)
    assert status["status"] == "succeeded"
    assert status["result"]["downloaded"] == 1

