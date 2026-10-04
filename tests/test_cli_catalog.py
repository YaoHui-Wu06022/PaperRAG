from __future__ import annotations

import json
from pathlib import Path

from paper_rag.cli.main import main


def test_catalog_cli_dry_run_and_sync_json(tmp_path: Path, capsys):
    source = tmp_path / "data" / "sources" / "arxiv" / "1706.03762"
    source.mkdir(parents=True)
    (source / "metadata.json").write_text(
        json.dumps(
            {
                "base_id": "1706.03762",
                "canonical_id": "1706.03762v7",
                "title": "Attention Is All You Need",
                "authors": ["Alice"],
                "abstract": "Transformer paper.",
                "categories": ["cs.CL"],
            }
        ),
        encoding="utf-8",
    )

    assert main(["--project-root", str(tmp_path), "catalog", "sync", "--dry-run", "--json"]) == 0
    dry_run = json.loads(capsys.readouterr().out)
    assert dry_run["status"] == "dry_run"
    assert dry_run["current"]["index_ready"] is False

    assert main(["--project-root", str(tmp_path), "catalog", "sync", "--json"]) == 0
    synced = json.loads(capsys.readouterr().out)
    assert synced["status"] == "rebuilt"
    assert synced["papers"] == 1
