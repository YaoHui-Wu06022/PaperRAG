"""Capture a non-secret, reproducible pre-migration inventory. Never alters sources."""

from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    target = (
        root
        / "data"
        / "baselines"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    target.mkdir(parents=True)
    tracked = (
        subprocess.run(
            ["rtk", "proxy", "git", "ls-files", "-z"],
            cwd=root,
            capture_output=True,
            check=True,
        )
        .stdout.decode("utf-8")
        .split("\0")
    )
    with zipfile.ZipFile(target / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in tracked:
            path = root / name
            if (
                name
                and path.is_file()
                and not name.startswith(("data/", "密匙/"))
                and path.name != ".env"
            ):
                archive.write(path, name)
    inventory = []
    for folder in ("pdf", "mineru_output", "paper_data"):
        for path in sorted((root / "data" / folder).rglob("*")):
            if path.is_file():
                inventory.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": hashlib.file_digest(
                            path.open("rb"), "sha256"
                        ).hexdigest(),
                    }
                )
    (target / "sources.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    from paper_rag.config import Settings
    from pymilvus import MilvusClient

    settings = Settings.load(root)
    client = MilvusClient(
        uri=settings.milvus_uri, token=settings.milvus_token or "", timeout=15
    )
    try:
        rows = []
        iterator = client.query_iterator(
            collection_name=settings.milvus_collection,
            output_fields=["chunk_id", "paper_id", "text"],
            batch_size=500,
        )
        try:
            while batch := iterator.next():
                rows.extend(
                    {
                        "chunk_id": r["chunk_id"],
                        "paper_id": r.get("paper_id"),
                        "text_sha256": hashlib.sha256(
                            r.get("text", "").encode()
                        ).hexdigest(),
                    }
                    for r in batch
                )
        finally:
            iterator.close()
        (target / "milvus.json").write_text(
            json.dumps(
                {"collection": settings.milvus_collection, "rows": rows},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    finally:
        client.close()
    print(target)


if __name__ == "__main__":
    main()
