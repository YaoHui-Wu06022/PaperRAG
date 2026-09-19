from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return {
        k.strip(): v.strip().strip('"').strip("'")
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#") and "=" in line
        for k, v in [line.split("=", 1)]
    }


@dataclass(frozen=True)
class LibrarySettings:
    root: Path
    home: Path
    env: dict[str, str]

    @classmethod
    def load(cls, root: Path | None = None) -> "LibrarySettings":
        root = (root or Path.cwd()).resolve()
        env = dotenv(root / ".env") | dict(os.environ)
        home = Path(env.get("PAPER_LIBRARY_DIR", "data/library"))
        return cls(root, (root / home).resolve(), env)

    @property
    def database(self) -> Path:
        return self.home / "catalog.sqlite3"

    @property
    def embedding_identity(self) -> dict:
        return {
            "endpoint": self.env.get("EMBEDDING_BASE_URL", "").rstrip("/"),
            "deployment": self.env.get("EMBEDDING_DEPLOYMENT", ""),
            "model": self.env.get("EMBEDDING_MODEL", "qwen3.7-text-embedding-flash"),
            "dimensions": int(self.env.get("EMBEDDING_DIM", "1024")),
            "text_version": "title-section-body-v2",
        }

    @property
    def collection_prefix(self) -> str:
        return self.env.get("MILVUS_V2_COLLECTION", "paper_library_v2")
