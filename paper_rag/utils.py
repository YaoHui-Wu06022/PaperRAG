from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def slugify_title(text: str, *, fallback: str = "untitled") -> str:
    text = text.strip()
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or fallback


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def safe_move_dir(src: Path, dst: Path) -> Path:
    if not src.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    final = dst
    counter = 2
    while final.exists():
        final = dst.with_name(f"{dst.name}_{counter}")
        counter += 1
    shutil.move(str(src), str(final))
    return final


def replace_dir(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def infer_title_from_pdf_name(path: Path) -> str:
    stem = path.stem
    parts = [p.strip() for p in stem.split(" - ")]
    if len(parts) >= 3:
        return parts[-1]
    return stem

