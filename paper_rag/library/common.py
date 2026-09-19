from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path


class LibraryError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value) -> str:
    return hashlib.sha256(
        (value if isinstance(value, str) else dumps(value)).encode("utf-8")
    ).hexdigest()


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def json_lines(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def identifier(value: str) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,160}", value):
        raise LibraryError("invalid_request", "Expected an ASCII identifier")
    return value


def evidence_id(
    document_id: str, revision_id: str, block_id: str, start: int, end: int
) -> str:
    return f"ev:{document_id}:{revision_id}:{block_id}:{start}:{end}"


def parse_evidence(value: str) -> tuple[str, str, str, int, int]:
    try:
        prefix, doc, rev, block, start, end = value.split(":")
        if prefix != "ev" or int(start) < 0 or int(end) <= int(start):
            raise ValueError
        return identifier(doc), identifier(rev), identifier(block), int(start), int(end)
    except (ValueError, TypeError):
        raise LibraryError("invalid_request", "Invalid evidence identifier") from None
