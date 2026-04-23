from __future__ import annotations

from dataclasses import asdict

from ...config import Settings
from ...dataprocess.manifest import Manifest, ManifestRecord
from ..sparse.bm25 import tokenize


def load_active_manifest_records(settings: Settings) -> list[ManifestRecord]:
    manifest = Manifest.load(settings.manifest_path)
    return [
        record
        for record in manifest.records.values()
        if record.status == "active" and record.title
    ]


def match_manifest_records(settings: Settings, query: str) -> list[dict]:
    records = load_active_manifest_records(settings)
    query_token_list = tokenize(query)
    query_tokens = set(query_token_list)
    query_compact = "".join(query_token_list)
    matches: list[tuple[float, ManifestRecord]] = []
    for record in records:
        title = record.title or ""
        title_tokens = set(tokenize(title))
        if not title_tokens:
            continue
        title_compact = "".join(tokenize(title))
        overlap = title_tokens & query_tokens
        score = 0.0
        if title_compact and title_compact in query_compact:
            score = 10.0
        elif len(overlap) >= 2:
            score = len(overlap) / len(title_tokens)
        elif overlap and any(len(token) >= 6 for token in overlap):
            score = 0.25
        if score > 0:
            matches.append((score, record))
    matches.sort(key=lambda item: item[0], reverse=True)
    if matches and matches[0][0] >= 10.0:
        matches = [item for item in matches if item[0] >= 10.0]
    return [manifest_record_to_evidence(record) for _, record in matches]


def manifest_record_to_evidence(record: ManifestRecord) -> dict:
    data = asdict(record)
    return {
        "file_hash": data["file_hash"],
        "title": data["title"],
        "author": data["author"],
        "year": data["year"],
        "venue": data["venue"],
        "pdf_path": data["pdf_path"],
        "paper_data_path": data["paper_data_path"],
    }
