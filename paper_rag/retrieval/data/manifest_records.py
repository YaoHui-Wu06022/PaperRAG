"""active manifest records 的读取、轻量召回和统一身份 key。"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from ...config import Settings
from ...dataprocess.manifest import Manifest, ManifestRecord
from .utils import dedupe_by, normalize_bm25_token


def load_active_manifest_records(settings: Settings) -> list[ManifestRecord]:
    """读取当前仍 active 且有标题的 manifest 记录。"""
    manifest = Manifest.load(settings.manifest_path)
    return [
        record
        for record in manifest.records.values()
        if record.status == "active" and record.title
    ]


def match_manifest_records(settings: Settings, query: str) -> list[dict]:
    """用标题轻量召回匹配 query 的 manifest records。"""
    records = load_active_manifest_records(settings)
    query_token_list = normalize_bm25_token(query)
    query_tokens = set(query_token_list)
    query_compact = "".join(query_token_list)
    matches: list[tuple[float, ManifestRecord]] = []
    for record in records:
        title = record.title or ""
        title_tokens = set(normalize_bm25_token(title))
        if not title_tokens:
            continue
        # 轻量召回只看标题 token，不改写原始 query，也不读取正文索引。
        title_compact = "".join(normalize_bm25_token(title))
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
    return [to_evidence_manifest_record(record) for _, record in matches]


def to_evidence_manifest_record(record: ManifestRecord) -> dict:
    """把 ManifestRecord 裁剪为检索层内部 record dict。"""
    data = asdict(record)
    return {
        "_record_key": paper_record_key(record),
        "file_hash": data["file_hash"],
        "title": data["title"],
        "author": data["author"],
        "year": data["year"],
        "venue": data["venue"],
        "pdf_path": data["pdf_path"],
        "paper_data_path": data["paper_data_path"],
    }


def paper_record_key(record: ManifestRecord | dict[str, Any]) -> str:
    """生成全项目统一使用的论文身份 key。"""
    if isinstance(record, dict):
        if record.get("_record_key"):
            return str(record["_record_key"])
        if record.get("paper_id"):
            return str(record["paper_id"])
        if record.get("paper_data_path"):
            return path_name(record["paper_data_path"])
        return str(record.get("title") or "")
    if record.paper_data_path:
        return path_name(record.paper_data_path)
    return str(record.title or "")


def dedupe_paper_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 paper_record_key 对论文 records 保序去重。"""
    return dedupe_by(records, paper_record_key)


def merge_paper_records(*paper_lists: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """合并多组论文 records，并按论文身份去重。"""
    return dedupe_paper_records([paper for paper_list in paper_lists for paper in paper_list])


def path_name(path: Any) -> str:
    """从路径字符串中取最后一级名称。"""
    return Path(str(path or "").replace("\\", "/").rstrip("/")).name
