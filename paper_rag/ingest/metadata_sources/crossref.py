"""Crossref 标题精确匹配客户端。"""

from __future__ import annotations

import html
import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from paper_rag.ingest.metadata_sources.retry import urlopen_with_retry
from paper_rag.utils import normalize_text


@dataclass(frozen=True)
class CrossrefMatch:
    title: str
    authors: list[str]
    year: int
    venue: str | None


class CrossrefClient:
    endpoint = "https://api.crossref.org/works"

    def __init__(self, mailto: str | None = None):
        self.mailto = mailto

    def lookup_exact_title(self, title: str, limit: int = 20, timeout: int = 30, retry_delay_seconds: float = 1.0) -> CrossrefMatch | None:
        params = urllib.parse.urlencode({"query.title": title, "rows": str(limit)})
        headers = {"User-Agent": "Paper_RAG/0.1 (local research library ingestion)"}
        if self.mailto:
            headers["Mailto"] = self.mailto
        request = urllib.request.Request(f"{self.endpoint}?{params}", headers=headers)
        with urlopen_with_retry(request, timeout=timeout, delay_seconds=retry_delay_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
        return select_exact_match(title, data)


def select_exact_match(title: str, data: dict[str, Any]) -> CrossrefMatch | None:
    expected = normalize_text(title)
    candidates: list[CrossrefMatch] = []
    for item in data.get("message", {}).get("items", []) if isinstance(data.get("message"), dict) else []:
        titles = item.get("title") or []
        candidate_title = " ".join(str(titles[0] if titles else "").split()).rstrip(".").strip()
        if not candidate_title or normalize_text(candidate_title) != expected:
            continue
        date = item.get("published-print") or item.get("published-online") or item.get("issued") or {}
        parts = date.get("date-parts") if isinstance(date, dict) else []
        year = parts[0][0] if parts and parts[0] and str(parts[0][0]).isdigit() else None
        if year is None:
            continue
        authors = []
        for author in item.get("author") or []:
            if not isinstance(author, dict):
                continue
            name = " ".join(str(author.get("given") or "").split() + str(author.get("family") or "").split()).strip()
            if name:
                authors.append(html.unescape(name))
        venue = str(item.get("container-title", [""])[0] if item.get("container-title") else "").strip() or None
        candidates.append(CrossrefMatch(candidate_title, authors, int(year), venue))
    return sorted(candidates, key=lambda item: (item.year, item.title))[0] if candidates else None
