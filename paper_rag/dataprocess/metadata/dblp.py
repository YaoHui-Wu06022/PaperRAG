from __future__ import annotations

import html
import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from ...utils import normalize_text
from .retry import urlopen_with_retry


@dataclass(frozen=True)
class DblpMatch:
    title: str
    authors: list[str]
    year: int
    venue: str


class DblpClient:
    endpoint = "https://dblp.org/search/publ/api"

    def lookup_exact_title(
        self,
        title: str,
        limit: int = 20,
        timeout: int = 30,
        retry_delay_seconds: float = 1.0,
    ) -> DblpMatch | None:
        query = build_query(title, limit)
        request = urllib.request.Request(
            f"{self.endpoint}?{query}",
            headers={"User-Agent": "Paper_RAG/0.1 (local research library ingestion)"},
        )
        with urlopen_with_retry(request, timeout=timeout, delay_seconds=retry_delay_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))

        return select_exact_match(title, data)


def build_query(title: str, limit: int) -> str:
    return urllib.parse.urlencode({"q": title, "format": "json", "h": str(limit)})


def select_exact_match(title: str, data: dict[str, Any]) -> DblpMatch | None:
    expected = normalize_title(title)
    candidates: list[DblpMatch] = []
    for hit in iter_hits(data):
        info = hit.get("info", {})
        candidate_title = clean_title(str(info.get("title") or ""))
        if normalize_title(candidate_title) != expected:
            continue
        year_text = str(info.get("year") or "")
        if not year_text.isdigit():
            continue
        candidates.append(
            DblpMatch(
                title=candidate_title,
                authors=parse_authors(info.get("authors", {})),
                year=int(year_text),
                venue=html.unescape(str(info.get("venue") or "")).strip(),
            )
        )
    if not candidates:
        return None
    return sorted(candidates, key=candidate_rank)[0]


def iter_hits(data: dict[str, Any]) -> list[dict[str, Any]]:
    hits = data.get("result", {}).get("hits", {}).get("hit", [])
    if isinstance(hits, dict):
        return [hits]
    if isinstance(hits, list):
        return [hit for hit in hits if isinstance(hit, dict)]
    return []


def parse_authors(authors_data: Any) -> list[str]:
    authors = authors_data.get("author", []) if isinstance(authors_data, dict) else []
    if isinstance(authors, dict):
        authors = [authors]
    output: list[str] = []
    for author in authors:
        if isinstance(author, dict):
            name = html.unescape(str(author.get("text") or "")).strip()
        else:
            name = html.unescape(str(author)).strip()
        if name:
            output.append(name)
    return output


def clean_title(title: str) -> str:
    title = html.unescape(title)
    title = " ".join(title.split())
    return title.rstrip(".").strip()


def normalize_title(title: str) -> str:
    return normalize_text(clean_title(title))


def candidate_rank(match: DblpMatch) -> tuple[int, int, str]:
    is_corr = normalize_text(match.venue) == "corr"
    return (1 if is_corr else 0, match.year, match.title)
