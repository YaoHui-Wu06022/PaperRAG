from __future__ import annotations

import html
import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from ...utils import normalize_text
from .retry import urlopen_with_retry


@dataclass(frozen=True)
class SemanticScholarMatch:
    title: str
    authors: list[str]
    year: int
    venue: str


class SemanticScholarClient:
    endpoint = "https://api.semanticscholar.org/graph/v1/paper/search/match"
    fields = "title,authors,year,venue,publicationVenue,externalIds,url"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key

    def lookup_exact_title(
        self,
        title: str,
        timeout: int = 30,
        retry_delay_seconds: float = 1.0,
    ) -> SemanticScholarMatch | None:
        query = urllib.parse.urlencode({"query": title, "fields": self.fields})
        headers = {"User-Agent": "Paper_RAG/0.1 (local research library ingestion)"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        request = urllib.request.Request(f"{self.endpoint}?{query}", headers=headers)
        try:
            with urlopen_with_retry(request, timeout=timeout, delay_seconds=retry_delay_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            raise SemanticScholarError(f"HTTP {exc.code}: {detail}") from exc
        return select_exact_match(title, data)


class SemanticScholarError(RuntimeError):
    pass


def select_exact_match(title: str, data: dict[str, Any]) -> SemanticScholarMatch | None:
    expected = normalize_title(title)
    for item in iter_papers(data):
        candidate_title = clean_text(str(item.get("title") or "")).rstrip(".").strip()
        if normalize_title(candidate_title) != expected:
            continue
        authors = parse_authors(item.get("authors"))
        year = parse_year(item.get("year"))
        venue = parse_venue(item)
        if not (authors and year and venue):
            continue
        return SemanticScholarMatch(
            title=candidate_title,
            authors=authors,
            year=year,
            venue=venue,
        )
    return None


def iter_papers(data: dict[str, Any]) -> list[dict[str, Any]]:
    papers = data.get("data", [])
    if isinstance(papers, dict):
        papers = [papers]
    if isinstance(papers, list):
        return [paper for paper in papers if isinstance(paper, dict)]
    return []


def parse_authors(authors_data: Any) -> list[str]:
    if not isinstance(authors_data, list):
        return []
    authors: list[str] = []
    for author in authors_data:
        if not isinstance(author, dict):
            continue
        name = clean_text(str(author.get("name") or ""))
        if name:
            authors.append(name)
    return authors


def parse_year(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def parse_venue(item: dict[str, Any]) -> str:
    venue = clean_text(str(item.get("venue") or ""))
    if venue:
        return venue
    publication_venue = item.get("publicationVenue")
    if isinstance(publication_venue, dict):
        return clean_text(str(publication_venue.get("name") or ""))
    return ""


def clean_text(text: str) -> str:
    return " ".join(html.unescape(text).split())


def normalize_title(title: str) -> str:
    return normalize_text(clean_text(title).rstrip(".").strip())
