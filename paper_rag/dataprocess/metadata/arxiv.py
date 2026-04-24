from __future__ import annotations

import html
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Iterable

from ...utils import normalize_text
from .retry import urlopen_with_retry


ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


@dataclass(frozen=True)
class ArxivMatch:
    title: str
    authors: list[str]
    preprint_year: int


class ArxivClient:
    endpoint = "https://export.arxiv.org/api/query"

    def lookup_exact_title(self, title: str, timeout: int = 30, retry_delay_seconds: float = 1.0) -> ArxivMatch | None:
        query = urllib.parse.urlencode(
            {
                "search_query": f'ti:"{title}"',
                "start": "0",
                "max_results": "10",
            }
        )
        request = urllib.request.Request(
            f"{self.endpoint}?{query}",
            headers={"User-Agent": "Paper_RAG/0.1 (local research library ingestion)"},
        )
        with urlopen_with_retry(request, timeout=timeout, delay_seconds=retry_delay_seconds) as response:
            xml_text = response.read().decode("utf-8")
        return select_exact_match(title, xml_text)


def select_exact_match(title: str, xml_text: str) -> ArxivMatch | None:
    expected = normalize_title(title)
    for entry in iter_entries(xml_text):
        candidate_title = clean_text(entry.findtext("atom:title", default="", namespaces=ATOM_NS))
        if normalize_title(candidate_title) != expected:
            continue
        year = parse_year(entry.findtext("atom:published", default="", namespaces=ATOM_NS))
        if year is None:
            year = parse_year(entry.findtext("atom:updated", default="", namespaces=ATOM_NS))
        if year is None:
            continue
        authors = [
            clean_text(author.findtext("atom:name", default="", namespaces=ATOM_NS))
            for author in entry.findall("atom:author", ATOM_NS)
        ]
        return ArxivMatch(
            title=candidate_title.rstrip(".").strip(),
            authors=[author for author in authors if author],
            preprint_year=year,
        )
    return None


def iter_entries(xml_text: str) -> Iterable[ET.Element]:
    root = ET.fromstring(xml_text)
    return root.findall("atom:entry", ATOM_NS)


def clean_text(text: str) -> str:
    return " ".join(html.unescape(text).split())


def normalize_title(title: str) -> str:
    return normalize_text(clean_text(title).rstrip(".").strip())


def parse_year(timestamp: str) -> int | None:
    if len(timestamp) < 4 or not timestamp[:4].isdigit():
        return None
    return int(timestamp[:4])
