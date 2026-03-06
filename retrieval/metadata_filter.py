from __future__ import annotations

from dataclasses import dataclass
import re

from langchain_core.documents import Document


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


@dataclass(frozen=True)
class QueryMetadataFilters:
    years: tuple[str, ...]
    source_terms: tuple[str, ...]
    title_terms: tuple[str, ...]
    author_terms: tuple[str, ...]
    venue_terms: tuple[str, ...]
    keyword_terms: tuple[str, ...]

    @property
    def has_constraints(self) -> bool:
        return any(
            [
                self.years,
                self.source_terms,
                self.title_terms,
                self.author_terms,
                self.venue_terms,
                self.keyword_terms,
            ]
        )


def parse_query_metadata_filters(query: str) -> QueryMetadataFilters:
    text = query.strip()
    years = tuple(sorted(set(re.findall(r"(19\d{2}|20\d{2})", text))))

    def _extract(pattern: str) -> tuple[str, ...]:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        cleaned = []
        for item in matches:
            value = str(item).strip().strip(".,;，；。")
            if value:
                cleaned.append(value)
        return tuple(cleaned)

    source_terms = _extract(r"(?:source|paper|文献)\s*[:：]\s*([^\n,，;；]+)")
    title_terms = _extract(r"(?:title|标题)\s*[:：]\s*([^\n,，;；]+)")
    author_terms = _extract(r"(?:author|作者)\s*[:：]\s*([^\n,，;；]+)")
    venue_terms = _extract(r"(?:venue|会议|期刊)\s*[:：]\s*([^\n,，;；]+)")
    keyword_terms = _extract(r"(?:keyword|关键词)\s*[:：]\s*([^\n,，;；]+)")

    return QueryMetadataFilters(
        years=years,
        source_terms=source_terms,
        title_terms=title_terms,
        author_terms=author_terms,
        venue_terms=venue_terms,
        keyword_terms=keyword_terms,
    )


def _contains_any(haystack: str, needles: tuple[str, ...]) -> bool:
    if not needles:
        return True
    normalized_haystack = _norm(haystack)
    return any(_norm(needle) in normalized_haystack for needle in needles if needle.strip())


def _match_doc(doc: Document, filters: QueryMetadataFilters) -> bool:
    metadata = doc.metadata or {}
    source = str(metadata.get("source", ""))
    title = str(metadata.get("paper_title", ""))
    year = str(metadata.get("paper_year", ""))
    authors = str(metadata.get("paper_authors", ""))
    venue = str(metadata.get("paper_venue", ""))
    keywords = str(metadata.get("paper_keywords", ""))

    if filters.years and year not in filters.years:
        return False
    if filters.source_terms and not _contains_any(source, filters.source_terms):
        return False
    if filters.title_terms and not _contains_any(title, filters.title_terms):
        return False
    if filters.author_terms and not _contains_any(authors, filters.author_terms):
        return False
    if filters.venue_terms and not _contains_any(venue, filters.venue_terms):
        return False
    if filters.keyword_terms and not _contains_any(keywords, filters.keyword_terms):
        return False
    return True


def apply_query_metadata_filter(
    query: str,
    docs: list[Document],
    *,
    enabled: bool = True,
) -> tuple[list[Document], set[str] | None]:
    """
    基于 query 中显式 metadata 约束过滤候选语料。
    无约束或过滤后为空时，自动回退到原语料，避免召回归零。
    """
    if not enabled or not docs:
        return docs, None

    filters = parse_query_metadata_filters(query)
    if not filters.has_constraints:
        return docs, None

    filtered = [doc for doc in docs if _match_doc(doc, filters)]
    if not filtered:
        return docs, None

    allowed_sources = {
        str(doc.metadata.get("source", ""))
        for doc in filtered
        if doc.metadata and doc.metadata.get("source")
    }
    return filtered, allowed_sources or None
