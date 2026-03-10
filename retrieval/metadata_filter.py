from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re

from langchain_core.documents import Document


_YEAR_MIN = 1950
_YEAR_MAX = 2100
_CN_NUMERAL_MAP = {
    "\u96f6": 0,
    "\u4e00": 1,
    "\u4e8c": 2,
    "\u4e24": 2,
    "\u4e09": 3,
    "\u56db": 4,
    "\u4e94": 5,
    "\u516d": 6,
    "\u4e03": 7,
    "\u516b": 8,
    "\u4e5d": 9,
    "\u5341": 10,
}
_FIELD_LABELS = {
    "source": ("source", "paper", "\u6587\u732e"),
    "title": ("title", "\u6807\u9898"),
    "author": ("author", "\u4f5c\u8005"),
    "venue": ("venue", "\u4f1a\u8bae", "\u671f\u520a"),
    "keyword": ("keyword", "\u5173\u952e\u8bcd"),
}
_ALL_FIELD_LABELS = tuple(
    label
    for labels in _FIELD_LABELS.values()
    for label in labels
)

# 这个模块负责解析半结构化检索约束，例如：
# `author: He venue: CVPR 2016` 或 `2020-2023`。
# 这里故意采用 fail-open 策略：如果过滤后为空，
# 检索会退回原始语料，而不是直接返回空结果。


def _norm(text: str) -> str:
    return " ".join(str(text or "").lower().split())


@dataclass(frozen=True)
class QueryMetadataFilters:
    years: tuple[str, ...] = ()
    source_terms: tuple[str, ...] = ()
    title_terms: tuple[str, ...] = ()
    author_terms: tuple[str, ...] = ()
    venue_terms: tuple[str, ...] = ()
    keyword_terms: tuple[str, ...] = ()

    @property
    def has_constraints(self) -> bool:
        return any(
            (
                self.years,
                self.source_terms,
                self.title_terms,
                self.author_terms,
                self.venue_terms,
                self.keyword_terms,
            )
        )


@dataclass(frozen=True)
class QueryMetadataFilterResult:
    docs: list[Document]
    allowed_doc_ids: set[str] | None
    filters: QueryMetadataFilters
    applied: bool
    milvus_expr: str | None = None


def _parse_small_int(text: str) -> int | None:
    value = str(text or "").strip()
    if not value:
        return None
    if value.isdigit():
        return int(value)
    if value in _CN_NUMERAL_MAP:
        return _CN_NUMERAL_MAP[value]
    if value == "\u5341":
        return 10
    if len(value) == 2 and value.endswith("\u5341") and value[0] in _CN_NUMERAL_MAP:
        return _CN_NUMERAL_MAP[value[0]] * 10
    if len(value) == 2 and value.startswith("\u5341") and value[1] in _CN_NUMERAL_MAP:
        return 10 + _CN_NUMERAL_MAP[value[1]]
    if (
        len(value) == 3
        and value[1] == "\u5341"
        and value[0] in _CN_NUMERAL_MAP
        and value[2] in _CN_NUMERAL_MAP
    ):
        return _CN_NUMERAL_MAP[value[0]] * 10 + _CN_NUMERAL_MAP[value[2]]
    return None


def _expand_years(start: int, end: int) -> set[int]:
    low = max(min(start, end), _YEAR_MIN)
    high = min(max(start, end), _YEAR_MAX)
    if low > high:
        return set()
    return set(range(low, high + 1))


def _parse_year_constraints(text: str) -> tuple[str, ...]:
    working = f" {str(text or '').strip()} "
    years: set[int] = set()
    current_year = datetime.now().year

    def _consume(pattern: str, handler) -> None:
        nonlocal working
        matches = list(re.finditer(pattern, working, flags=re.IGNORECASE))
        for match in matches:
            handler(match)
        working = re.sub(pattern, " ", working, flags=re.IGNORECASE)

    _consume(
        r"\b(19\d{2}|20\d{2})\s*(?:-|–|—|~|\u81f3|\u5230|to)\s*(19\d{2}|20\d{2})\b",
        lambda match: years.update(
            _expand_years(int(match.group(1)), int(match.group(2)))
        ),
    )
    _consume(
        r"\bbetween\s+(19\d{2}|20\d{2})\s+and\s+(19\d{2}|20\d{2})\b",
        lambda match: years.update(
            _expand_years(int(match.group(1)), int(match.group(2)))
        ),
    )
    _consume(
        r"\bafter\s+(19\d{2}|20\d{2})\b",
        lambda match: years.update(
            _expand_years(int(match.group(1)) + 1, current_year)
        ),
    )
    _consume(
        r"\b(19\d{2}|20\d{2})\s*(?:\u4e4b\u540e|\u4ee5\u540e)\b",
        lambda match: years.update(
            _expand_years(int(match.group(1)) + 1, current_year)
        ),
    )
    _consume(
        r"\b(?:since|from)\s+(19\d{2}|20\d{2})\b",
        lambda match: years.update(_expand_years(int(match.group(1)), current_year)),
    )
    _consume(
        r"\b(19\d{2}|20\d{2})\s*(?:\u4ee5\u6765|\u8d77)\b",
        lambda match: years.update(_expand_years(int(match.group(1)), current_year)),
    )
    _consume(
        r"\bbefore\s+(19\d{2}|20\d{2})\b",
        lambda match: years.update(
            _expand_years(_YEAR_MIN, int(match.group(1)) - 1)
        ),
    )
    _consume(
        r"\b(19\d{2}|20\d{2})\s*(?:\u4e4b\u524d|\u4ee5\u524d)\b",
        lambda match: years.update(
            _expand_years(_YEAR_MIN, int(match.group(1)) - 1)
        ),
    )

    recent_pattern = (
        r"(?:\u8fd1|\u6700\u8fd1)\s*([0-9\u4e00\u4e8c\u4e24\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341]{1,3})\s*\u5e74"
        r"|(?:last|past)\s+(\d{1,2})\s+years?"
    )

    def _handle_recent(match) -> None:
        raw_span = match.group(1) or match.group(2)
        span = _parse_small_int(raw_span)
        if span is None or span <= 0:
            return
        start_year = max(_YEAR_MIN, current_year - span + 1)
        years.update(_expand_years(start_year, current_year))

    _consume(recent_pattern, _handle_recent)

    for item in re.findall(r"\b(19\d{2}|20\d{2})\b", working):
        years.add(int(item))

    return tuple(str(year) for year in sorted(years))


def _strip_outer_punctuation(value: str) -> str:
    return re.sub(
        r"^[\s\"'`.,;:()\[\]{}<>/\u3002\uff0c\u3001\uff1b\uff1a\uff08\uff09]+|[\s\"'`.,;:()\[\]{}<>/\u3002\uff0c\u3001\uff1b\uff1a\uff08\uff09]+$",
        "",
        str(value or ""),
    ).strip()


def _trim_trailing_year_clauses(value: str) -> str:
    cleaned = str(value or "").strip()
    trailing_patterns = (
        r"\s+(19\d{2}|20\d{2})\s*(?:-|–|—|~|\u81f3|\u5230|to)\s*(19\d{2}|20\d{2})$",
        r"\s+(?:between\s+(19\d{2}|20\d{2})\s+and\s+(19\d{2}|20\d{2}))$",
        r"\s+(?:after|since|from|before)\s+(19\d{2}|20\d{2})$",
        r"\s+(19\d{2}|20\d{2})\s*(?:\u4e4b\u540e|\u4ee5\u540e|\u4ee5\u6765|\u8d77|\u4e4b\u524d|\u4ee5\u524d)$",
        r"\s+(?:\u8fd1|\u6700\u8fd1)\s*[0-9\u4e00\u4e8c\u4e24\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341]{1,3}\s*\u5e74$",
        r"\s+(?:last|past)\s+\d{1,2}\s+years?$",
        r"\s+(19\d{2}|20\d{2})$",
    )
    while cleaned:
        updated = cleaned
        for pattern in trailing_patterns:
            updated = re.sub(pattern, "", updated, flags=re.IGNORECASE).strip()
        if updated == cleaned:
            break
        cleaned = updated
    return cleaned


def _extract_terms(text: str, labels: tuple[str, ...]) -> tuple[str, ...]:
    label_group = "|".join(re.escape(label) for label in labels)
    stop_group = "|".join(re.escape(label) for label in _ALL_FIELD_LABELS)
    pattern = (
        rf"(?:{label_group})\s*[:\uff1a]\s*(.+?)"
        rf"(?=(?:\s+(?:{stop_group})\s*[:\uff1a])|[,，;；\n]|$)"
    )
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    cleaned: list[str] = []
    for item in matches:
        value = _strip_outer_punctuation(item)
        value = _trim_trailing_year_clauses(value)
        value = _strip_outer_punctuation(value)
        if value:
            cleaned.append(" ".join(value.split()))
    return tuple(cleaned)


def parse_query_metadata_filters(query: str) -> QueryMetadataFilters:
    text = str(query or "").strip()
    years = _parse_year_constraints(text)
    return QueryMetadataFilters(
        years=years,
        source_terms=_extract_terms(text, _FIELD_LABELS["source"]),
        title_terms=_extract_terms(text, _FIELD_LABELS["title"]),
        author_terms=_extract_terms(text, _FIELD_LABELS["author"]),
        venue_terms=_extract_terms(text, _FIELD_LABELS["venue"]),
        keyword_terms=_extract_terms(text, _FIELD_LABELS["keyword"]),
    )


def _contains_any(haystack: str, needles: tuple[str, ...]) -> bool:
    if not needles:
        return True
    normalized_haystack = _norm(haystack)
    return any(_norm(needle) in normalized_haystack for needle in needles if needle.strip())


def _match_doc(doc: Document, filters: QueryMetadataFilters) -> bool:
    metadata = doc.metadata or {}
    source = str(metadata.get("source", ""))
    title = " ".join(
        part
        for part in (
            str(metadata.get("title", "")),
            str(metadata.get("paper_title", "")),
        )
        if part.strip()
    )
    year = str(metadata.get("year", metadata.get("paper_year", ""))).strip()
    authors = " ".join(
        part
        for part in (
            str(metadata.get("authors", "")),
            str(metadata.get("paper_authors", "")),
        )
        if part.strip()
    )
    venue = " ".join(
        part
        for part in (
            str(metadata.get("venue", "")),
            str(metadata.get("paper_venue", "")),
        )
        if part.strip()
    )
    keywords = " ".join(
        part
        for part in (
            str(metadata.get("keywords", "")),
            str(metadata.get("paper_keywords", "")),
        )
        if part.strip()
    )

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


def _quote_milvus_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def build_doc_id_expr(doc_ids: set[str] | None) -> str | None:
    """构造一个只允许命中指定论文 doc_id 的 Milvus 过滤表达式。"""
    if not doc_ids:
        return None
    normalized = sorted({str(item).strip() for item in doc_ids if str(item).strip()})
    if not normalized:
        return None
    quoted = ", ".join(_quote_milvus_string(item) for item in normalized)
    return f"doc_id in [{quoted}]"


def apply_query_metadata_filter(
    query: str,
    docs: list[Document],
    *,
    enabled: bool = True,
) -> QueryMetadataFilterResult:
    """把 query 中解析出的元数据约束应用到内存语料上。

    如果没有识别到约束，或者过滤后会把所有文档都删光，
    就直接返回原始语料，避免把检索链路做得过于脆弱。
    """
    if not enabled or not docs:
        return QueryMetadataFilterResult(
            docs=docs,
            allowed_doc_ids=None,
            filters=QueryMetadataFilters(),
            applied=False,
            milvus_expr=None,
        )

    filters = parse_query_metadata_filters(query)
    if not filters.has_constraints:
        return QueryMetadataFilterResult(
            docs=docs,
            allowed_doc_ids=None,
            filters=filters,
            applied=False,
            milvus_expr=None,
        )

    filtered = [doc for doc in docs if _match_doc(doc, filters)]
    if not filtered:
        return QueryMetadataFilterResult(
            docs=docs,
            allowed_doc_ids=None,
            filters=filters,
            applied=False,
            milvus_expr=None,
        )

    allowed_doc_ids = {
        str(doc.metadata.get("doc_id", "")).strip()
        for doc in filtered
        if doc.metadata and str(doc.metadata.get("doc_id", "")).strip()
    }
    return QueryMetadataFilterResult(
        docs=filtered,
        allowed_doc_ids=allowed_doc_ids or None,
        filters=filters,
        applied=True,
        milvus_expr=build_doc_id_expr(allowed_doc_ids or None),
    )
