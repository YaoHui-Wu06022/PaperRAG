from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any
import unicodedata

from langchain_core.documents import Document


# 这个模块把原始 parent/chunk 数据提升成 paper 级表示：
# 包括 paper summary、section summary、catalog 行和 citation edge。
@dataclass(frozen=True)
class PaperAssets:
    paper_docs: list[Document]
    section_docs: list[Document]
    catalog_rows: list[dict[str, Any]]
    citation_rows: list[dict[str, Any]]


_TEXT_REPLACEMENTS = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\u001b": "fi",
    "\u001c": "ffi",
    "\u001d": "ff",
    "\u001e": "fl",
    "\u0080": "fl",
    "\u00a0": " ",
}
_ABSTRACT_INLINE_RE = re.compile(
    r"\b(?:abstract|a\s*b\s*s\s*t\s*r\s*a\s*c\s*t)\b\s*[:.]?\s*",
    flags=re.IGNORECASE,
)
_FRONT_MATTER_RE = [
    re.compile(r"^article in press\s*", flags=re.IGNORECASE),
    re.compile(
        r"^pdf download .*? total citations:\s*\d+ .*? total downloads:\s*\d+\s*",
        flags=re.IGNORECASE,
    ),
    re.compile(r"^open access support provided by:?\s*", flags=re.IGNORECASE),
    re.compile(r"^citation in bibtex format\s*", flags=re.IGNORECASE),
]
_NOISE_SECTION_RE = [
    re.compile(r"^(?:article|article in press|research-article)$", flags=re.IGNORECASE),
    re.compile(r"^(?:a\s*r\s*t\s*i\s*c\s*l\s*e\s*i\s*n\s*f\s*o)$", flags=re.IGNORECASE),
    re.compile(r"^(?:g\s*r\s*a\s*p\s*h\s*i\s*c\s*a\s*l\s*a\s*b\s*s\s*t\s*r\s*a\s*c\s*t)$", flags=re.IGNORECASE),
    re.compile(r"^(?:h\s*i\s*g\s*h\s*l\s*i\s*g\s*h\s*t\s*s)$", flags=re.IGNORECASE),
    re.compile(r"^(?:abstract|a\s*b\s*s\s*t\s*r\s*a\s*c\s*t)$", flags=re.IGNORECASE),
    re.compile(r"^(?:keywords|ccs concepts|acknowledg?ement|appendix a\. supplementary data)$", flags=re.IGNORECASE),
]


_ABSTRACT_MARKERS = ("abstract", "摘要")
_REFERENCE_MARKERS = ("references", "reference", "参考文献")


def build_paper_assets(
    parent_docs: list[Document],
    block_rows: list[dict[str, Any]],
    reference_docs: list[Document],
    *,
    source_root: Path | None = None,
    paper_summary_max_chars: int = 2400,
    section_summary_max_chars: int = 700,
) -> PaperAssets:
    """从一份已解析语料中构建全部 paper 级资产。

    输出会刻意拆成四层：
    - paper_docs：粗粒度的 paper-level 检索单元
    - section_docs：供 survey / metadata 路由使用的 section 摘要
    - catalog_rows：供 SQLite / JSONL catalog 查询的结构化元数据
    - citation_rows：从参考文献区解析出来的引用边
    """
    parents_by_doc: dict[str, list[Document]] = defaultdict(list)
    blocks_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    refs_by_doc: dict[str, list[Document]] = defaultdict(list)

    for doc in parent_docs:
        doc_id = str((doc.metadata or {}).get("doc_id", "")).strip()
        if doc_id:
            parents_by_doc[doc_id].append(doc)
    for row in block_rows:
        doc_id = str(row.get("doc_id", "")).strip()
        if doc_id:
            blocks_by_doc[doc_id].append(dict(row))
    for doc in reference_docs:
        doc_id = str((doc.metadata or {}).get("doc_id", "")).strip()
        if doc_id:
            refs_by_doc[doc_id].append(doc)

    paper_docs: list[Document] = []
    section_docs: list[Document] = []
    catalog_rows: list[dict[str, Any]] = []
    citation_rows: list[dict[str, Any]] = []

    all_doc_ids = sorted(set(parents_by_doc) | set(blocks_by_doc) | set(refs_by_doc))
    for doc_id in all_doc_ids:
        doc_parents = sorted(
            parents_by_doc.get(doc_id, []),
            key=lambda item: (
                int((item.metadata or {}).get("page", 0) or 0),
                str((item.metadata or {}).get("section_path", "")),
            ),
        )
        doc_blocks = sorted(
            blocks_by_doc.get(doc_id, []),
            key=lambda item: (int(item.get("page", 0) or 0), str(item.get("block_id", ""))),
        )
        if not doc_parents and not doc_blocks:
            continue

        meta = _collect_doc_metadata(doc_parents, doc_blocks, source_root=source_root)
        doc_citation_rows = _build_citation_rows(
            refs_by_doc.get(doc_id, []),
            source_doc_id=doc_id,
            source_title=meta["title"],
        )
        citation_rows.extend(doc_citation_rows)

        doc_section_docs = _build_section_summary_docs(
            doc_parents,
            meta=meta,
            max_chars=section_summary_max_chars,
        )
        section_docs.extend(doc_section_docs)

        citation_titles = [
            str(row.get("cited_title", "")).strip()
            for row in doc_citation_rows
            if str(row.get("cited_title", "")).strip()
        ]
        abstract_text = _extract_abstract(doc_blocks, doc_section_docs)
        paper_summary = _build_paper_summary(
            meta=meta,
            abstract_text=abstract_text,
            section_docs=doc_section_docs,
            citation_titles=citation_titles,
            max_chars=paper_summary_max_chars,
        )
        paper_docs.append(
            Document(
                page_content=paper_summary,
                metadata={
                    "doc_id": doc_id,
                    "source": meta["source"],
                    "source_path": meta["source_path"],
                    "title": meta["title"],
                    "year": meta["year"],
                    "authors": meta["authors_text"],
                    "venue": meta["venue"],
                    "keywords": meta["keywords_text"],
                    "paper_title": meta["title"],
                    "paper_year": meta["year"],
                    "paper_authors": meta["authors_text"],
                    "paper_venue": meta["venue"],
                    "paper_keywords": meta["keywords_text"],
                    "paper_language": meta["language"],
                    "paper_abstract": abstract_text,
                    "paper_summary": paper_summary,
                    "paper_level": True,
                    "representation": "paper_summary",
                    "total_pages": meta["total_pages"],
                    "citation_count": len(doc_citation_rows),
                    "section_names": meta["section_names"],
                },
            )
        )
        catalog_rows.append(
            {
                "doc_id": doc_id,
                "source": meta["source"],
                "source_path": meta["source_path"],
                "title": meta["title"],
                "abstract": abstract_text,
                "year": meta["year"],
                "venue": meta["venue"],
                "authors_text": meta["authors_text"],
                "normalized_authors": meta["normalized_authors"],
                "keywords_text": meta["keywords_text"],
                "keywords": meta["keywords"],
                "language": meta["language"],
                "total_pages": meta["total_pages"],
                "paper_summary": paper_summary,
                "section_names": meta["section_names"],
                "citation_count": len(doc_citation_rows),
                "citation_titles": citation_titles[:128],
            }
        )

    return PaperAssets(
        paper_docs=paper_docs,
        section_docs=section_docs,
        catalog_rows=catalog_rows,
        citation_rows=_dedupe_rows(citation_rows, key="edge_id"),
    )


def _collect_doc_metadata(
    parent_docs: list[Document],
    block_rows: list[dict[str, Any]],
    *,
    source_root: Path | None,
) -> dict[str, Any]:
    """为单篇论文收集一份稳定的元数据快照。

    这里优先使用已经挂在解析文档上的 metadata，再用 block 行补齐缺口，
    这样后续层就不需要直接依赖底层 MinerU 细节。
    """
    base_meta: dict[str, Any] = {}
    for doc in parent_docs:
        base_meta = dict(doc.metadata or {})
        if base_meta:
            break
    if not base_meta and block_rows:
        base_meta = dict(block_rows[0])

    source = str(base_meta.get("source", "")).strip()
    source_path = str((source_root / source).resolve()) if source_root and source else source
    title = _clean_text(base_meta.get("paper_title") or base_meta.get("title") or source)
    authors_text = _clean_text(base_meta.get("paper_authors") or base_meta.get("authors"))
    year = _clean_text(base_meta.get("paper_year") or base_meta.get("year"))
    venue = _clean_text(base_meta.get("paper_venue") or base_meta.get("venue"))
    keywords_text = _clean_text(base_meta.get("paper_keywords") or base_meta.get("keywords"))
    language = _clean_text(base_meta.get("paper_language"))

    total_pages = 0
    for row in block_rows:
        try:
            total_pages = max(total_pages, int(row.get("page", 0) or 0))
        except (TypeError, ValueError):
            continue
    if total_pages <= 0:
        for doc in parent_docs:
            try:
                total_pages = max(total_pages, int((doc.metadata or {}).get("page_end", 0) or 0))
            except (TypeError, ValueError):
                continue

    section_names: list[str] = []
    seen = set()
    for doc in parent_docs:
        section_name = _top_section_name(str((doc.metadata or {}).get("section_path", "")))
        if not section_name or _is_reference_section(section_name) or _is_noise_section_name(section_name):
            continue
        key = section_name.lower()
        if key in seen:
            continue
        seen.add(key)
        section_names.append(section_name)

    return {
        "source": source,
        "source_path": source_path,
        "title": title,
        "authors_text": authors_text,
        "normalized_authors": _normalize_authors(authors_text),
        "year": year,
        "venue": venue,
        "keywords_text": keywords_text,
        "keywords": _normalize_keywords(keywords_text),
        "language": language,
        "total_pages": total_pages,
        "section_names": section_names,
    }


def _build_section_summary_docs(
    parent_docs: list[Document],
    *,
    meta: dict[str, Any],
    max_chars: int,
) -> list[Document]:
    """把 parent docs 聚合成按顶层 section 划分的摘要文档。"""
    grouped: dict[str, list[Document]] = defaultdict(list)
    for doc in parent_docs:
        section_name = _top_section_name(str((doc.metadata or {}).get("section_path", "")))
        if not section_name or _is_reference_section(section_name) or _is_noise_section_name(section_name):
            continue
        grouped[section_name].append(doc)

    section_docs: list[Document] = []
    for section_name, docs in sorted(
        grouped.items(),
        key=lambda item: min(int((doc.metadata or {}).get("page", 1) or 1) for doc in item[1]),
    ):
        summary = _clean_section_summary_text(
            _summarize_docs(docs, max_chars=max_chars),
            meta=meta,
            section_name=section_name,
        )
        if not summary:
            continue
        pages = [
            int((doc.metadata or {}).get("page", 1) or 1)
            for doc in docs
            if str((doc.metadata or {}).get("page", "")).strip()
        ]
        page_range = f"{min(pages)}-{max(pages)}" if pages else ""
        section_docs.append(
            Document(
                page_content=summary,
                metadata={
                    "doc_id": str((docs[0].metadata or {}).get("doc_id", "")),
                    "source": meta["source"],
                    "source_path": meta["source_path"],
                    "title": meta["title"],
                    "year": meta["year"],
                    "authors": meta["authors_text"],
                    "venue": meta["venue"],
                    "keywords": meta["keywords_text"],
                    "paper_title": meta["title"],
                    "paper_year": meta["year"],
                    "paper_authors": meta["authors_text"],
                    "paper_venue": meta["venue"],
                    "paper_keywords": meta["keywords_text"],
                    "paper_language": meta["language"],
                    "paper_level": True,
                    "representation": "section_summary",
                    "section_name": section_name,
                    "section_path": section_name,
                    "page": min(pages) if pages else 1,
                    "page_end": max(pages) if pages else 1,
                    "page_range": page_range,
                },
            )
        )
    return section_docs


def _build_citation_rows(
    reference_docs: list[Document],
    *,
    source_doc_id: str,
    source_title: str,
) -> list[dict[str, Any]]:
    """把解析后的参考文献 chunk 转成 citation edge 行数据。"""
    rows: list[dict[str, Any]] = []
    seen = set()
    for doc in reference_docs:
        text = _clean_text(doc.page_content)
        if len(text) < 12:
            continue
        key = hashlib.sha1(f"{source_doc_id}::{text}".encode("utf-8")).hexdigest()[:16]
        if key in seen:
            continue
        seen.add(key)
        cited_authors, cited_year, cited_title = _parse_reference_entry(text)
        rows.append(
            {
                "edge_id": key,
                "doc_id": source_doc_id,
                "source_doc_id": source_doc_id,
                "source_title": source_title,
                "cited_title": cited_title,
                "cited_year": cited_year,
                "cited_authors": cited_authors,
                "citation_text": text[:1200],
            }
        )
    return rows


def _parse_reference_entry(text: str) -> tuple[str, str, str]:
    cleaned = _clean_text(text)
    year_match = re.search(r"\b(19\d{2}|20\d{2})\b", cleaned)
    authors = ""
    year = ""
    title = cleaned[:240]
    if year_match:
        year = year_match.group(1)
        prefix = cleaned[: year_match.start()].strip(" .,;:-")
        suffix = cleaned[year_match.end() :].strip(" .,;:-")
        authors = prefix[:200]
        title = re.split(r"\.\s+|\?\s+|!\s+", suffix, maxsplit=1)[0].strip(" .,;:-") or title
    return authors[:200], year, title[:260]


def _extract_abstract(block_rows: list[dict[str, Any]], section_docs: list[Document]) -> str:
    abstract_parts: list[str] = []
    in_abstract_section = False
    for row in block_rows:
        section_path = str(row.get("section_path", "")).strip().lower()
        raw_text = row.get("text")
        cleaned_text = _clean_text(raw_text)
        section_has_abstract = _is_abstract_section(section_path)
        text = _extract_abstract_candidate(raw_text)
        if section_has_abstract:
            in_abstract_section = True
            if text:
                abstract_parts.append(text)
                continue
            if _is_abstract_heading_text(cleaned_text):
                continue
            if _looks_like_abstract_stop_text(cleaned_text):
                break
            cleaned_text = _strip_front_matter_text(cleaned_text)
            if cleaned_text:
                abstract_parts.append(cleaned_text)
            continue
        elif text and len(text) >= 120:
            in_abstract_section = True
            abstract_parts.append(text)
            continue
        elif in_abstract_section:
            if _looks_like_abstract_stop_text(cleaned_text):
                break
            cleaned_text = _strip_front_matter_text(cleaned_text)
            if cleaned_text:
                abstract_parts.append(cleaned_text)
            continue
    if abstract_parts:
        return _truncate(" ".join(abstract_parts), 1200)

    for doc in section_docs:
        section_name = str((doc.metadata or {}).get("section_name", "")).strip().lower()
        if _is_abstract_section(section_name):
            return _truncate(_extract_abstract_candidate(doc.page_content), 1200)

    lead_parts: list[str] = []
    for row in block_rows:
        if row.get("is_reference"):
            continue
        text = _extract_abstract_candidate(row.get("text")) or _strip_front_matter_text(
            _clean_text(row.get("text"))
        )
        if not text:
            continue
        lead_parts.append(text)
        if len(" ".join(lead_parts)) >= 900:
            break
    return _truncate(" ".join(lead_parts), 1200)


def _build_paper_summary(
    *,
    meta: dict[str, Any],
    abstract_text: str,
    section_docs: list[Document],
    citation_titles: list[str],
    max_chars: int,
) -> str:
    """组装面向检索的 paper summary 文本。

    这份 summary 会把结构化元数据、摘要、若干重要 section 摘要，
    以及可选的引用提示拼成一份可检索文档。
    """
    section_lines = []
    for doc in section_docs[:8]:
        section_name = str((doc.metadata or {}).get("section_name", "")).strip() or "正文"
        section_text = _truncate(_clean_text(doc.page_content), 220)
        if _should_skip_section_in_summary(section_name, section_text, meta=meta):
            continue
        section_lines.append(f"- {section_name}: {section_text}")

    header_parts = [
        f"标题: {meta['title']}",
        f"作者: {meta['authors_text']}" if meta["authors_text"] else "",
        f"年份: {meta['year']}" if meta["year"] else "",
        f"来源: {meta['venue']}" if meta["venue"] else "",
        f"关键词: {meta['keywords_text']}" if meta["keywords_text"] else "",
        f"摘要: {abstract_text}" if abstract_text else "",
    ]
    header = "\n".join(part for part in header_parts if part)
    sections = "\n".join(section_lines)
    citations = ""
    if citation_titles:
        sampled_titles = _select_citation_titles_for_summary(citation_titles)
        sampled = ", ".join(sampled_titles[:10])
        if sampled:
            citations = f"\n引用线索: {sampled}"
    summary = "\n".join(part for part in (header, sections, citations) if part)
    return _truncate(summary, max_chars)


def _summarize_docs(docs: list[Document], *, max_chars: int) -> str:
    snippets: list[str] = []
    for doc in docs:
        text = _clean_text(doc.page_content)
        if not text:
            continue
        snippets.append(text)
        if len(" ".join(snippets)) >= max_chars:
            break
    return _truncate(" ".join(snippets), max_chars)


def _top_section_name(section_path: str) -> str:
    path = _clean_text(section_path)
    if not path:
        return ""
    first = path.split(" > ", 1)[0].strip()
    return first if first else path


def _is_reference_section(section_name: str) -> bool:
    low = section_name.strip().lower()
    return any(marker in low for marker in _REFERENCE_MARKERS)


def _is_noise_section_name(section_name: str) -> bool:
    cleaned = _clean_text(section_name)
    low = cleaned.lower()
    if not cleaned:
        return True
    if any(pattern.match(cleaned) for pattern in _NOISE_SECTION_RE):
        return True
    if "et al" in low:
        return True
    if any(token in low for token in ("university", "department", "institute", "hospital")):
        return True
    if re.match(r"^\d+\s+[A-Z][A-Za-z].*(?:university|department|institute|hospital)", cleaned):
        return True
    return False


def _is_generic_body_section(section_name: str) -> bool:
    cleaned = _clean_text(section_name)
    low = cleaned.lower()
    if low in {"main", "body"}:
        return True
    if not re.search(r"[A-Za-z0-9]", cleaned) and 1 <= len(cleaned) <= 8:
        return True
    return False


def _extract_abstract_candidate(value: Any) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    match = _ABSTRACT_INLINE_RE.search(text)
    if match:
        return _strip_front_matter_text(text[match.end() :])
    return ""


def _is_abstract_section(value: str) -> bool:
    text = _clean_text(value)
    if not text:
        return False
    low = text.lower()
    return any(marker in low for marker in _ABSTRACT_MARKERS) or bool(
        _ABSTRACT_INLINE_RE.search(text)
    )


def _is_abstract_heading_text(text: str) -> bool:
    cleaned = _clean_text(text)
    if not cleaned:
        return False
    return bool(_ABSTRACT_INLINE_RE.fullmatch(cleaned))


def _looks_like_abstract_stop_text(text: str) -> bool:
    cleaned = _clean_text(text)
    low = cleaned.lower()
    if not cleaned:
        return False
    if re.match(r"^\d+(?:\.\d+)*[\.\)]?\s+\S+", cleaned):
        return True
    if re.match(r"^(?:introduction|background|related work|methods?|materials|results?|discussion|conclusion)\b", low):
        return True
    if re.match(r"^(?:contents lists available|keywords:|ccs concepts|acknowledg)", low):
        return True
    if re.search(r"https?://|doi\.org", low):
        return True
    return False


def _clean_section_summary_text(text: str, *, meta: dict[str, Any], section_name: str) -> str:
    cleaned = _clean_text(text)
    if not cleaned:
        return ""
    if _is_generic_body_section(section_name):
        cleaned = _strip_front_matter_text(
            cleaned,
            title=meta.get("title", ""),
            authors=meta.get("authors_text", ""),
        )
    return cleaned


def _should_skip_section_in_summary(
    section_name: str,
    section_text: str,
    *,
    meta: dict[str, Any],
) -> bool:
    cleaned = _clean_text(section_text)
    if not cleaned:
        return True
    if _is_generic_body_section(section_name):
        lowered = cleaned.lower()
        if any(
            token in lowered
            for token in (
                "article in press",
                "pdf download",
                "total citations",
                "total downloads",
                "open access support provided by",
            )
        ):
            return True
        title = _clean_text(meta.get("title", ""))
        if title and cleaned.startswith(title):
            return True
    return False


def _select_citation_titles_for_summary(citation_titles: list[str]) -> list[str]:
    selected: list[str] = []
    for title in citation_titles:
        cleaned = _clean_text(title)
        low = cleaned.lower()
        if not cleaned:
            continue
        if "http://" in low or "https://" in low or "doi" in low:
            continue
        if len(cleaned) < 12:
            continue
        selected.append(cleaned)
    return selected


def _strip_front_matter_text(
    text: str,
    *,
    title: str = "",
    authors: str = "",
) -> str:
    cleaned = _clean_text(text)
    if not cleaned:
        return ""
    for pattern in _FRONT_MATTER_RE:
        cleaned = pattern.sub("", cleaned).strip()
    for prefix in (title, authors):
        prefix_clean = _clean_text(prefix)
        if prefix_clean and cleaned.startswith(prefix_clean):
            cleaned = cleaned[len(prefix_clean) :].strip(" .,:;-")
    cleaned = re.sub(
        r"^(?:we are providing an unedited version of this manuscript .*? legal disclaimers apply\.?)\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"^(?:if this paper is publishing under a transparent peer review model .*? final article\.?)\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"^(?:copyright .*?)\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip(" .,:;-")
    return cleaned


def _normalize_authors(text: str) -> list[str]:
    cleaned = _clean_text(text)
    if not cleaned:
        return []
    parts = [
        _clean_text(item)
        for item in re.split(r"[;,，、]|(?:\band\b)|(?:\bet al\b)", cleaned, flags=re.IGNORECASE)
        if _clean_text(item)
    ]
    return parts[:24]


def _normalize_keywords(text: str) -> list[str]:
    cleaned = _clean_text(text)
    if not cleaned:
        return []
    return [
        item
        for item in (
            _clean_text(part)
            for part in re.split(r"[;,，、|/]", cleaned)
        )
        if item
    ][:32]


def _truncate(text: str, limit: int) -> str:
    normalized = _clean_text(text)
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(limit - 3, 0)].rstrip() + "..."


def _clean_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    for src, dst in _TEXT_REPLACEMENTS.items():
        text = text.replace(src, dst)
    text = text.replace("\\textcircled", " ")
    text = text.replace("\\", " ")
    text = re.sub(r"[{}$^_~|]", " ", text)
    text = re.sub(r"\s*&\s*", " & ", text)
    text = "".join(ch for ch in text if ch >= " ")
    return " ".join(text.replace("\r", " ").replace("\n", " ").split())


def _dedupe_rows(rows: list[dict[str, Any]], *, key: str) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen = set()
    for row in rows:
        value = str(row.get(key, "")).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        deduped.append(row)
    return deduped
