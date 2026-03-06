from __future__ import annotations

import re


REFERENCE_SECTION_TOKENS = (
    "references",
    "bibliography",
    "reference",
    "参考文献",
    "参考资料",
)

REFERENCE_VENUE_HINTS = (
    "arxiv",
    "ieee",
    "acm",
    "springer",
    "nature",
    "science",
    "cvpr",
    "iccv",
    "eccv",
    "neurips",
    "icml",
    "acl",
    "emnlp",
    "naacl",
    "aaai",
    "ijcai",
    "kdd",
    "miccai",
    "tpami",
    "jmlr",
    "journal",
    "conference",
    "proceedings",
    "transactions",
)


def _normalize_space(text: str) -> str:
    return " ".join(str(text or "").replace("\t", " ").split())


def _clean_heading_text(text: str) -> str:
    first_line = str(text or "").strip().splitlines()[0] if str(text or "").strip() else ""
    return _normalize_space(first_line).strip(":- ").strip()


def normalize_reference_heading(text: str) -> str:
    heading = _clean_heading_text(text)
    lower = heading.lower()
    if lower.startswith("bibliography"):
        return "Bibliography"
    if lower.startswith("references") or lower.startswith("reference"):
        return "References"
    if heading.startswith("参考文献"):
        return "参考文献"
    if heading.startswith("参考资料"):
        return "参考资料"
    return heading


def is_reference_heading(text: str) -> bool:
    heading = normalize_reference_heading(text)
    lowered = heading.lower()
    return any(
        lowered.startswith(token)
        for token in ("references", "bibliography", "reference")
    ) or heading.startswith(("参考文献", "参考资料"))


def is_reference_section(section_path: str) -> bool:
    lowered = str(section_path or "").lower()
    return any(token in lowered for token in REFERENCE_SECTION_TOKENS)


def _reference_year_count(text: str) -> int:
    return len(re.findall(r"\b(?:19|20)\d{2}[a-z]?\b", text.lower()))


def _has_reference_author_pattern(text: str) -> bool:
    normalized = _normalize_space(text)
    lower = normalized.lower()
    return bool(
        "et al" in lower
        or re.search(r"(?:\b[A-Z]\.\s*){2,}", normalized)
        or re.search(r"\b[A-Z][a-z'`-]+,\s*(?:[A-Z]\.)", normalized)
    )


def _has_reference_venue_hint(text: str) -> bool:
    lowered = _normalize_space(text).lower()
    if any(token in lowered for token in REFERENCE_VENUE_HINTS):
        return True
    return bool(
        re.search(
            r"\b(?:doi|arxiv|pp?\.|vol\.|no\.|proc\.|in\s+proceedings)\b",
            lowered,
        )
    )


def reference_signal_score(text: str) -> int:
    normalized = _normalize_space(text)
    if not normalized:
        return 0

    score = 0
    if re.match(r"^(?:\[\d+\]|\(\d+\)|\d+\.)\s+", normalized):
        score += 1

    year_count = _reference_year_count(normalized)
    score += min(year_count, 2)

    if _has_reference_author_pattern(normalized):
        score += 1
    if _has_reference_venue_hint(normalized):
        score += 1
    return score


def looks_like_reference_entry(text: str) -> bool:
    normalized = _normalize_space(text)
    if len(normalized) < 20:
        return False

    has_prefix = bool(re.match(r"^(?:\[\d+\]|\(\d+\)|\d+\.)\s+", normalized))
    score = reference_signal_score(normalized)
    if has_prefix:
        return score >= 3
    return score >= 4


def looks_like_reference_continuation(text: str) -> bool:
    normalized = _normalize_space(text)
    if len(normalized) < 20:
        return False
    if is_reference_heading(normalized):
        return True
    return reference_signal_score(normalized) >= 3 and _reference_year_count(normalized) >= 1


def looks_like_reference_chunk(text: str) -> bool:
    return (
        is_reference_heading(text)
        or looks_like_reference_entry(text)
        or looks_like_reference_continuation(text)
    )


def looks_like_body_paragraph_after_references(text: str) -> bool:
    normalized = _normalize_space(text)
    if len(normalized) < 40:
        return False
    if looks_like_reference_chunk(normalized):
        return False

    lowered = normalized.lower()
    if re.match(
        r"^(?:appendix|supplement|supplementary|acknowledg|conclusion|discussion)\b",
        lowered,
    ):
        return True
    if re.match(r"^(?:\d+(?:\.\d+)*|[A-Z]|[IVXLCM]+)[\.\)]?\s+[A-Z]", normalized):
        return True
    return _reference_year_count(normalized) == 0 and bool(re.search(r"[.!?]", normalized))
