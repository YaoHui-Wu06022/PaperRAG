from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import re
import time
from typing import Any, Callable
import zipfile
import unicodedata

from langchain_core.documents import Document
import requests

from ingestion.reference_detection import (
    is_reference_heading,
    is_reference_section,
    looks_like_body_paragraph_after_references,
    looks_like_reference_continuation,
    looks_like_reference_entry,
    normalize_reference_heading,
)


# This module is the PDF parsing bridge:
# MinerU content_list -> normalized blocks -> chunk input documents.
VENUE_PATTERNS = [
    "arxiv",
    "ieee",
    "acm",
    "springer",
    "nature",
    "science",
    "talanta",
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
]

_GENERIC_FRONT_MATTER_PATTERNS = (
    r"^article in press$",
    r"^pdf download\b",
    r"^download\b",
    r"^open access support provided by",
    r"^citation in bibtex",
    r"^handling editor:\s*",
    r"^a\s*r\s*t\s*i\s*c\s*l\s*e\s*i\s*n\s*f\s*o$",
    r"^a\s*b\s*s\s*t\s*r\s*a\s*c\s*t$",
    r"^published:\s*",
    r"^received:\s*",
    r"^accepted:\s*",
    r"^available online\b",
    r"^supplementary material\b",
    r"^total citations?\b",
)
_PUBLICATION_HINTS = (
    "conference",
    "proceedings",
    "journal",
    "communications",
    "transaction",
    "transactions",
    "published",
    "doi",
    "arxiv",
    "vol.",
    "volume",
    "issue",
)
_AFFILIATION_HINTS = (
    "university",
    "college",
    "school",
    "department",
    "institute",
    "laboratory",
    "hospital",
    "faculty",
    "momenta",
    "microsoft research",
)
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


@dataclass
class PdfLoadResult:
    documents: list[Document]
    blocks: list[dict[str, Any]]


def _normalize_space(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    for src, dst in _TEXT_REPLACEMENTS.items():
        normalized = normalized.replace(src, dst)
    normalized = normalized.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    normalized = "".join(
        ch
        for ch in normalized
        if ch >= " " or ch in {"\n", "\t"}
    )
    return " ".join(normalized.split())


def _is_generic_front_matter(line: str) -> bool:
    low = _normalize_space(line).lower()
    if not low:
        return False
    if any(re.search(pattern, low) for pattern in _GENERIC_FRONT_MATTER_PATTERNS):
        return True
    if "total downloads" in low or "total citations" in low:
        return True
    if low.startswith(("http://", "https://", "www.")):
        return True
    return False


def _clean_filename_title(pdf_path: Path) -> str:
    stem = pdf_path.stem
    parts = [part.strip() for part in stem.split(" - ") if part.strip()]
    if len(parts) >= 3 and re.fullmatch(r"(19\d{2}|20\d{2})", parts[1]):
        return parts[-1]
    if len(parts) >= 2:
        return parts[-1]
    return stem


def _looks_like_title(line: str) -> bool:
    line = _normalize_space(line)
    low = line.lower()
    if len(line) < 8 or len(line) > 220:
        return False
    if _is_generic_front_matter(line):
        return False
    if low.startswith(("abstract", "keywords", "introduction")):
        return False
    if re.search(r"\bdoi\b|http[s]?://|www\.", low):
        return False
    if "@" in line:
        return False
    if re.search(r"\b(total citations|total downloads)\b", low):
        return False
    return True


def _looks_like_author_line(line: str) -> bool:
    line = _normalize_space(line)
    low = line.lower()
    if not line or len(line) > 420 or _is_generic_front_matter(line):
        return False
    if low.startswith(("abstract", "keywords", "introduction")):
        return False
    if re.search(r"\bdoi\b|http[s]?://|www\.", low):
        return False
    if re.search(r"\b(received|accepted|published)\b", low):
        return False
    if (
        (any(pattern in low for pattern in VENUE_PATTERNS) or any(hint in low for hint in _PUBLICATION_HINTS))
        and "," not in line
        and " and " not in low
        and "&" not in line
        and not re.search(r"\d", line)
    ):
        return False
    if line.count("@") >= 2:
        return False
    tokens = line.split()
    nameish_tokens = [
        tok
        for tok in tokens
        if re.match(r"^[A-Z][A-Za-z.'-]+$", tok.strip(",;:"))
    ]
    if "," in line or " and " in low or "&" in line:
        return len(nameish_tokens) >= 2 and (
            len(nameish_tokens) / max(len(tokens), 1)
        ) >= 0.25
    upper_tokens = [tok for tok in tokens if tok[:1].isupper()]
    return len(upper_tokens) >= 2 and len(tokens) <= 18


def _looks_like_publication_line(line: str) -> bool:
    normalized = _normalize_space(line)
    low = normalized.lower()
    if not normalized:
        return False
    if _is_generic_front_matter(normalized):
        return True
    if re.search(r"\bdoi\b|http[s]?://|www\.", low):
        return True
    if any(hint in low for hint in _PUBLICATION_HINTS):
        return True
    if re.search(r"\b(received|accepted|published|copyright)\b", low):
        return True
    return False


def _title_candidate_score(lines: list[str], idx: int) -> int:
    line = _normalize_space(lines[idx])
    low = line.lower()
    if not _looks_like_title(line):
        return -10_000

    score = 10
    if idx <= 2:
        score += 4
    elif idx <= 5:
        score += 2
    elif idx <= 8:
        score += 1

    words = line.split()
    if 4 <= len(words) <= 18:
        score += 3
    if len(re.findall(r"\b[A-Z][A-Za-z0-9-]{2,}\b", line)) >= 3:
        score += 2
    if line.count(",") >= 2:
        score -= 6
    if len(re.findall(r"\d", line)) >= 3:
        score -= 6
    if re.search(r"\b(and|&)\b", low):
        score -= 2
    if line.endswith((".", ";", ":")):
        score -= 3
    if re.search(r"\b(article|download|citations?|downloads?)\b", line, flags=re.IGNORECASE):
        score -= 8
    if re.search(r"\b(19\d{2}|20\d{2})\b", line):
        score -= 2
    if any(_looks_like_author_line(next_line) for next_line in lines[idx + 1 : idx + 3]):
        score += 4
    if any(
        _normalize_space(next_line).lower().startswith(("abstract", "摘要"))
        for next_line in lines[idx + 1 : idx + 4]
    ):
        score += 5
    if any(
        token in line.lower()
        for token in ("university", "department", "institute", "school", "college")
    ):
        score -= 4
    return score


def _extract_title(lines: list[str], pdf_path: Path) -> tuple[str, int]:
    """Pick the best title candidate from the first-page text lines.

    We score several nearby lines instead of taking the first non-empty line,
    because publisher headers like "Article in Press" or "PDF Download" often
    appear above the real title.
    """
    candidates: list[tuple[int, int, str]] = []
    for idx, line in enumerate(lines[:18]):
        score = _title_candidate_score(lines, idx)
        if score > 0:
            candidates.append((score, -idx, _normalize_space(line)))
    if candidates:
        best = max(candidates)
        title = best[2]
        for idx, line in enumerate(lines[:18]):
            if _normalize_space(line) == title:
                return title, idx
    fallback = _clean_filename_title(pdf_path)
    return fallback, -1


def _extract_year(lines: list[str], pdf_path: Path, title_index: int) -> str:
    """Estimate publication year from the first page, biased around title."""
    year_scores: dict[str, int] = {}
    start = max(0, title_index - 3) if title_index >= 0 else 0
    end = min(len(lines), max((title_index + 18) if title_index >= 0 else 18, 18))
    for line in lines[start:end]:
        normalized = _normalize_space(line)
        low = normalized.lower()
        if _is_generic_front_matter(normalized):
            continue
        years = re.findall(r"\b(19\d{2}|20\d{2})\b", normalized)
        if not years:
            continue
        score = 1
        if any(hint in low for hint in _PUBLICATION_HINTS):
            score += 3
        if any(pattern in low for pattern in VENUE_PATTERNS):
            score += 2
        if len(normalized) <= 80 and len(years) == 1:
            score += 2
        if "received" in low:
            score -= 1
        if "accepted" in low:
            score -= 1
        if "introduction" in low:
            score -= 3
        for year in years:
            if 1950 <= int(year) <= 2100:
                year_scores[year] = year_scores.get(year, 0) + score
    if year_scores:
        return sorted(year_scores.items(), key=lambda item: (item[1], item[0]), reverse=True)[0][0]
    from_name = re.findall(r"(19\d{2}|20\d{2})", pdf_path.stem)
    if from_name:
        return from_name[0]
    return ""


def _strip_author_affiliation(line: str) -> str:
    text = _normalize_space(line)
    text = re.sub(r"\\[A-Za-z]+", " ", text)
    text = re.sub(r"[{}$^_~|]", " ", text)
    text = re.sub(r"(?<=[A-Za-z])\d+(?:,\d+)*", "", text)
    text = re.sub(r"\b\d+(?:,\d+)*\*?\b", " ", text)
    text = re.sub(r"[*†‡]+", " ", text)
    text = text.replace("\\", " ")
    text = re.sub(r"\b([A-Za-z])\s+([A-Za-z])\b", r"\1\2", text)
    text = re.sub(r"\s*&\s*", " & ", text)
    text = re.sub(r"\s+", " ", text).strip()
    lowered = text.lower()
    cut_positions = []
    had_affiliation_hint = False
    if "@" in text:
        cut_positions.append(text.index("@"))
    for hint in _AFFILIATION_HINTS:
        pos = lowered.find(hint)
        if pos > 0:
            had_affiliation_hint = True
            cut_positions.append(pos)
    if cut_positions:
        text = text[: min(cut_positions)].strip(" ,;:-")
    if had_affiliation_hint and "," in text:
        text = text.split(",", 1)[0].strip()
    if "," in text:
        parts = [part.strip() for part in text.split(",") if part.strip()]
        if parts:
            text = ", ".join(parts[:24])
    return _normalize_space(text)


def _extract_authors(lines: list[str], title_index: int) -> str:
    """Extract author lines near the detected title.

    The heuristics deliberately strip affiliations, email fragments, and
    publisher metadata so catalog fields stay usable for filtering.
    """
    candidate_lines: list[str] = []
    search_ranges: list[tuple[int, int]] = []
    if title_index >= 0:
        search_ranges.append((max(0, title_index - 8), title_index))
        search_ranges.append((title_index + 1, min(len(lines), title_index + 7)))
    else:
        search_ranges.append((1, min(len(lines), 8)))

    for start, end in search_ranges:
        for line in lines[start:end]:
            normalized = _normalize_space(line)
            low = normalized.lower()
            if _is_generic_front_matter(normalized):
                if candidate_lines:
                    break
                continue
            if low.startswith(("abstract", "keywords", "introduction")):
                break
            if re.search(r"\bdoi\b|http[s]?://|www\.", low):
                if candidate_lines:
                    break
                continue
            if not _looks_like_author_line(normalized):
                if candidate_lines and _looks_like_publication_line(normalized):
                    break
                continue
            cleaned = _strip_author_affiliation(normalized)
            if cleaned and len(cleaned.split()) >= 2:
                candidate_lines.append(cleaned)
            elif candidate_lines:
                break
        if candidate_lines:
            break

    deduped: list[str] = []
    seen = set()
    for item in candidate_lines:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return " ; ".join(deduped[:6])


def _extract_keywords(text: str) -> str:
    m = re.search(r"(?:keywords?)\s*[:：]\s*(.+)", text, re.IGNORECASE)
    if not m:
        return ""
    raw = m.group(1).split("\n", 1)[0].strip()
    raw = re.sub(r"\s+", " ", raw)
    return raw[:220]


def _extract_venue(lines: list[str], pdf_path: Path, title_index: int) -> str:
    search_lines = lines[: max(12, title_index + 10 if title_index >= 0 else 14)]
    best: tuple[int, str] | None = None
    for line in search_lines:
        normalized = _normalize_space(line)
        low = normalized.lower()
        if not normalized or _is_generic_front_matter(normalized):
            continue
        if any(hint in low for hint in _AFFILIATION_HINTS) and not any(
            marker in low for marker in _PUBLICATION_HINTS
        ):
            continue
        for pattern in VENUE_PATTERNS:
            if not re.search(rf"\b{re.escape(pattern)}\b", low):
                continue
            score = 1
            if any(hint in low for hint in _PUBLICATION_HINTS):
                score += 3
            if normalized.lower().startswith(pattern):
                score += 2
            if "|" in normalized or ":" in normalized:
                score += 1
            if best is None or score > best[0]:
                best = (score, pattern.upper())
    if best is not None:
        return best[1]
    stem_low = pdf_path.stem.lower()
    for pattern in VENUE_PATTERNS:
        if re.search(rf"\b{re.escape(pattern)}\b", stem_low):
            return pattern.upper()
    return ""


def _guess_language(text: str) -> str:
    if not text:
        return "unknown"
    zh_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    ratio = zh_chars / max(len(text), 1)
    if ratio > 0.15:
        return "zh"
    return "en"


def _extract_paper_metadata(pdf_path: Path, first_page_text: str) -> dict[str, str]:
    normalized = first_page_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_normalize_space(line) for line in normalized.split("\n")]
    lines = [line for line in lines if line]

    title, title_index = _extract_title(lines, pdf_path)
    year = _extract_year(lines, pdf_path, title_index)
    authors = _extract_authors(lines, title_index)
    keywords = _extract_keywords(normalized)
    venue = _extract_venue(lines, pdf_path, title_index)
    language = _guess_language(normalized)

    return {
        "paper_title": title,
        "paper_year": year,
        "paper_authors": authors,
        "paper_venue": venue,
        "paper_keywords": keywords,
        "paper_language": language,
    }


def _mineru_job_dir(pdf_path: Path, output_root: Path) -> Path:
    stat = pdf_path.stat()
    fingerprint = hashlib.sha1(
        f"{pdf_path.resolve()}::{stat.st_size}::{stat.st_mtime_ns}".encode("utf-8")
    ).hexdigest()[:12]
    safe_stem = re.sub(r"[^a-zA-Z0-9._-]+", "_", pdf_path.stem)[:64]
    return output_root / f"{safe_stem}_{fingerprint}"


def _find_mineru_content_list(job_dir: Path, pdf_path: Path) -> Path | None:
    candidates = sorted(job_dir.rglob("*_content_list.json"))
    if not candidates:
        return None

    stem = pdf_path.stem.lower()
    for path in candidates:
        if stem in path.name.lower():
            return path
    return candidates[0]


def _safe_mineru_response_json(response: requests.Response, context: str) -> dict:
    try:
        payload = response.json()
    except ValueError as exc:
        detail = response.text[:400] if response.text else ""
        raise RuntimeError(f"MinerU {context} returned non-JSON response: {detail}") from exc

    if response.status_code >= 400:
        detail = json.dumps(payload, ensure_ascii=False)[:600]
        raise RuntimeError(f"MinerU {context} failed ({response.status_code}): {detail}")

    if isinstance(payload, dict) and payload.get("code", 0) != 0:
        detail = json.dumps(payload, ensure_ascii=False)[:600]
        raise RuntimeError(f"MinerU {context} API error: {detail}")
    return payload if isinstance(payload, dict) else {}


def _submit_mineru_cloud_task(
    pdf_path: Path,
    *,
    api_token: str,
    api_base_url: str,
    model_version: str,
) -> str:
    endpoint = f"{api_base_url.rstrip('/')}/file-urls/batch"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "files": [{"name": pdf_path.name}],
        "model_version": model_version,
    }
    response = requests.post(endpoint, headers=headers, json=payload, timeout=60)
    data = _safe_mineru_response_json(response, "submit task").get("data") or {}
    batch_id = str(data.get("batch_id", "")).strip()
    file_urls = data.get("file_urls") or []
    if not batch_id or not isinstance(file_urls, list) or not file_urls:
        raise RuntimeError("MinerU submit task returned empty batch_id or upload url.")

    upload_url = file_urls[0]
    if not isinstance(upload_url, str) or not upload_url:
        raise RuntimeError("MinerU submit task returned invalid upload url.")

    with pdf_path.open("rb") as f:
        upload_resp = requests.put(upload_url, data=f, timeout=600)
    if upload_resp.status_code >= 400:
        detail = upload_resp.text[:400] if upload_resp.text else ""
        raise RuntimeError(
            f"MinerU upload file failed ({upload_resp.status_code}): {detail}"
        )
    return batch_id


def _pick_extract_result_item(
    extract_result: object,
    *,
    file_name: str,
) -> dict | None:
    if isinstance(extract_result, dict):
        return extract_result
    if not isinstance(extract_result, list):
        return None

    for item in extract_result:
        if isinstance(item, dict) and str(item.get("file_name", "")).strip() == file_name:
            return item

    if len(extract_result) == 1 and isinstance(extract_result[0], dict):
        return extract_result[0]
    return None


def _poll_mineru_cloud_zip_url(
    *,
    api_token: str,
    api_base_url: str,
    batch_id: str,
    file_name: str,
    poll_interval_sec: int,
    timeout_sec: int,
) -> str:
    endpoint = f"{api_base_url.rstrip('/')}/extract-results/batch/{batch_id}"
    headers = {"Authorization": f"Bearer {api_token}"}
    interval = max(int(poll_interval_sec), 1)
    deadline = time.time() + max(int(timeout_sec), 30)

    while True:
        response = requests.get(endpoint, headers=headers, timeout=60)
        payload = _safe_mineru_response_json(response, "poll result")
        data = payload.get("data") or {}
        item = _pick_extract_result_item(
            data.get("extract_result"),
            file_name=file_name,
        )

        if item:
            state = str(item.get("state", "")).strip().lower()
            if state == "done":
                zip_url = str(item.get("full_zip_url", "")).strip()
                if not zip_url:
                    raise RuntimeError("MinerU task finished but full_zip_url is empty.")
                return zip_url
            if state == "failed":
                detail = item.get("err_msg") or item.get("error_msg") or "unknown"
                raise RuntimeError(f"MinerU cloud parse failed: {detail}")

        if time.time() >= deadline:
            raise TimeoutError(
                f"MinerU cloud parsing timed out after {timeout_sec} seconds."
            )
        time.sleep(interval)


def _download_mineru_cloud_content_list(
    *,
    full_zip_url: str,
    job_dir: Path,
    pdf_path: Path,
) -> Path:
    response = requests.get(full_zip_url, timeout=600)
    if response.status_code >= 400:
        detail = response.text[:400] if response.text else ""
        raise RuntimeError(
            f"MinerU download result zip failed ({response.status_code}): {detail}"
        )

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        candidates = [
            name
            for name in archive.namelist()
            if name.lower().endswith("_content_list.json")
        ]
        if not candidates:
            raise RuntimeError("MinerU result zip missing *_content_list.json.")

        stem = pdf_path.stem.lower()
        target_name = next(
            (name for name in candidates if stem in Path(name).name.lower()),
            candidates[0],
        )
        target_bytes = archive.read(target_name)

    output_path = job_dir / Path(target_name).name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(target_bytes)
    return output_path


def _run_mineru_cloud(
    pdf_path: Path,
    *,
    job_dir: Path,
    api_token: str,
    api_base_url: str,
    model_version: str,
    poll_interval_sec: int,
    timeout_sec: int,
) -> Path:
    if not api_token.strip():
        raise RuntimeError(
            "MINERU_API_TOKEN is empty. Set it in .env to use MinerU cloud parser."
        )

    batch_id = _submit_mineru_cloud_task(
        pdf_path,
        api_token=api_token,
        api_base_url=api_base_url,
        model_version=model_version,
    )
    full_zip_url = _poll_mineru_cloud_zip_url(
        api_token=api_token,
        api_base_url=api_base_url,
        batch_id=batch_id,
        file_name=pdf_path.name,
        poll_interval_sec=poll_interval_sec,
        timeout_sec=timeout_sec,
    )
    return _download_mineru_cloud_content_list(
        full_zip_url=full_zip_url,
        job_dir=job_dir,
        pdf_path=pdf_path,
    )


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_bbox(value: Any) -> list[float]:
    if not isinstance(value, list) or len(value) < 4:
        return []
    bbox: list[float] = []
    for item in value[:4]:
        try:
            bbox.append(float(item))
        except (TypeError, ValueError):
            return []
    return bbox


def _to_text_list(value: Any) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        items: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                items.append(item.strip())
        return items
    return []


def _strip_html_tags(text: str) -> str:
    cleaned = re.sub(r"<[^>]+>", " ", text)
    return _normalize_space(cleaned)


def _item_to_text(item: dict) -> str:
    direct_keys = ("text", "content", "latex", "markdown", "md")
    for key in direct_keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    html_value = item.get("html")
    if isinstance(html_value, str) and html_value.strip():
        html_text = _strip_html_tags(html_value)
        if html_text:
            return html_text

    fragments: list[str] = []
    for key in ("image_caption", "image_footnote", "img_caption", "img_footnote"):
        fragments.extend(_to_text_list(item.get(key)))
    for key in ("table_caption", "table_footnote"):
        fragments.extend(_to_text_list(item.get(key)))

    table_body = item.get("table_body")
    if isinstance(table_body, str) and table_body.strip():
        table_text = _strip_html_tags(table_body)
        if table_text:
            fragments.append(table_text[:1200])

    if fragments:
        return "\n".join(fragments).strip()
    return ""


def _build_doc_id(pdf_path: Path) -> str:
    stat = pdf_path.stat()
    raw = f"{pdf_path.resolve()}::{stat.st_size}".encode("utf-8")
    digest = hashlib.sha1(raw).hexdigest()[:16]
    safe_stem = re.sub(r"[^a-zA-Z0-9._-]+", "_", pdf_path.stem)[:48]
    return f"{safe_stem}_{digest}"


def _build_media_id(img_path: Any, prefix: str) -> str:
    if isinstance(img_path, str) and img_path.strip():
        name = Path(img_path).stem
        safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name)
        return f"{prefix}_{safe_name[:24]}"
    return f"{prefix}_unknown"


def _clean_heading_text(text: str) -> str:
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    return _normalize_space(first_line).strip(":- ").strip()


def _infer_heading_level(text: str) -> int | None:
    line = _clean_heading_text(text)
    if not line or len(line) > 140:
        return None
    lower = line.lower()

    if is_reference_heading(line):
        return 1

    m = re.match(r"^(\d+(?:\.\d+)*)\s+\S+", line)
    if m:
        return m.group(1).count(".") + 1

    if re.match(r"^[ivxlcdm]+[\.\)]\s+\S+", lower):
        return 1
    if re.match(r"^(chapter|section|appendix)\b", lower):
        return 1
    if re.match(r"^第[一二三四五六七八九十百0-9]+[章节部分]", line):
        return 1
    if line.isupper() and len(line) <= 80:
        return 1
    return None


def _update_section_stack(
    section_stack: list[str],
    heading_text: str,
    heading_level: int | None,
) -> list[str]:
    if not heading_text:
        return section_stack
    if heading_level is None:
        heading_level = 1
    level = max(1, min(heading_level, 6))
    next_stack = section_stack[: level - 1]
    next_stack.append(heading_text)
    return next_stack


def _build_page_range(pages: list[int]) -> str:
    if not pages:
        return ""
    ordered = sorted(set(pages))
    return f"{ordered[0]}-{ordered[-1]}"


def _parse_mineru_content_list(
    content_list_path: Path,
    *,
    pdf_path: Path,
    paper_meta: dict[str, str],
) -> tuple[list[dict[str, Any]], int]:
    """Parse MinerU `content_list` into normalized block rows.

    This is where layout output becomes RAG-ready metadata: doc_id, page/bbox,
    section stack, reference flags, figure/table ids, and paper metadata.
    """
    payload = json.loads(content_list_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError(f"Invalid MinerU content list format: {content_list_path}")

    doc_id = _build_doc_id(pdf_path)
    section_stack: list[str] = []
    in_reference_section = False
    reference_heading_level: int | None = None
    reference_last_page = 0
    blocks: list[dict[str, Any]] = []
    max_page_idx = -1

    for order, item in enumerate(payload):
        if not isinstance(item, dict):
            continue

        page_idx = _safe_int(item.get("page_idx"), 0)
        page_idx = max(0, page_idx)
        page = page_idx + 1
        max_page_idx = max(max_page_idx, page_idx)

        block_type = str(item.get("type", "text")).strip().lower() or "text"
        text = _item_to_text(item)
        bbox = _normalize_bbox(item.get("bbox"))
        heading_level: int | None = None
        heading_text = ""
        is_reference_heading_block = False

        if in_reference_section and reference_last_page and page > reference_last_page + 1:
            in_reference_section = False
            reference_heading_level = None

        if block_type in {"text", "discarded"} and text:
            heading_level = _infer_heading_level(text)
            if heading_level is not None:
                heading_text = _clean_heading_text(text)
                is_reference_heading_block = is_reference_heading(heading_text)
                if (
                    in_reference_section
                    and not is_reference_heading_block
                    and reference_heading_level is not None
                    and heading_level <= reference_heading_level
                ):
                    in_reference_section = False
                    reference_heading_level = None
                section_stack = _update_section_stack(
                    section_stack,
                    heading_text=normalize_reference_heading(heading_text),
                    heading_level=heading_level,
                )
                if is_reference_heading_block:
                    in_reference_section = True
                    reference_heading_level = heading_level or 1
                    reference_last_page = page

        section_path = " > ".join(section_stack) if section_stack else "正文"
        figure_ids: list[str] = []
        table_ids: list[str] = []
        img_path = item.get("img_path")
        if block_type == "image":
            figure_ids.append(_build_media_id(img_path, "fig"))
        if block_type == "table":
            table_ids.append(_build_media_id(img_path, "tab"))

        is_reference = False
        if is_reference_heading_block:
            is_reference = True
        elif in_reference_section and text:
            if looks_like_reference_entry(text) or looks_like_reference_continuation(text):
                is_reference = True
            elif looks_like_body_paragraph_after_references(text):
                in_reference_section = False
                reference_heading_level = None
        elif is_reference_section(section_path) and looks_like_reference_continuation(text):
            is_reference = True

        if is_reference:
            reference_last_page = page

        blocks.append(
            {
                "block_id": f"{doc_id}::b{order:06d}",
                "doc_id": doc_id,
                "source": pdf_path.name,
                "title": paper_meta.get("paper_title", ""),
                "section_path": section_path,
                "page": page,
                "page_idx": page_idx,
                "page_range": f"{page}-{page}",
                "type": block_type,
                "bbox": bbox,
                "text": text,
                "figure_ids": figure_ids,
                "table_ids": table_ids,
                "year": paper_meta.get("paper_year", ""),
                "authors": paper_meta.get("paper_authors", ""),
                "is_reference": is_reference,
            }
        )

    total_pages = max_page_idx + 1 if max_page_idx >= 0 else 1
    tail_page_start = max(1, int(total_pages * 0.75))
    for block in blocks:
        if block["is_reference"]:
            continue
        if int(block["page"]) >= tail_page_start and looks_like_reference_entry(
            str(block.get("text", ""))
        ):
            block["is_reference"] = True
    return blocks, max(total_pages, 1)


def _build_chunk_input_documents(
    blocks: list[dict[str, Any]],
    *,
    total_pages: int,
    paper_meta: dict[str, str],
    max_chars: int = 1800,
    min_chars: int = 450,
) -> list[Document]:
    """Pack normalized blocks into chunk input docs for later semantic splitting.

    At this stage we keep layout-aware boundaries such as section changes,
    page jumps, and reference/body transitions. Finer chunking happens later.
    """
    docs: list[Document] = []
    current: list[dict[str, Any]] = []
    current_chars = 0
    bind_types = {"equation", "image", "table"}

    def flush() -> None:
        nonlocal current, current_chars
        if not current:
            return
        text_parts = [str(row.get("text", "")).strip() for row in current]
        text_parts = [item for item in text_parts if item]
        if not text_parts:
            current = []
            current_chars = 0
            return

        pages = [int(row.get("page", 1)) for row in current]
        page_range = _build_page_range(pages)
        block_types = sorted({str(row.get("type", "text")) for row in current})
        section_paths = sorted(
            {str(row.get("section_path", "")).strip() for row in current if str(row.get("section_path", "")).strip()}
        )
        figure_ids = sorted(
            {
                item
                for row in current
                for item in row.get("figure_ids", [])
                if isinstance(item, str) and item
            }
        )
        table_ids = sorted(
            {
                item
                for row in current
                for item in row.get("table_ids", [])
                if isinstance(item, str) and item
            }
        )
        metadata = {
            "source": current[0]["source"],
            "page": min(pages),
            "page_end": max(pages),
            "page_range": page_range,
            "total_pages": total_pages,
            "pdf_parser_backend": "mineru_cloud",
            "doc_id": current[0]["doc_id"],
            "title": current[0]["title"],
            "section_path": current[0]["section_path"],
            "section_paths": section_paths,
            "block_types": block_types,
            "figure_ids": figure_ids,
            "table_ids": table_ids,
            "year": current[0]["year"],
            "authors": current[0]["authors"],
            "is_reference": bool(current[0]["is_reference"]),
            "layout_reading_order": "mineru_content_list",
            "reading_block_start": current[0]["block_id"],
            "reading_block_end": current[-1]["block_id"],
            "mineru_block_ids": [row["block_id"] for row in current],
            "mineru_bboxes": [
                {"page": row["page"], "bbox": row["bbox"]}
                for row in current
                if row.get("bbox")
            ],
            "paper_title": paper_meta.get("paper_title", ""),
            "paper_year": paper_meta.get("paper_year", ""),
            "paper_authors": paper_meta.get("paper_authors", ""),
            "paper_venue": paper_meta.get("paper_venue", ""),
            "paper_keywords": paper_meta.get("paper_keywords", ""),
            "paper_language": paper_meta.get("paper_language", ""),
        }
        docs.append(Document(page_content="\n\n".join(text_parts), metadata=metadata))
        current = []
        current_chars = 0

    for block in blocks:
        text = str(block.get("text", "")).strip()
        if not text:
            continue
        block_chars = len(text)
        if not current:
            current = [block]
            current_chars = block_chars
            continue

        prev = current[-1]
        block_type = str(block.get("type", "text")).strip().lower()
        prev_type = str(prev.get("type", "text")).strip().lower()
        same_scope = (
            str(prev.get("doc_id", "")) == str(block.get("doc_id", ""))
            and str(prev.get("section_path", "")) == str(block.get("section_path", ""))
            and bool(prev.get("is_reference")) == bool(block.get("is_reference"))
            and abs(int(prev.get("page", 1)) - int(block.get("page", 1))) <= 1
        )
        will_overflow = current_chars + block_chars > max_chars and current_chars >= min_chars
        hard_overflow = current_chars + block_chars > int(max_chars * 1.6)
        bind_with_context = (
            block_type in bind_types
            or prev_type in bind_types
            or (prev_type in bind_types and block_type in {"text", "discarded"})
        )

        if not same_scope or hard_overflow or (will_overflow and not bind_with_context):
            flush()
        current.append(block)
        current_chars += block_chars
    flush()
    return docs


def _save_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str))
            f.write("\n")


def load_pdf_pages(
    pdf_path: Path,
    *,
    mineru_output_dir: Path | None = None,
    mineru_api_token: str = "",
    mineru_api_base_url: str = "https://mineru.net/api/v4",
    mineru_cloud_model_version: str = "pipeline",
    mineru_cloud_poll_interval_sec: int = 5,
    mineru_cloud_timeout_sec: int = 900,
) -> PdfLoadResult:
    """Parse one PDF into chunk-input documents and structured block rows."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    output_dir = mineru_output_dir or (Path("data") / "mineru_output")
    job_dir = _mineru_job_dir(pdf_path, output_dir)
    content_list_path = _find_mineru_content_list(job_dir, pdf_path)
    if content_list_path is None:
        content_list_path = _run_mineru_cloud(
            pdf_path,
            job_dir=job_dir,
            api_token=mineru_api_token,
            api_base_url=mineru_api_base_url,
            model_version=mineru_cloud_model_version,
            poll_interval_sec=mineru_cloud_poll_interval_sec,
            timeout_sec=mineru_cloud_timeout_sec,
        )

    # Paper metadata is extracted only from page 1 so later pages do not
    # pollute title/authors/year with references or footer text.
    raw_payload = json.loads(content_list_path.read_text(encoding="utf-8"))
    first_page_text_parts: list[str] = []
    if isinstance(raw_payload, list):
        for item in raw_payload:
            if not isinstance(item, dict):
                continue
            if _safe_int(item.get("page_idx"), 0) != 0:
                continue
            text = _item_to_text(item)
            if text:
                first_page_text_parts.append(text)
    first_page_text = "\n".join(first_page_text_parts).strip()
    paper_meta = _extract_paper_metadata(pdf_path, first_page_text)

    blocks, total_pages = _parse_mineru_content_list(
        content_list_path,
        pdf_path=pdf_path,
        paper_meta=paper_meta,
    )
    docs = _build_chunk_input_documents(
        blocks,
        total_pages=total_pages,
        paper_meta=paper_meta,
    )

    structured_rows: list[dict[str, Any]] = []
    for block in blocks:
        row = dict(block)
        row["total_pages"] = total_pages
        row["mineru_cloud_model_version"] = mineru_cloud_model_version
        row["pdf_parser_backend"] = "mineru_cloud"
        structured_rows.append(row)
    _save_jsonl_rows(job_dir / "blocks_structured.jsonl", structured_rows)

    if docs:
        return PdfLoadResult(documents=docs, blocks=structured_rows)
    raise RuntimeError(f"MinerU cloud parsed no usable text from {pdf_path.name}.")


def load_documents_from_dir(
    pdf_dir: Path,
    *,
    mineru_output_dir: Path | None = None,
    mineru_api_token: str = "",
    mineru_api_base_url: str = "https://mineru.net/api/v4",
    mineru_cloud_model_version: str = "pipeline",
    mineru_cloud_poll_interval_sec: int = 5,
    mineru_cloud_timeout_sec: int = 900,
    progress_callback: Callable[[str], None] | None = None,
) -> PdfLoadResult:
    """Parse every PDF in one directory and merge the results."""
    if not pdf_dir.exists():
        return PdfLoadResult(documents=[], blocks=[])

    all_docs: list[Document] = []
    all_blocks: list[dict[str, Any]] = []
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    total = len(pdf_paths)
    for idx, pdf_path in enumerate(pdf_paths, start=1):
        if progress_callback is not None:
            progress_callback(f"[{idx}/{total}] parsing {pdf_path.name}")
        result = load_pdf_pages(
            pdf_path,
            mineru_output_dir=mineru_output_dir,
            mineru_api_token=mineru_api_token,
            mineru_api_base_url=mineru_api_base_url,
            mineru_cloud_model_version=mineru_cloud_model_version,
            mineru_cloud_poll_interval_sec=mineru_cloud_poll_interval_sec,
            mineru_cloud_timeout_sec=mineru_cloud_timeout_sec,
        )
        all_docs.extend(result.documents)
        all_blocks.extend(result.blocks)
        if progress_callback is not None:
            page_count = len(
                {
                    int(row.get("page", 0))
                    for row in result.blocks
                    if str(row.get("doc_id", "")).strip()
                }
            )
            progress_callback(
                f"[{idx}/{total}] parsed {pdf_path.name} -> {page_count} pages, "
                f"{len(result.documents)} raw docs"
            )
    return PdfLoadResult(documents=all_docs, blocks=all_blocks)
