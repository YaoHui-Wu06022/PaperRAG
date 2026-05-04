"""venue 名称的存储规范化、展示和别名匹配。"""

from __future__ import annotations

import json
import re
from typing import Any

from ..config import Settings


def load_venue_aliases(settings: Settings) -> list[dict[str, Any]]:
    """读取 venue_aliases.json；缺失时保持零配置可运行。"""
    path = settings.data_dir / "venue_aliases.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else []


def normalize_venue_for_storage(settings: Settings, venue: Any) -> str | None:
    """把外部来源的 venue 映射成项目内统一展示名。"""
    text = clean_venue_text(venue)
    if not text:
        return None
    return display_venue(settings, text)


def display_venue(settings: Settings, venue: Any) -> str:
    """返回规范展示名，未命中别名表时保留清洗后的原值。"""
    text = clean_venue_text(venue)
    if not text:
        return ""
    text_key = venue_key(text)
    for entry in load_venue_aliases(settings):
        display = venue_entry_display(entry)
        for candidate in venue_entry_terms(entry):
            if venue_keys_match(text_key, venue_key(candidate)):
                return display
    return text


def expand_venue_query_terms(settings: Settings, values: list[str]) -> list[str]:
    """检索时把用户给出的 venue 展开成 canonical/display/aliases 候选。"""
    expanded: list[str] = []
    for value in values:
        value_key = venue_key(value)
        matched = False
        for entry in load_venue_aliases(settings):
            term_keys = [venue_key(term) for term in venue_entry_terms(entry)]
            if value_key and value_key in term_keys:
                expanded.extend(venue_entry_terms(entry))
                matched = True
                break
        if not matched:
            expanded.append(clean_venue_text(value))
    return unique_terms(expanded)


def expand_venue_record_terms(settings: Settings, venue: Any) -> list[str]:
    """把记录中的单个 venue 展开，供 filter 与用户 query 做宽松匹配。"""
    text = clean_venue_text(venue)
    if not text:
        return []
    text_key = venue_key(text)
    expanded = [text]
    for entry in load_venue_aliases(settings):
        for candidate in venue_entry_terms(entry):
            if venue_keys_match(text_key, venue_key(candidate)):
                expanded.extend(venue_entry_terms(entry))
                expanded.append(venue_entry_display(entry))
                return unique_terms(expanded)
    return unique_terms(expanded)


def venue_entry_display(entry: dict[str, Any]) -> str:
    canonical = str(entry.get("canonical") or "").strip()
    display = str(entry.get("display") or "").strip()
    return display or canonical


def venue_entry_terms(entry: dict[str, Any]) -> list[str]:
    """一条 venue 规则的所有可匹配名字：canonical、display 和 aliases。"""
    canonical = str(entry.get("canonical") or "").strip()
    display = str(entry.get("display") or "").strip()
    aliases = [str(alias).strip() for alias in entry.get("aliases") or [] if str(alias).strip()]
    return [term for term in [canonical, display, *aliases] if term]


def unique_terms(values: list[str]) -> list[str]:
    """按 venue_key 去重，保留第一次出现的展示文本。"""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = clean_venue_text(value)
        key = venue_key(text)
        if key and key not in seen:
            seen.add(key)
            result.append(text)
    return result


def venue_key(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", clean_venue_text(value).lower()))


def venue_keys_match(left: str, right: str) -> bool:
    """允许简称和全称互相包含，例如 CVPR 与 Conference on CVPR。"""
    if not left or not right:
        return False
    return left == right or left in right or right in left


def clean_venue_text(value: Any) -> str:
    """去掉 venue 中的年份，使 CVPR 2016 和 CVPR 可以共用别名匹配。"""
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"\b(?:19|20)\d{2}\b", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ,.;:-")
