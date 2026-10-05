"""关键词检索的中译英查询预处理和翻译服务适配。"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Protocol

from paper_rag.config import Settings
from paper_rag.lexical import build_fts_query, normalize_lexical_text
from paper_rag.llamaindex.query_rewriter import QueryRewrite, QueryRewriterClient, QueryRewriterError


class TranslationError(RuntimeError):
    """翻译服务不可用或返回空结果。"""


class Translator(Protocol):
    provider: str

    def translate(self, text: str) -> str:
        """把中文查询翻译成英文。"""


class QueryRewriter(Protocol):
    def rewrite(self, query: str) -> QueryRewrite:
        """提取 BM25 核心词和完整短语。"""


@dataclass(frozen=True)
class LexicalQuery:
    """词法检索实际使用的查询及其可观测状态。"""

    original_query: str
    query: str
    fts_query: str
    translation_used: bool
    translation_provider: str | None
    translation_fallback: bool
    stopwords_removed: tuple[str, ...]
    rewriter_used: bool
    rewriter_fallback: bool
    core_terms: tuple[str, ...]
    warnings: tuple[str, ...]

    def debug(self) -> dict[str, object]:
        """转换为不会泄露原始问题的调试字段。"""

        return {
            "lexical_query": self.fts_query,
            "translation_used": self.translation_used,
            "translation_provider": self.translation_provider,
            "translation_fallback": self.translation_fallback,
            "stopwords_removed": list(self.stopwords_removed),
            "rewriter_used": self.rewriter_used,
            "rewriter_fallback": self.rewriter_fallback,
            "core_terms": list(self.core_terms),
        }


def prepare_lexical_query(
    query: str,
    settings: Settings,
    *,
    rewriter: QueryRewriter | None = None,
) -> LexicalQuery:
    """先提取完整核心检索短语，再翻译为 FTS5 查询；语义检索仍使用原始问题。"""

    original = str(query or "").strip()
    if not original:
        return _plain_result(original, "", False, None, False, (), (), False, False)
    rewrite: QueryRewrite | None = None
    rewriter_fallback = False
    warnings: list[str] = []
    if settings.query_rewriter_enabled and settings.query_rewriter_api_key:
        try:
            rewrite = (rewriter or QueryRewriterClient(settings)).rewrite(original)
        except QueryRewriterError as exc:
            rewriter_fallback = True
            warnings.append(_format_failure("query_rewriter", exc))
    if rewrite is not None:
        try:
            return _prepare_rewritten_query(original, settings, rewrite, rewriter_fallback, warnings)
        except (TranslationError, ValueError) as exc:
            warnings.append(_format_failure("tencent", exc))
            rewriter_fallback = True
    if not settings.bm25_translation_enabled or not _contains_chinese(original):
        lexical, removed = normalize_lexical_text(original)
        return _plain_result(original, lexical, False, None, False, tuple(removed), tuple(warnings), False, rewriter_fallback)
    if len(original) > settings.bm25_translation_max_chars:
        lexical, removed = normalize_lexical_text(original)
        return LexicalQuery(
            original_query=original,
            query=lexical,
            fts_query=build_fts_query(lexical),
            translation_used=False,
            translation_provider=None,
            translation_fallback=False,
            stopwords_removed=tuple(removed),
            rewriter_used=False,
            rewriter_fallback=rewriter_fallback,
            core_terms=(),
            warnings=tuple([*warnings, "translation_skipped:query_too_long"]),
        )

    failures: list[str] = []
    provider_name = "tencent"
    provider = _make_translator(settings)
    for _ in range(settings.bm25_translation_retry_count + 1):
        try:
            translated = provider.translate(original).strip()
            if not translated:
                raise TranslationError("empty translation")
            technical = _technical_tokens(original)
            lexical, removed = normalize_lexical_text(f"{translated} {' '.join(technical)}")
            return LexicalQuery(
                original_query=original,
                query=lexical,
                fts_query=build_fts_query(lexical),
                translation_used=True,
                translation_provider=provider_name,
                translation_fallback=False,
                stopwords_removed=tuple(removed),
                rewriter_used=False,
                rewriter_fallback=rewriter_fallback,
                core_terms=(),
                warnings=tuple([*warnings, *failures]),
            )
        except Exception as exc:
            failures.append(_format_failure(provider_name, exc))

    lexical, removed = normalize_lexical_text(original)
    return LexicalQuery(
        original_query=original,
        query=lexical,
        fts_query=build_fts_query(lexical),
        translation_used=False,
        translation_provider=None,
        translation_fallback=True,
        stopwords_removed=tuple(removed),
        rewriter_used=False,
        rewriter_fallback=rewriter_fallback,
        core_terms=(),
        warnings=tuple([*warnings, *failures]),
    )


def _prepare_rewritten_query(
    original: str,
    settings: Settings,
    rewrite: QueryRewrite,
    rewriter_fallback: bool,
    warnings: list[str],
) -> LexicalQuery:
    """翻译 Query Rewriter 的完整核心短语，不拆分成独立词。"""

    provider_name: str | None = None
    translation_used = False
    translated_terms = list(rewrite.core_terms)
    if settings.bm25_translation_enabled:
        provider = _make_translator(settings)
        provider_name = provider.provider
        translated_terms, translation_used = _translate_parts(provider, translated_terms)
    core_terms, removed_terms = _normalize_parts(translated_terms)
    if not core_terms:
        raise ValueError("Query Rewriter 结果没有可用检索词")
    query_text = " ".join(core_terms)
    fts_query = build_fts_query(
        "",
        remove_stopwords=False,
        phrases=core_terms,
    )
    return LexicalQuery(
        original_query=original,
        query=query_text,
        fts_query=fts_query,
        translation_used=translation_used,
        translation_provider=provider_name if translation_used else None,
        translation_fallback=False,
        stopwords_removed=tuple(removed_terms),
        rewriter_used=True,
        rewriter_fallback=rewriter_fallback,
        core_terms=tuple(core_terms),
        warnings=tuple(warnings),
    )


def _translate_parts(provider: Translator, parts: list[str]) -> tuple[list[str], bool]:
    translated: list[str] = []
    used = False
    for part in parts:
        if _contains_chinese(part):
            value = provider.translate(part).strip()
            if not value:
                raise TranslationError("empty translation")
            translated.append(value)
            used = True
        else:
            translated.append(part)
    return translated, used


def _normalize_parts(parts: list[str]) -> tuple[list[str], list[str]]:
    normalized: list[str] = []
    removed: list[str] = []
    for part in parts:
        value, dropped = normalize_lexical_text(part)
        if value:
            normalized.append(value)
        removed.extend(dropped)
    return list(dict.fromkeys(normalized)), list(dict.fromkeys(removed))


def _plain_result(
    original: str,
    lexical: str,
    translation_used: bool,
    provider: str | None,
    translation_fallback: bool,
    removed: tuple[str, ...],
    warnings: tuple[str, ...],
    rewriter_used: bool,
    rewriter_fallback: bool,
) -> LexicalQuery:
    return LexicalQuery(
        original_query=original,
        query=lexical,
        fts_query=build_fts_query(lexical),
        translation_used=translation_used,
        translation_provider=provider,
        translation_fallback=translation_fallback,
        stopwords_removed=removed,
        rewriter_used=rewriter_used,
        rewriter_fallback=rewriter_fallback,
        core_terms=(),
        warnings=warnings,
    )


def _make_translator(settings: Settings) -> Translator:
    """构造唯一的腾讯云翻译适配器，保留小函数便于测试替换。"""

    return TencentTranslator(settings)


class TencentTranslator:
    """腾讯云机器翻译适配器，依赖在调用时惰性导入。"""

    provider = "tencent"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def translate(self, text: str) -> str:
        try:
            from tencentcloud.common import credential
            from tencentcloud.common.profile.client_profile import ClientProfile
            from tencentcloud.common.profile.http_profile import HttpProfile
            from tencentcloud.tmt.v20180321 import models, tmt_client
        except ImportError as exc:
            raise TranslationError("Tencent translation SDK is not installed") from exc
        cred = credential.Credential(
            self.settings.tencent_translate_secret_id,
            self.settings.tencent_translate_secret_key,
        )
        http_profile = HttpProfile()
        http_profile.endpoint = self.settings.tencent_translate_endpoint
        http_profile.reqTimeout = self.settings.bm25_translation_timeout_seconds * 1000
        client_profile = ClientProfile(httpProfile=http_profile)
        client = tmt_client.TmtClient(cred, self.settings.tencent_translate_region, client_profile)
        request = models.TextTranslateRequest()
        request.SourceText = text
        request.Source = "zh"
        request.Target = "en"
        request.ProjectId = 0
        response = client.TextTranslate(request)
        translated = str(getattr(response, "TargetText", "") or "").strip()
        if not translated:
            raise TranslationError("Tencent translation returned empty text")
        return translated


def _format_failure(provider: str, exc: Exception) -> str:
    """把翻译失败压缩成可诊断但不携带密钥的 warning。"""

    pattern = r"(?i)\b(access[_-]?key(?:[_-]?(?:id|secret))?|secret[_-]?(?:id|key)|token|password|api[-_]?key)[=: ]+\S+"
    message = re.sub(pattern, lambda match: f"{match.group(1)}=<redacted>", str(exc))
    message = " ".join(message.split())[:240]
    suffix = f":{message}" if message else ""
    return f"translation_failed:{provider}:{type(exc).__name__}{suffix}"


def _contains_chinese(value: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", value))


def _technical_tokens(value: str) -> list[str]:
    """保留缩写、数字模型名和内部大写技术名。"""

    result: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"[A-Za-z][A-Za-z0-9._-]*", value):
        if any(character.isdigit() for character in token) or token.isupper() or any(character.isupper() for character in token[1:]):
            key = token.casefold()
            if key not in seen:
                result.append(token)
                seen.add(key)
    return result


__all__ = [
    "LexicalQuery",
    "TencentTranslator",
    "TranslationError",
    "prepare_lexical_query",
]
