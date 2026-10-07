"""关键词检索的中译英查询预处理和翻译服务适配。"""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Any, Iterable, Protocol

from paper_rag.config import Settings
from paper_rag.lexical import build_fts_query, normalize_lexical_text
from paper_rag.llamaindex.query_rewriter import QueryRewrite, QueryRewriterClient, QueryRewriterError


class TranslationError(RuntimeError):
    """翻译服务不可用或返回空结果。"""

    def __init__(self, message: str, *, warnings: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.warnings = warnings


class Translator(Protocol):
    provider: str

    def translate(self, text: str) -> str:
        """把中文查询翻译成英文。"""


class QueryRewriter(Protocol):
    def rewrite(self, query: str, *, purpose: str, task: str | None, filters: dict[str, Any]) -> QueryRewrite:
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
    rewriter_error: str | None = None
    entities: tuple[str, ...] = ()
    original_core_terms: tuple[str, ...] = ()
    translated_entities: tuple[str, ...] = ()
    purpose: str = "metadata"
    task: str | None = None

    def with_terms(self, terms: Iterable[str]) -> LexicalQuery:
        """复用已完成的改写和翻译，仅改变当前阶段的词法查询。"""
        values = tuple(dict.fromkeys(terms))
        return replace(self, query=" ".join(values), fts_query=build_fts_query("", phrases=values))

    def debug(self) -> dict[str, object]:
        """转换为不会泄露原始问题的调试字段。"""

        return {
            "purpose": self.purpose,
            "task": self.task,
            "lexical_query": self.fts_query,
            "translation_used": self.translation_used,
            "translation_provider": self.translation_provider,
            "translation_fallback": self.translation_fallback,
            "stopwords_removed": list(self.stopwords_removed),
            "rewriter_used": self.rewriter_used,
            "rewriter_fallback": self.rewriter_fallback,
            "entities": list(self.entities),
            "core_terms": list(self.original_core_terms),
            "translated_entities": list(self.translated_entities),
            "translated_core_terms": list(self.core_terms),
            "rewriter_error": self.rewriter_error,
        }


def prepare_lexical_query(
    query: str,
    settings: Settings,
    *,
    rewriter: QueryRewriter | None = None,
    purpose: str = "metadata",
    task: str | None = None,
    filters: dict[str, Any] | None = None,
) -> LexicalQuery:
    """用途和任务只由服务入口传入，不让改写器再次判断。"""
    result = _prepare_lexical_query(query, settings, rewriter=rewriter, purpose=purpose, task=task, filters=filters or {})
    return replace(result, purpose=purpose, task=task)


def _prepare_lexical_query(
    query: str, settings: Settings, *, rewriter: QueryRewriter | None,
    purpose: str, task: str | None, filters: dict[str, Any],
) -> LexicalQuery:
    """先提取完整核心检索短语，再翻译为 FTS5 查询；语义检索仍使用原始问题。"""

    original = str(query or "").strip()
    if not original:
        return _plain_result(original, "", False, None, False, (), (), False, False)
    rewrite: QueryRewrite | None = None
    rewriter_fallback = False
    rewriter_error: str | None = None
    warnings: list[str] = []
    if settings.query_rewriter_enabled and settings.query_rewriter_api_key:
        try:
            rewrite = (rewriter or QueryRewriterClient(settings)).rewrite(original, purpose=purpose, task=task, filters=filters)
        except QueryRewriterError as exc:
            rewriter_fallback = True
            rewriter_error = _rewriter_error_code(exc)
            warnings.append(_format_rewriter_failure(exc))
    if rewrite is not None:
        try:
            return _prepare_rewritten_query(original, settings, rewrite, rewriter_fallback, warnings)
        except (TranslationError, ValueError) as exc:
            if isinstance(exc, TranslationError):
                warnings.extend(exc.warnings)
            else:
                warnings.append(_format_rewriter_failure(exc))
            rewriter_fallback = True
    if not settings.bm25_translation_enabled or not _contains_chinese(original):
        lexical, removed = normalize_lexical_text(original)
        return _plain_result(original, lexical, False, None, False, tuple(removed), tuple(warnings), False, rewriter_fallback, rewriter_error)
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
            rewriter_error=rewriter_error,
        )

    try:
        translated, provider_name, provider_fallback, failures = _translate_with_fallback(settings, original)
        technical = _technical_tokens(original)
        lexical, removed = normalize_lexical_text(f"{translated} {' '.join(technical)}")
        return LexicalQuery(
            original_query=original,
            query=lexical,
            fts_query=build_fts_query(lexical),
            translation_used=True,
            translation_provider=provider_name,
            translation_fallback=provider_fallback,
            stopwords_removed=tuple(removed),
            rewriter_used=False,
            rewriter_fallback=rewriter_fallback,
            core_terms=(),
            warnings=tuple([*warnings, *failures]),
            rewriter_error=rewriter_error,
        )
    except TranslationError as exc:
        failures = list(exc.warnings)

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
        rewriter_error=rewriter_error,
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
    translation_fallback = False
    translated_terms = list(rewrite.entities + rewrite.core_terms)
    if settings.bm25_translation_enabled:
        try:
            translated_terms, provider_name, translation_fallback, failures = _translate_parts_with_fallback(
                settings, translated_terms
            )
            warnings.extend(failures)
        except TranslationError as exc:
            # 改写已成功，翻译失败不应伪装成改写失败或丢失原始对象。
            warnings.extend(exc.warnings)
            translation_fallback = True
        translation_used = provider_name is not None
    entities, entity_removed = _normalize_parts(translated_terms[:len(rewrite.entities)])
    core_terms, removed_terms = _normalize_parts(translated_terms[len(rewrite.entities):])
    terms = list(dict.fromkeys(entities + core_terms))
    if not terms:
        raise ValueError("Query Rewriter 结果没有可用检索词")
    query_text = " ".join(terms)
    fts_query = build_fts_query(
        "",
        remove_stopwords=False,
        phrases=terms,
    )
    return LexicalQuery(
        original_query=original,
        query=query_text,
        fts_query=fts_query,
        translation_used=translation_used,
        translation_provider=provider_name if translation_used else None,
        translation_fallback=translation_fallback,
        stopwords_removed=tuple(dict.fromkeys(entity_removed + removed_terms)),
        rewriter_used=True,
        rewriter_fallback=rewriter_fallback,
        core_terms=tuple(core_terms),
        entities=rewrite.entities,
        original_core_terms=rewrite.core_terms,
        translated_entities=tuple(translated_terms[:len(rewrite.entities)]),
        warnings=tuple(warnings),
    )


def _translate_parts(provider: Translator, parts: list[str]) -> tuple[list[str], bool]:
    translated: list[str] = []
    used = False
    for part in parts:
        if _contains_chinese(part):
            try:
                value = provider.translate(part).strip()
            except Exception as exc:
                # 将第三方翻译 SDK 的异常统一转换为可回退的领域错误。
                raise TranslationError(str(exc)) from exc
            if not value:
                raise TranslationError("empty translation")
            translated.append(value)
            used = True
        else:
            translated.append(part)
    return translated, used


def _translation_chain(settings: Settings) -> list[Translator]:
    """按优先级构造翻译服务；腾讯失败后才尝试阿里云。"""

    providers: list[Translator] = [_make_translator(settings)]
    if (
        settings.aliyun_translation_enabled
        and settings.aliyun_translation_access_key_id
        and settings.aliyun_translation_access_key_secret
    ):
        providers.append(AliyunTranslator(settings))
    return providers


def _translate_with_fallback(
    settings: Settings,
    text: str,
) -> tuple[str, str, bool, list[str]]:
    """依次调用腾讯云和阿里云，返回译文、提供商、是否发生备用切换及警告。"""

    failures: list[str] = []
    for provider_index, provider in enumerate(_translation_chain(settings)):
        for _ in range(settings.bm25_translation_retry_count + 1):
            try:
                translated = provider.translate(text).strip()
                if not translated:
                    raise TranslationError("empty translation")
                if provider_index:
                    failures.append(f"translation_fallback:{provider.provider}")
                return translated, provider.provider, provider_index > 0, failures
            except Exception as exc:
                failures.append(_format_failure(provider.provider, exc))
    raise TranslationError("all translation providers failed", warnings=tuple(failures))


def _translate_parts_with_fallback(
    settings: Settings,
    parts: list[str],
) -> tuple[list[str], str | None, bool, list[str]]:
    """使用同一个翻译服务完成全部核心短语，避免一次查询混用语言模型。"""

    if not any(_contains_chinese(part) for part in parts):
        return list(parts), None, False, []
    failures: list[str] = []
    for provider_index, provider in enumerate(_translation_chain(settings)):
        for _ in range(settings.bm25_translation_retry_count + 1):
            try:
                translated, used = _translate_parts(provider, parts)
                if provider_index:
                    failures.append(f"translation_fallback:{provider.provider}")
                return translated, provider.provider if used else None, provider_index > 0, failures
            except Exception as exc:
                failures.append(_format_failure(provider.provider, exc))
    raise TranslationError("all translation providers failed", warnings=tuple(failures))


def _normalize_parts(parts: list[str]) -> tuple[list[str], list[str]]:
    normalized: list[str] = []
    removed: list[str] = []
    for part in parts:
        value, dropped = normalize_lexical_text(part)
        # 普通单字词可过滤，但表/图编号不可丢，避免 Table 2 退化成 Table。
        if re.search(r"\b(?:table|tab|figure|fig)\.?\s*\d+\b|(?:表|图)\s*\d+", part, re.IGNORECASE):
            value = re.sub(r"\s+", " ", part.casefold()).strip()
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
    rewriter_error: str | None = None,
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
        rewriter_error=rewriter_error,
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


class AliyunTranslator:
    """阿里云机器翻译适配器，作为腾讯云失败后的备用服务。"""

    provider = "aliyun"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def translate(self, text: str) -> str:
        try:
            from alibabacloud_alimt20181012.client import Client
            from alibabacloud_alimt20181012 import models
            from alibabacloud_tea_openapi import models as open_api_models
        except ImportError as exc:
            raise TranslationError("Aliyun translation SDK is not installed") from exc

        config = open_api_models.Config(
            access_key_id=self.settings.aliyun_translation_access_key_id,
            access_key_secret=self.settings.aliyun_translation_access_key_secret,
            security_token=self.settings.aliyun_translation_security_token or None,
            region_id=self.settings.aliyun_translation_region_id,
            endpoint=self.settings.aliyun_translation_endpoint,
            read_timeout=self.settings.bm25_translation_timeout_seconds * 1000,
            connect_timeout=self.settings.bm25_translation_timeout_seconds * 1000,
        )
        client = Client(config)
        request = models.TranslateGeneralRequest(
            format_type="text",
            scene="general",
            source_language="zh",
            source_text=text,
            target_language="en",
        )
        response = client.translate_general(request)
        body = getattr(response, "body", None)
        code = str(getattr(body, "code", "") or "")
        if code and code != "200":
            message = str(getattr(body, "message", "") or "request failed")
            raise TranslationError(f"Aliyun translation failed: {message}")
        data = getattr(body, "data", None)
        translated = str(getattr(data, "translated", "") or "").strip()
        if not translated:
            raise TranslationError("Aliyun translation returned empty text")
        return translated


def _format_failure(provider: str, exc: Exception) -> str:
    """把翻译失败压缩成可诊断但不携带密钥的 warning。"""

    pattern = r"(?i)\b(access[_-]?key(?:[_-]?(?:id|secret))?|secret[_-]?(?:id|key)|token|password|api[-_]?key)[=: ]+\S+"
    message = re.sub(pattern, lambda match: f"{match.group(1)}=<redacted>", str(exc))
    message = " ".join(message.split())[:240]
    suffix = f":{message}" if message else ""
    return f"translation_failed:{provider}:{type(exc).__name__}{suffix}"


def _format_rewriter_failure(exc: Exception) -> str:
    """区分 Query Rewriter 失败和腾讯云翻译失败。"""

    pattern = r"(?i)\b(access[_-]?key(?:[_-]?(?:id|secret))?|secret[_-]?(?:id|key)|token|password|api[-_]?key)[=: ]+\S+"
    message = re.sub(pattern, lambda match: f"{match.group(1)}=<redacted>", str(exc))
    message = " ".join(message.split())[:240]
    suffix = f":{message}" if message else ""
    return f"query_rewriter_failed:{type(exc).__name__}{suffix}"


def _rewriter_error_code(exc: Exception) -> str:
    """把严格契约错误压缩为 Agent 可判断的稳定代码。"""

    message = str(exc).casefold()
    if "entities" in message and "字符串数组" in str(exc):
        return "invalid_entities"
    if "必须仅返回" in str(exc):
        return "invalid_fields"
    if "core_terms" in message and "字符串数组" in str(exc):
        return "invalid_core_terms"
    if "没有返回核心" in str(exc) or "没有可用检索词" in str(exc):
        return "empty_core_terms"
    if "json" in message or "choices" in message or "message.content" in message:
        return "invalid_response"
    return "request_failed"


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
    "AliyunTranslator",
    "LexicalQuery",
    "TencentTranslator",
    "TranslationError",
    "prepare_lexical_query",
]
