"""content BM25 关键词翻译，当前接入腾讯云和阿里云。"""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import hmac
import json
import re
import uuid
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from ....config import Settings
from ...data.utils import dedupe_text_values_for_search


CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
TENCENT_ACTION = "TextTranslate"
TENCENT_SERVICE = "tmt"
TENCENT_VERSION = "2018-03-21"
ALIYUN_ACTION = "TranslateGeneral"


class KeywordTranslator(Protocol):
    """BM25 关键词翻译器协议，便于测试时注入 mock translator。"""

    def translate(self, text: str, provider: str, settings: Settings) -> str | list[str] | None:
        """把一个关键词短语翻译成英文候选。"""
        ...


class CloudKeywordTranslator:
    """调用腾讯云和阿里云翻译接口，为 BM25 关键词生成英文候选。"""

    def translate(self, text: str, provider: str, settings: Settings) -> str | list[str] | None:
        """按 provider 名称分派到具体云厂商翻译实现。"""
        name = provider.strip().lower()
        if name == "tencent":
            return translate_with_tencent(settings, text)
        if name == "aliyun":
            return translate_with_aliyun(settings, text)
        return None


def contains_chinese(text: str) -> bool:
    """判断关键词是否包含中文字符。"""
    return bool(CHINESE_RE.search(text))


def translate_bm25_terms(
    settings: Settings,
    terms: list[str],
    *,
    translator: KeywordTranslator | None = None,
    warnings: list[str] | None = None,
) -> list[str]:
    """用可插拔翻译器为 BM25 中文关键词补充英文候选。"""
    translated: list[str] = []
    if translator is None:
        return []
    for term in terms:
        if not contains_chinese(term):
            continue
        for provider in configured_translation_providers(settings):
            try:
                # 翻译失败只写 warning，不阻断 dense/BM25 主检索链路。
                translated.extend(normalize_translation_result(translator.translate(term, provider, settings)))
            except Exception as exc:
                if warnings is not None:
                    warnings.append(f"{provider} translation failed for BM25 term {term!r}: {exc}")
    return dedupe_text_values_for_search(translated)


def configured_translation_providers(settings: Settings) -> list[str]:
    """根据 Settings 中已配置密钥筛出可尝试的翻译 provider。"""
    providers: list[str] = []
    for provider in settings.plan_bm25_translate_providers:
        name = provider.strip().lower()
        if name == "tencent" and settings.tencent_translate_secret_id and settings.tencent_translate_secret_key:
            providers.append(name)
        elif name == "aliyun" and settings.aliyun_translate_access_key_id and settings.aliyun_translate_access_key_secret:
            providers.append(name)
    return providers


def translate_with_tencent(settings: Settings, text: str) -> str | None:
    """调用腾讯云 TMT TextTranslate。"""
    secret_id = require_config(settings.tencent_translate_secret_id, "TENCENT_TRANSLATE_SECRET_ID")
    secret_key = require_config(settings.tencent_translate_secret_key, "TENCENT_TRANSLATE_SECRET_KEY")
    host, url = endpoint_host_url(settings.tencent_translate_endpoint)
    timestamp = int(dt.datetime.now(dt.timezone.utc).timestamp())
    payload = json.dumps(
        {
            "SourceText": text,
            "Source": "zh",
            "Target": "en",
            "ProjectId": 0,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    headers = tencent_headers(
        secret_id=secret_id,
        secret_key=secret_key,
        host=host,
        payload=payload,
        timestamp=timestamp,
        region=settings.tencent_translate_region,
    )
    response = request_json(url, headers, payload.encode("utf-8"), settings.plan_bm25_translate_timeout_seconds)
    body = response.get("Response") or response
    if body.get("Error"):
        error = body["Error"]
        raise RuntimeError(f"{error.get('Code')}: {error.get('Message')}")
    translated = body.get("TargetText")
    return str(translated).strip() if translated else None


def tencent_headers(
    *,
    secret_id: str,
    secret_key: str,
    host: str,
    payload: str,
    timestamp: int,
    region: str,
) -> dict[str, str]:
    """生成腾讯云 API 3.0 TC3-HMAC-SHA256 请求头。"""
    method = "POST"
    content_type = "application/json; charset=utf-8"
    canonical_headers = f"content-type:{content_type}\nhost:{host}\n"
    signed_headers = "content-type;host"
    hashed_payload = sha256_hex(payload.encode("utf-8"))
    canonical_request = "\n".join([
        method,
        "/",
        "",
        canonical_headers,
        signed_headers,
        hashed_payload,
    ])
    date = dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc).strftime("%Y-%m-%d")
    credential_scope = f"{date}/{TENCENT_SERVICE}/tc3_request"
    string_to_sign = "\n".join([
        "TC3-HMAC-SHA256",
        str(timestamp),
        credential_scope,
        sha256_hex(canonical_request.encode("utf-8")),
    ])
    # 腾讯 TC3 签名按 date -> service -> tc3_request 逐级派生密钥。
    secret_date = hmac_sha256(("TC3" + secret_key).encode("utf-8"), date)
    secret_service = hmac_sha256(secret_date, TENCENT_SERVICE)
    secret_signing = hmac_sha256(secret_service, "tc3_request")
    signature = hmac.new(secret_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
    authorization = (
        "TC3-HMAC-SHA256 "
        f"Credential={secret_id}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, "
        f"Signature={signature}"
    )
    return {
        "Authorization": authorization,
        "Content-Type": content_type,
        "Host": host,
        "X-TC-Action": TENCENT_ACTION,
        "X-TC-Timestamp": str(timestamp),
        "X-TC-Version": TENCENT_VERSION,
        "X-TC-Region": region,
    }


def translate_with_aliyun(settings: Settings, text: str) -> str | None:
    """调用阿里云机器翻译 TranslateGeneral。"""
    access_key_id = require_config(settings.aliyun_translate_access_key_id, "ALIYUN_TRANSLATE_ACCESS_KEY_ID")
    access_key_secret = require_config(settings.aliyun_translate_access_key_secret, "ALIYUN_TRANSLATE_ACCESS_KEY_SECRET")
    host, _ = endpoint_host_url(settings.aliyun_translate_endpoint)
    params = {
        "Action": ALIYUN_ACTION,
        "Format": "JSON",
        "Version": settings.aliyun_translate_version,
        "AccessKeyId": access_key_id,
        "SignatureMethod": "HMAC-SHA1",
        "SignatureVersion": "1.0",
        "SignatureNonce": str(uuid.uuid4()),
        "Timestamp": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "FormatType": "text",
        "SourceLanguage": "zh",
        "TargetLanguage": "en",
        "SourceText": text,
        "Scene": "general",
    }
    params["Signature"] = aliyun_rpc_signature("GET", params, access_key_secret)
    url = f"https://{host}/?{canonical_query(params)}"
    response = request_json_url(url, settings.plan_bm25_translate_timeout_seconds)
    body = response.get("TranslateGeneralResponse") or response
    code = str(body.get("Code") or body.get("code") or "")
    if code and code != "200":
        raise RuntimeError(f"{code}: {body.get('Message') or body.get('message')}")
    data = body.get("Data") or body.get("data") or {}
    translated = data.get("Translated") or data.get("translated")
    return str(translated).strip() if translated else None


def request_json(url: str, headers: dict[str, str], payload: bytes, timeout_seconds: int) -> dict[str, Any]:
    """发送 HTTP 请求并解析 JSON 响应。"""
    request = Request(url, data=payload, headers=headers, method="POST")
    return request_json_request(request, timeout_seconds)


def request_json_url(url: str, timeout_seconds: int) -> dict[str, Any]:
    """发送 GET 请求并解析 JSON 响应。"""
    request = Request(url, method="GET")
    return request_json_request(request, timeout_seconds)


def request_json_request(request: Request, timeout_seconds: int) -> dict[str, Any]:
    """执行 HTTP 请求并解析 JSON 响应。"""
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(str(exc.reason)) from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"non-json translation response: {raw[:200]}") from exc


def normalize_translation_result(value: Any) -> list[str]:
    """把 provider 返回值规整成非空字符串列表。"""
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def endpoint_host_url(endpoint: str) -> tuple[str, str]:
    """把 endpoint 配置规整成 host 和 https URL。"""
    value = endpoint.strip().removeprefix("https://").removeprefix("http://").strip("/")
    if not value:
        raise RuntimeError("translation endpoint is empty")
    return value, f"https://{value}/"


def require_config(value: str | None, name: str) -> str:
    """读取必填配置，缺失时抛出清晰错误。"""
    if not value:
        raise RuntimeError(f"{name} is not configured")
    return value


def hmac_sha256(key: bytes, message: str) -> bytes:
    """HMAC-SHA256 原始字节结果。"""
    return hmac.new(key, message.encode("utf-8"), hashlib.sha256).digest()


def sha256_hex(value: bytes) -> str:
    """SHA256 十六进制摘要。"""
    return hashlib.sha256(value).hexdigest()


def aliyun_rpc_signature(method: str, params: dict[str, str], access_key_secret: str) -> str:
    """生成阿里云 RPC 风格公共参数签名。"""
    query = canonical_query(params)
    # 阿里云 RPC 签名要求对 canonical query 再做一次百分号编码。
    string_to_sign = f"{method}&%2F&{aliyun_percent_encode(query)}"
    digest = hmac.new(
        (access_key_secret + "&").encode("utf-8"),
        string_to_sign.encode("utf-8"),
        hashlib.sha1,
    ).digest()
    return base64.b64encode(digest).decode("utf-8")


def canonical_query(params: dict[str, str]) -> str:
    """按阿里云 RPC 签名要求排序并编码 query 参数。"""
    return "&".join(
        f"{aliyun_percent_encode(key)}={aliyun_percent_encode(value)}"
        for key, value in sorted(params.items())
    )


def aliyun_percent_encode(value: str) -> str:
    """阿里云 RPC 签名使用的 RFC3986 百分号编码。"""
    return quote(str(value), safe="~")
