"""项目内部的标准库 HTTP JSON 请求工具。"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Callable


RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class HttpRequestError(RuntimeError):
    """HTTP 请求失败，保留状态码和响应正文供上层转换错误。"""

    def __init__(self, message: str, *, status: int | None = None, body: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.body = body


class JsonHttpClient:
    """统一 JSON 请求、超时、重试和响应解析。"""

    def __init__(
        self,
        *,
        opener: Callable[..., Any] | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.opener = opener or urllib.request.urlopen
        self.sleeper = sleeper

    def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        *,
        headers: dict[str, str],
        timeout: int,
        retries: int,
        error_prefix: str,
    ) -> Any:
        return self.request_json(
            "POST",
            url,
            payload,
            headers=headers,
            timeout=timeout,
            retries=retries,
            error_prefix=error_prefix,
        )

    def request_json(
        self,
        method: str,
        url: str,
        payload: dict[str, Any] | None,
        *,
        headers: dict[str, str],
        timeout: int,
        retries: int,
        error_prefix: str,
    ) -> Any:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None,
            method=method,
            headers=headers,
        )
        last_error: Exception | None = None
        for attempt in range(max(0, int(retries)) + 1):
            try:
                with self.opener(request, timeout=timeout) as response:
                    status = int(getattr(response, "status", response.getcode()))
                    raw = response.read().decode("utf-8", errors="replace")
                    if status < 200 or status >= 300:
                        if status in RETRYABLE_STATUS:
                            raise HttpRequestError(f"HTTP {status}", status=status, body=raw[:500])
                        raise HttpRequestError(
                            f"{error_prefix}请求失败：HTTP {status} {raw[:500]}",
                            status=status,
                            body=raw[:500],
                        )
                    try:
                        return json.loads(raw)
                    except json.JSONDecodeError as exc:
                        raise HttpRequestError(f"{error_prefix}返回内容不是 JSON", body=raw[:500]) from exc
            except urllib.error.HTTPError as exc:
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", errors="replace")[:500]
                except OSError:
                    pass
                last_error = HttpRequestError(
                    f"{error_prefix}请求失败：HTTP {exc.code} {detail}",
                    status=exc.code,
                    body=detail,
                )
                if exc.code not in RETRYABLE_STATUS:
                    raise last_error from exc
            except (urllib.error.URLError, TimeoutError, OSError, HttpRequestError) as exc:
                last_error = exc
                if isinstance(exc, HttpRequestError) and exc.status is not None and exc.status not in RETRYABLE_STATUS:
                    raise
            if attempt < max(0, int(retries)):
                self.sleeper(2**attempt)
        raise HttpRequestError(f"{error_prefix}请求重试失败：{last_error}") from last_error

    def open(
        self,
        request: urllib.request.Request,
        *,
        timeout: int,
        retries: int,
        error_prefix: str,
    ) -> Any:
        """打开流式响应，交由调用方读取并关闭响应对象。"""

        last_error: Exception | None = None
        for attempt in range(max(0, int(retries)) + 1):
            try:
                response = self.opener(request, timeout=timeout)
                status = int(getattr(response, "status", response.getcode()))
                if status < 200 or status >= 300:
                    body = ""
                    try:
                        body = response.read().decode("utf-8", errors="replace")[:500]
                    finally:
                        response.close()
                    if status not in RETRYABLE_STATUS:
                        raise HttpRequestError(
                            f"{error_prefix}请求失败：HTTP {status} {body}",
                            status=status,
                            body=body,
                        )
                    raise HttpRequestError(f"HTTP {status}", status=status, body=body)
                return response
            except urllib.error.HTTPError as exc:
                last_error = HttpRequestError(f"{error_prefix}请求失败：HTTP {exc.code}", status=exc.code)
                if exc.code not in RETRYABLE_STATUS:
                    raise last_error from exc
            except (urllib.error.URLError, TimeoutError, OSError, HttpRequestError) as exc:
                last_error = exc
                if isinstance(exc, HttpRequestError) and exc.status is not None and exc.status not in RETRYABLE_STATUS:
                    raise
            if attempt < max(0, int(retries)):
                self.sleeper(2**attempt)
        raise HttpRequestError(f"{error_prefix}请求重试失败：{last_error}") from last_error


__all__ = ["HttpRequestError", "JsonHttpClient", "RETRYABLE_STATUS"]
