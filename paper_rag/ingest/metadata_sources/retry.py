"""外部元数据请求的轻量重试封装。"""

from __future__ import annotations

import socket
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from http.client import HTTPResponse


RETRIABLE_HTTP_CODES = {429, 500, 502, 503, 504}


def urlopen_with_retry(
    request: urllib.request.Request,
    *,
    timeout: int,
    attempts: int = 2,
    delay_seconds: float = 1.0,
) -> HTTPResponse:
    """只重试限流、服务端错误和临时网络异常。"""
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            if exc.code not in RETRIABLE_HTTP_CODES or attempt == attempts - 1:
                raise
            last_error = exc
        except (TimeoutError, socket.timeout, urllib.error.URLError) as exc:
            if attempt == attempts - 1:
                raise
            last_error = exc
        if delay_seconds > 0:
            time.sleep(delay_seconds)
    if last_error:
        raise last_error
    raise RuntimeError("urlopen_with_retry 未返回结果且没有记录异常")
