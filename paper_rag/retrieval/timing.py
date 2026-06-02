"""debug 模式下的轻量分段耗时记录。"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator


class Timings:
    """只在启用时记录毫秒耗时。"""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.values: dict[str, float] = {}

    @contextmanager
    def measure(self, key: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            self.values[key] = self.values.get(key, 0.0) + (time.perf_counter() - start) * 1000

    def as_dict(self) -> dict[str, float]:
        return {key: round(value, 2) for key, value in self.values.items()}


def attach_timings(payload: dict[str, Any], timings: Timings) -> dict[str, Any]:
    """把 timing 写入 debug 节点。"""
    if not timings.enabled or not timings.values:
        return payload
    debug = payload.setdefault("debug", {})
    debug["timings_ms"] = timings.as_dict()
    return payload
