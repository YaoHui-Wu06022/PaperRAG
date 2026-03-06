from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Iterator


@dataclass
class OperationTrace:
    operation: str
    log_path: Path
    metadata: dict[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=time.perf_counter)
    started_wall_time: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    stage_timings_ms: dict[str, float] = field(default_factory=dict)
    fields: dict[str, Any] = field(default_factory=dict)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.stage_timings_ms[name] = round(
                (time.perf_counter() - start) * 1000.0,
                3,
            )

    def set_field(self, name: str, value: Any) -> None:
        self.fields[name] = value

    def finish(self, *, status: str, error: str | None = None) -> None:
        payload = {
            "operation": self.operation,
            "status": status,
            "started_at": self.started_wall_time,
            "total_ms": round((time.perf_counter() - self.started_at) * 1000.0, 3),
            "metadata": self.metadata,
            "stage_timings_ms": self.stage_timings_ms,
            "fields": self.fields,
        }
        if error:
            payload["error"] = error
        _append_jsonl(self.log_path, payload)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str))
        handle.write("\n")
