"""MCP 写任务队列与持久化状态。"""

from __future__ import annotations

from collections.abc import Callable
import datetime as dt
import inspect
import json
from pathlib import Path
import re
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any
from uuid import uuid4

from paper_rag.config import Settings


JobWorker = Callable[..., dict[str, Any]]


class JobManager:
    """用单线程执行器串行运行会修改本地原始资料的任务。"""

    def __init__(self, settings: Settings):
        self.path = settings.mcp_job_log_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._records: dict[str, dict[str, Any]] = {}
        self._futures: dict[str, Future[Any]] = {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="paper-rag-mcp")
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict) and record.get("job_id"):
                    self._records[str(record["job_id"])] = record
        for record in list(self._records.values()):
            if record.get("status") in {"queued", "running"}:
                record["status"] = "interrupted"
                record["finished_at"] = _utc_now()
                record["error"] = "MCP 进程重启，任务未自动恢复"
                self._append(record)

    def submit(self, kind: str, worker: JobWorker) -> dict[str, Any]:
        job_id = f"job_{uuid4().hex}"
        record = {
            "job_id": job_id,
            "kind": kind,
            "status": "queued",
            "created_at": _utc_now(),
            "started_at": None,
            "finished_at": None,
            "progress": {"phase": None, "completed": 0, "total": None, "message": "等待执行"},
            "result": {},
            "error": None,
        }
        with self._lock:
            self._records[job_id] = record
            self._append(record)
            self._futures[job_id] = self._executor.submit(self._run, job_id, worker)
        return self._copy(record)

    def _run(self, job_id: str, worker: JobWorker) -> None:
        self._update(job_id, status="running", started_at=_utc_now())
        try:
            reporter = lambda message: self.update_progress(job_id, message)
            parameters = inspect.signature(worker).parameters
            result = worker(reporter) if parameters else worker()
            self._update(job_id, status="succeeded", finished_at=_utc_now(), result=_json_safe(result))
        except Exception as exc:
            self._update(job_id, status="failed", finished_at=_utc_now(), error=sanitize_error(exc))

    def update_progress(self, job_id: str, message: str) -> None:
        with self._lock:
            record = self._records.get(job_id)
            if record:
                progress = dict(record.get("progress") or {})
                progress.update({"phase": "download", "message": str(message)})
                self._update(job_id, progress=progress)

    def status(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            return self._copy(self._records.get(str(job_id))) or {"job_id": str(job_id), "status": "not_found"}

    def active_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                self._copy(record)
                for record in self._records.values()
                if record.get("status") in {"queued", "running"}
            ]

    def _update(self, job_id: str, **changes: Any) -> None:
        record = self._records.get(job_id)
        if record is None:
            return
        record.update(changes)
        self._append(record)

    def _append(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    @staticmethod
    def _copy(record: dict[str, Any] | None) -> dict[str, Any]:
        return json.loads(json.dumps(record or {}, ensure_ascii=False))


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "__dict__"):
        return _json_safe(vars(value))
    return str(value)


def sanitize_error(exc: Exception) -> str:
    message = str(exc) or exc.__class__.__name__
    message = re.sub(r"(?i)(authorization|api[-_ ]?key|bearer)\s*[:=]\s*[^,; ]+", r"\1=[REDACTED]", message)
    return f"{exc.__class__.__name__}: {message[:1000]}"

