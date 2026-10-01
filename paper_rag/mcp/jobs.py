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


ProgressReporter = Callable[..., None]
JobWorker = Callable[..., dict[str, Any]]


class JobManager:
    """用单线程执行器串行运行会修改本地库的任务。"""

    def __init__(self, settings: Settings):
        self.path = Path(settings.mcp_job_log_path or settings.data_dir / "index" / "mcp_jobs.jsonl")
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
        interrupted: list[dict[str, Any]] = []
        for record in self._records.values():
            if record.get("status") in {"queued", "running"}:
                record["status"] = "interrupted"
                record["finished_at"] = _utc_now()
                record["error"] = "MCP 进程重启，任务未自动恢复"
                interrupted.append(record.copy())
        for record in interrupted:
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
        self._update(
            job_id,
            status="running",
            started_at=_utc_now(),
            progress={"phase": "queued", "completed": 0, "total": None, "message": "任务已开始"},
        )
        try:
            reporter = lambda phase, message, completed=None, total=None: self.update_progress(
                job_id, phase, message, completed, total
            )
            parameters = inspect.signature(worker).parameters
            result = worker(reporter, job_id) if len(parameters) >= 2 else worker(reporter)
            self._update(job_id, status="succeeded", finished_at=_utc_now(), result=_json_safe(result), error=None)
        except Exception as exc:
            self._update(job_id, status="failed", finished_at=_utc_now(), error=sanitize_error(exc))

    def update_progress(
        self,
        job_id: str,
        phase: str,
        message: str,
        completed: int | None = None,
        total: int | None = None,
    ) -> None:
        with self._lock:
            record = self._records.get(job_id)
            if not record:
                return
            progress = dict(record.get("progress") or {})
            progress.update({"phase": phase, "message": str(message)})
            if completed is not None:
                progress["completed"] = int(completed)
            if total is not None:
                progress["total"] = int(total)
            self._update(job_id, progress=progress)

    def status(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._records.get(str(job_id))
            return self._copy(record) if record else {"job_id": str(job_id), "status": "not_found"}

    def active_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                self._copy(record)
                for record in self._records.values()
                if record.get("status") in {"queued", "running"}
            ]

    def _update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            record = self._records.get(job_id)
            if not record:
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
    """去除 API key、Authorization 和完整请求头后保留错误摘要。"""
    message = str(exc) or exc.__class__.__name__
    message = re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", message)
    message = re.sub(r"(?i)(authorization|api[-_ ]?key|bearer)\s*[:=]\s*[^,; ]+", r"\1=[REDACTED]", message)
    return f"{exc.__class__.__name__}: {message[:1000]}"
