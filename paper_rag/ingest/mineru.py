"""MinerU v4 本地 ArXiv PDF 解析客户端。"""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime as dt
import hashlib
import http.client
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import tempfile
import time
from typing import Any, Callable
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile

from paper_rag.acquisition.arxiv import normalize_arxiv_input, safe_base_id
from paper_rag.config import Settings
from paper_rag.http import HttpRequestError, JsonHttpClient


Progress = Callable[[str], None]
_TERMINAL_STATES = {"done", "failed"}
_MAX_RESULT_BYTES = 200 * 1024 * 1024


class MinerUError(RuntimeError):
    """MinerU 请求、上传、轮询或结果落盘失败。"""


@dataclass
class MinerUItemResult:
    input: str
    canonical_id: str | None
    status: str
    output_dir: str | None = None
    task_id: str | None = None
    batch_id: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "input": self.input,
            "canonical_id": self.canonical_id,
            "status": self.status,
            "output_dir": self.output_dir,
            "task_id": self.task_id,
            "batch_id": self.batch_id,
            "error": self.error,
        }


@dataclass
class MinerUBatchResult:
    items: list[MinerUItemResult] = field(default_factory=list)

    @property
    def succeeded(self) -> int:
        return sum(item.status == "ingested" for item in self.items)

    @property
    def skipped(self) -> int:
        return sum(item.status == "skipped" for item in self.items)

    @property
    def failed(self) -> int:
        return sum(item.status == "failed" for item in self.items)

    def to_dict(self) -> dict[str, Any]:
        return {
            "items": [item.to_dict() for item in self.items],
            "succeeded": self.succeeded,
            "skipped": self.skipped,
            "failed": self.failed,
        }


@dataclass(frozen=True)
class _IngestPlan:
    raw_input: str
    base_id: str | None
    canonical_id: str | None
    source_dir: Path | None
    pdf_path: Path | None
    metadata: dict[str, Any] | None
    status: str
    error: str | None = None
    duplicate_of: int | None = None


class MinerUClient:
    """使用 MinerU v4 签名上传接口解析单个本地 PDF。"""

    def __init__(
        self,
        settings: Settings,
        *,
        opener: Callable[..., Any] | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        self.settings = settings
        self.opener = opener or urllib.request.urlopen
        self.sleeper = sleeper
        self.http = JsonHttpClient(opener=self.opener, sleeper=self.sleeper)

    def parse_pdf(
        self,
        pdf_path: Path,
        output_dir: Path,
        *,
        base_id: str,
        canonical_id: str,
        source_sha256: str,
        reporter: Progress | None = None,
    ) -> MinerUItemResult:
        if not self.settings.mineru_api_key:
            raise MinerUError("MINERU_API_KEY 未配置")
        data_id = _data_id(canonical_id)
        _emit(reporter, f"正在向 MinerU 提交 {canonical_id}")
        batch = self._create_upload_batch(pdf_path, data_id)
        batch_id = str(batch.get("batch_id") or "")
        upload_urls = batch.get("file_urls") or []
        if not batch_id or len(upload_urls) != 1:
            raise MinerUError("MinerU 返回的上传批次缺少 batch_id 或 file_urls")
        self._upload_file(str(upload_urls[0]), pdf_path)
        _emit(reporter, f"已上传 {canonical_id}，等待 MinerU 解析")
        result = self._wait_result(batch_id, data_id, pdf_path.name, reporter)
        manifest = {
            "schema_version": 1,
            "source": "arxiv",
            "base_id": base_id,
            "canonical_id": canonical_id,
            "source_sha256": source_sha256,
            "task_id": result.get("task_id"),
            "batch_id": batch_id,
            "model_version": self.settings.mineru_model_version,
            "language": self.settings.mineru_language,
            "state": "done",
            "processed_at": _utc_now(),
        }
        output = self._download_and_prepare_result(
            str(result.get("full_zip_url") or ""),
            output_dir,
            manifest,
        )
        return MinerUItemResult(
            input=canonical_id,
            canonical_id=canonical_id,
            status="ingested",
            output_dir=str(output),
            task_id=_optional_str(result.get("task_id")),
            batch_id=batch_id,
        )

    def _create_upload_batch(self, pdf_path: Path, data_id: str) -> dict[str, Any]:
        payload = {
            "files": [{"name": pdf_path.name, "data_id": data_id}],
            "model_version": self.settings.mineru_model_version,
            "language": self.settings.mineru_language,
            "enable_formula": True,
            "enable_table": True,
        }
        response = self._request_json("POST", "/file-urls/batch", payload)
        data = response.get("data")
        if not isinstance(data, dict):
            raise MinerUError("MinerU 上传接口返回缺少 data")
        return data

    def _upload_file(self, upload_url: str, pdf_path: Path) -> None:
        parsed = urllib.parse.urlparse(upload_url)
        if parsed.scheme.lower() != "https" or not parsed.netloc:
            raise MinerUError("MinerU 签名上传 URL 必须是 HTTPS")
        path = parsed.path + (f"?{parsed.query}" if parsed.query else "")
        connection = http.client.HTTPSConnection(
            parsed.netloc,
            timeout=self.settings.mineru_upload_timeout_seconds,
        )
        try:
            headers = {"Content-Length": str(pdf_path.stat().st_size)}
            with pdf_path.open("rb") as handle:
                connection.request("PUT", path, body=handle, headers=headers, encode_chunked=False)
                response = connection.getresponse()
                detail = response.read().decode("utf-8", errors="replace").strip()
                if response.status < 200 or response.status >= 300:
                    raise MinerUError(
                        f"MinerU 上传失败：HTTP {response.status}: {detail[:300]}"
                    )
        except MinerUError:
            raise
        except (OSError, http.client.HTTPException) as exc:
            raise MinerUError(f"MinerU 上传失败：{exc}") from exc
        finally:
            connection.close()

    def _wait_result(
        self,
        batch_id: str,
        data_id: str,
        file_name: str,
        reporter: Progress | None,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + self.settings.mineru_poll_timeout_seconds
        last_state = "pending"
        while time.monotonic() < deadline:
            response = self._request_json("GET", f"/extract-results/batch/{batch_id}")
            data = response.get("data") or {}
            results = data.get("extract_result") if isinstance(data, dict) else []
            if isinstance(results, dict):
                results = [results]
            match = _match_result(results, data_id, file_name)
            if match is not None:
                state = str(match.get("state") or "pending")
                last_state = state
                _emit(reporter, f"MinerU {file_name} 状态：{state}")
                if state == "done":
                    if not match.get("full_zip_url"):
                        raise MinerUError("MinerU 完成结果缺少 full_zip_url")
                    return match
                if state == "failed":
                    raise MinerUError(f"MinerU 解析失败：{match.get('err_msg') or match}")
                if state not in {"pending", "running", "converting", "waiting-file", "uploading"}:
                    raise MinerUError(f"MinerU 返回未知任务状态：{state}")
            self.sleeper(self.settings.mineru_poll_interval_seconds)
        raise MinerUError(f"MinerU 轮询超时：{batch_id}（最后状态：{last_state}）")

    def _download_and_prepare_result(
        self,
        zip_url: str,
        output_dir: Path,
        manifest: dict[str, Any],
    ) -> Path:
        if not zip_url:
            raise MinerUError("MinerU 结果缺少下载地址")
        stage_parent = output_dir.parent
        stage_parent.mkdir(parents=True, exist_ok=True)
        stage_root = Path(tempfile.mkdtemp(prefix=".mineru-", dir=stage_parent))
        extracted = stage_root / "extracted"
        normalized = stage_root / "normalized"
        extracted.mkdir()
        normalized.mkdir()
        archive_path = stage_root / "result.zip"
        try:
            self._download_zip(zip_url, archive_path)
            with zipfile.ZipFile(archive_path) as archive:
                _validate_zip_members(archive)
                archive.extractall(extracted)
            _normalize_result_tree(extracted, normalized)
            manifest["files"] = _relative_files(normalized)
            (normalized / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            target = _promote_directory(normalized, output_dir)
            return target
        finally:
            shutil.rmtree(stage_root, ignore_errors=True)

    def _download_zip(self, zip_url: str, destination: Path) -> None:
        request = urllib.request.Request(zip_url, method="GET", headers={"Accept": "application/zip"})
        try:
            with self.opener(request, timeout=self.settings.mineru_request_timeout_seconds) as response:
                status = _response_status(response)
                if status < 200 or status >= 300:
                    raise MinerUError(f"MinerU 结果下载失败：HTTP {status}")
                length = response.headers.get("Content-Length")
                if length and int(length) > _MAX_RESULT_BYTES:
                    raise MinerUError("MinerU 结果 ZIP 超过 200MB 限制")
                total = 0
                with destination.open("wb") as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        total += len(chunk)
                        if total > _MAX_RESULT_BYTES:
                            raise MinerUError("MinerU 结果 ZIP 超过 200MB 限制")
                        handle.write(chunk)
        except MinerUError:
            raise
        except (urllib.error.HTTPError, urllib.error.URLError, OSError, ValueError) as exc:
            raise MinerUError(f"MinerU 结果下载失败：{exc}") from exc

    def _request_json(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.settings.mineru_api_base_url.rstrip('/')}/{path.lstrip('/')}"
        try:
            data = self.http.request_json(
                method,
                url,
                payload,
                headers={
                    "Authorization": f"Bearer {self.settings.mineru_api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                timeout=self.settings.mineru_request_timeout_seconds,
                retries=2,
                error_prefix="MinerU ",
            )
        except HttpRequestError as exc:
            raise MinerUError(str(exc)) from exc
        if not isinstance(data, dict):
            raise MinerUError("MinerU 返回格式不是 JSON 对象")
        if data.get("code") not in (None, 0):
            raise MinerUError(f"MinerU API 调用失败：{data.get('msg') or data.get('code')}")
        return data


def preview_arxiv_ingest(settings: Settings, inputs: list[str]) -> list[dict[str, Any]]:
    """检查 ArXiv 本地资产和 MinerU 输出，不访问 MinerU 网络接口。"""

    plans = _plan_inputs(settings, inputs)
    return [_plan_to_dict(plan, settings) for plan in plans]


def ingest_arxiv_inputs(
    settings: Settings,
    inputs: list[str],
    reporter: Progress | None = None,
) -> MinerUBatchResult:
    """逐项把已下载的 ArXiv PDF 送入 MinerU，并保留部分成功结果。"""

    plans = _plan_inputs(settings, inputs)
    result = MinerUBatchResult()
    client = MinerUClient(settings)
    completed: dict[str, MinerUItemResult] = {}
    for number, plan in enumerate(plans, start=1):
        _emit(reporter, f"正在处理第 {number}/{len(plans)} 个 MinerU 输入")
        if plan.duplicate_of is not None and plan.base_id:
            previous = completed.get(plan.base_id)
            result.items.append(
                MinerUItemResult(
                    plan.raw_input,
                    previous.canonical_id if previous else plan.canonical_id,
                    "skipped",
                    output_dir=previous.output_dir if previous else None,
                    error="批次内重复输入",
                )
            )
            continue
        if plan.status == "already_parsed":
            item = MinerUItemResult(
                plan.raw_input,
                plan.canonical_id,
                "skipped",
                output_dir=str((plan.source_dir or Path()) / "mineru"),
            )
        elif plan.status != "ready" or not plan.pdf_path or not plan.source_dir or not plan.metadata:
            item = MinerUItemResult(plan.raw_input, plan.canonical_id, "failed", error=plan.error or plan.status)
        else:
            try:
                item = client.parse_pdf(
                    plan.pdf_path,
                    plan.source_dir / "mineru",
                    base_id=plan.base_id or "",
                    canonical_id=plan.canonical_id or "",
                    source_sha256=_sha256(plan.pdf_path),
                    reporter=reporter,
                )
                item.input = plan.raw_input
            except Exception as exc:
                item = MinerUItemResult(plan.raw_input, plan.canonical_id, "failed", error=str(exc))
        if plan.base_id:
            completed[plan.base_id] = item
        result.items.append(item)
    return result


def _plan_inputs(settings: Settings, inputs: list[str]) -> list[_IngestPlan]:
    if not isinstance(inputs, list) or not inputs:
        raise MinerUError("inputs 必须是非空列表")
    plans: list[_IngestPlan] = []
    seen: dict[str, int] = {}
    for index, raw in enumerate(inputs):
        raw_input = str(raw)
        try:
            ref = normalize_arxiv_input(raw_input)
            if ref.base_id in seen:
                first = plans[seen[ref.base_id]]
                plans.append(
                    _IngestPlan(
                        raw_input,
                        ref.base_id,
                        first.canonical_id,
                        first.source_dir,
                        first.pdf_path,
                        first.metadata,
                        "duplicate",
                        duplicate_of=seen[ref.base_id],
                    )
                )
                continue
            seen[ref.base_id] = index
            source_dir = settings.arxiv_data_dir / safe_base_id(ref.base_id)
            pdf_path = source_dir / "paper.pdf"
            metadata_path = source_dir / "metadata.json"
            if not pdf_path.is_file() or not metadata_path.is_file():
                plans.append(_IngestPlan(raw_input, ref.base_id, None, source_dir, pdf_path, None, "missing_source", "ArXiv PDF 或 metadata.json 不存在"))
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            canonical_id = str(metadata.get("canonical_id") or "")
            if not canonical_id:
                raise MinerUError("metadata.json 缺少 canonical_id")
            status = _local_status(source_dir / "mineru", metadata, pdf_path, settings)
            plans.append(_IngestPlan(raw_input, ref.base_id, canonical_id, source_dir, pdf_path, metadata, status))
        except Exception as exc:
            plans.append(_IngestPlan(raw_input, None, None, None, None, None, "failed", str(exc)))
    return plans


def _local_status(output_dir: Path, metadata: dict[str, Any], pdf_path: Path, settings: Settings) -> str:
    manifest_path = output_dir / "manifest.json"
    if not output_dir.is_dir() or not manifest_path.is_file() or not (output_dir / "full.md").is_file():
        return "ready"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "ready"
    if (
        manifest.get("canonical_id") == metadata.get("canonical_id")
        and manifest.get("source_sha256") == _sha256(pdf_path)
        and manifest.get("model_version") == settings.mineru_model_version
        and manifest.get("language") == settings.mineru_language
    ):
        return "already_parsed"
    return "ready"


def _plan_to_dict(plan: _IngestPlan, settings: Settings) -> dict[str, Any]:
    result: dict[str, Any] = {
        "input": plan.raw_input,
        "status": plan.status,
        "source_pdf": str(plan.pdf_path) if plan.pdf_path else None,
        "output_dir": str(plan.source_dir / "mineru") if plan.source_dir else None,
        "canonical_id": plan.canonical_id,
    }
    if plan.metadata:
        result["model_version"] = settings.mineru_model_version
        result["language"] = settings.mineru_language
    if plan.error:
        result["error"] = plan.error
    if plan.duplicate_of is not None:
        result["duplicate_of"] = plan.duplicate_of
    return result


def _normalize_result_tree(extracted: Path, destination: Path) -> None:
    markdown_files = sorted(extracted.rglob("full.md"))
    if len(markdown_files) != 1:
        if not markdown_files:
            raise MinerUError("MinerU ZIP 结果缺少 full.md")
        raise MinerUError("MinerU ZIP 结果包含多个 full.md，无法确定主结果")
    try:
        if not markdown_files[0].read_text(encoding="utf-8").strip():
            raise MinerUError("MinerU ZIP 结果缺少 full.md")
    except (OSError, UnicodeError) as exc:
        raise MinerUError("MinerU ZIP 的 full.md 无法读取") from exc
    all_content_lists = sorted(path for path in extracted.rglob("*.json") if "content_list" in path.name.casefold())
    exact_lists = [path for path in all_content_lists if path.name.casefold() in {"content_list.json", "_content_list.json"} or path.name.casefold().endswith("_content_list.json")]
    content_lists = exact_lists if exact_lists else all_content_lists
    if len(content_lists) != 1:
        raise MinerUError("MinerU ZIP 结果必须包含唯一主 content_list.json")
    try:
        parsed = json.loads(content_lists[0].read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MinerUError("MinerU ZIP 的 content_list.json 无法解析") from exc
    if not isinstance(parsed, (list, dict)):
        raise MinerUError("MinerU ZIP 的 content_list.json 格式无效")
    root = markdown_files[0].parent
    for source in root.rglob("*"):
        relative = source.relative_to(root)
        target = destination / relative
        if source.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    if not (destination / "content_list.json").is_file():
        candidates = sorted(path for path in destination.glob("*.json") if "content_list" in path.name.casefold())
        if len(candidates) == 1:
            candidates[0].replace(destination / "content_list.json")
        else:
            shutil.copy2(content_lists[0], destination / "content_list.json")
    if not (destination / "full.md").is_file() or not (destination / "content_list.json").is_file():
        raise MinerUError("MinerU ZIP 结果缺少必要文件")


def _validate_zip_members(archive: zipfile.ZipFile) -> None:
    total = 0
    for member in archive.infolist():
        safe_zip_member_target(Path("/safe-root"), member.filename)
        if member.is_dir():
            continue
        if (member.external_attr >> 16) & 0o170000 == 0o120000:
            raise MinerUError(f"MinerU ZIP 不允许符号链接: {member.filename}")
        total += int(member.file_size)
        if total > _MAX_RESULT_BYTES:
            raise MinerUError("MinerU ZIP 解压总大小超过 200MB 限制")


def safe_zip_member_target(root: Path, member_name: str) -> Path:
    """校验 ZIP 成员路径，拒绝绝对路径、盘符和目录逃逸。"""

    normalized = member_name.replace("\\", "/")
    posix_path = PurePosixPath(normalized)
    windows_path = PureWindowsPath(member_name)
    if not normalized.strip() or posix_path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
        raise MinerUError(f"不安全的 MinerU ZIP 成员路径：{member_name}")
    if any(part == ".." for part in posix_path.parts):
        raise MinerUError(f"不安全的 MinerU ZIP 成员路径：{member_name}")
    target = (root / Path(*posix_path.parts)).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as exc:
        raise MinerUError(f"不安全的 MinerU ZIP 成员路径：{member_name}") from exc
    return target


def _promote_directory(staged: Path, target: Path) -> Path:
    backup = target.with_name(f".{target.name}.old-{uuid.uuid4().hex}")
    if target.exists():
        target.replace(backup)
    try:
        staged.replace(target)
    except Exception:
        if backup.exists() and not target.exists():
            backup.replace(target)
        raise
    finally:
        shutil.rmtree(backup, ignore_errors=True)
    return target


def _match_result(results: Any, data_id: str, file_name: str) -> dict[str, Any] | None:
    if not isinstance(results, list):
        return None
    for item in results:
        if not isinstance(item, dict):
            continue
        if item.get("data_id") == data_id or item.get("file_name") == file_name:
            return item
    return results[0] if len(results) == 1 and isinstance(results[0], dict) else None


def _relative_files(root: Path) -> list[str]:
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )


def _data_id(canonical_id: str) -> str:
    value = safe_base_id(canonical_id).replace("/", "__")
    return value[:128]


def _response_status(response: Any) -> int:
    status = getattr(response, "status", None)
    return int(status if status is not None else response.getcode())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_str(value: Any) -> str | None:
    return str(value) if value else None


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _emit(reporter: Progress | None, message: str) -> None:
    if reporter:
        reporter(message)


__all__ = [
    "MinerUBatchResult",
    "MinerUClient",
    "MinerUError",
    "MinerUItemResult",
    "ingest_arxiv_inputs",
    "preview_arxiv_ingest",
    "safe_zip_member_target",
]

