"""ArXiv 原始论文获取：元数据查询、PDF 下载和原子落盘。"""

from __future__ import annotations

import datetime as dt
import hashlib
import html
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
import shutil
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any, Callable
from uuid import uuid4

from paper_rag.config import Settings


ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
ARXIV_HOSTS = {"arxiv.org", "www.arxiv.org", "export.arxiv.org"}
MODERN_ID = re.compile(r"^(?P<base>\d{4}\.\d{4,5})(?:v(?P<version>[1-9]\d*))?$", re.I)
LEGACY_ID = re.compile(
    r"^(?P<base>[a-z][a-z0-9.-]*/\d{7})(?:v(?P<version>[1-9]\d*))?$", re.I
)
Progress = Callable[[str], None]


class ArxivError(RuntimeError):
    """ArXiv 输入、接口、下载或本地存储失败。"""


class ArxivInputError(ArxivError, ValueError):
    """输入不是允许的 ArXiv ID 或 URL。"""


@dataclass(frozen=True)
class ArxivRef:
    base_id: str
    requested_version: int | None
    canonical_input: str


@dataclass(frozen=True)
class ArxivMetadata:
    base_id: str
    version: int
    canonical_id: str
    title: str
    authors: list[str]
    abstract: str
    categories: list[str]
    published_at: str
    updated_at: str
    abs_url: str
    pdf_url: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ArxivItemResult:
    input: str
    canonical_id: str | None
    status: str
    pdf_path: str | None = None
    metadata_path: str | None = None
    sha256: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ArxivBatchResult:
    items: list[ArxivItemResult] = field(default_factory=list)

    @property
    def downloaded(self) -> int:
        return sum(item.status == "downloaded" for item in self.items)

    @property
    def skipped(self) -> int:
        return sum(item.status == "skipped" for item in self.items)

    @property
    def failed(self) -> int:
        return sum(item.status == "failed" for item in self.items)

    def to_dict(self) -> dict[str, Any]:
        return {
            "items": [item.to_dict() for item in self.items],
            "downloaded": self.downloaded,
            "skipped": self.skipped,
            "failed": self.failed,
        }


@dataclass
class _Plan:
    raw_input: str
    ref: ArxivRef | None = None
    metadata: ArxivMetadata | None = None
    local: dict[str, Any] | None = None
    status: str = "new"
    error: str | None = None
    duplicate_of: int | None = None


def normalize_arxiv_input(value: str) -> ArxivRef:
    """规范化现代/旧式 ArXiv ID 和 abs/pdf URL。"""
    raw = str(value or "").strip()
    if not raw:
        raise ArxivInputError("ArXiv 输入不能为空")
    candidate = raw
    if "://" in raw:
        parsed = urllib.parse.urlparse(raw)
        if parsed.scheme.lower() != "https":
            raise ArxivInputError("ArXiv URL 只允许使用 HTTPS")
        if parsed.username or parsed.password:
            raise ArxivInputError("ArXiv URL 不允许包含用户凭据")
        host = (parsed.hostname or "").casefold()
        if host not in ARXIV_HOSTS:
            raise ArxivInputError("URL 必须指向 arxiv.org")
        path = urllib.parse.unquote(parsed.path).rstrip("/")
        match = re.fullmatch(r"/(?:abs|pdf)/(.+?)(?:\.pdf)?", path, flags=re.I)
        if not match:
            raise ArxivInputError("URL 必须是 ArXiv abs 或 pdf 地址")
        candidate = match.group(1)
    candidate = candidate.strip().rstrip(".").casefold()
    match = MODERN_ID.fullmatch(candidate) or LEGACY_ID.fullmatch(candidate)
    if not match:
        raise ArxivInputError("不是合法的 ArXiv ID")
    base_id = match.group("base")
    requested = match.group("version")
    version = int(requested) if requested else None
    canonical = f"{base_id}{f'v{version}' if version else ''}"
    return ArxivRef(base_id=base_id, requested_version=version, canonical_input=canonical)


class ArxivClient:
    """使用标准库访问 ArXiv API 和 PDF 端点。"""

    def __init__(
        self,
        settings: Settings,
        *,
        opener: Callable[..., Any] | None = None,
        sleeper: Callable[[float], None] | None = None,
    ):
        self.settings = settings
        self.opener = opener or urllib.request.urlopen
        self.sleeper = sleeper or time.sleep
        self._last_request_at = 0.0

    def resolve_latest(self, ref: ArxivRef) -> ArxivMetadata:
        """查询论文当前版本；输入版本号不改变“保存最新版”的规则。"""
        query = urllib.parse.urlencode({"id_list": ref.base_id})
        request = urllib.request.Request(
            f"{self.settings.arxiv_api_base_url}?{query}",
            headers={"User-Agent": self.settings.arxiv_user_agent, "Accept": "application/atom+xml"},
        )
        xml_text = self._read(request, self.settings.arxiv_timeout_seconds, max_bytes=8 * 1024 * 1024)
        try:
            return parse_latest_metadata(xml_text, ref.base_id)
        except (ET.ParseError, ArxivError) as exc:
            if isinstance(exc, ArxivError):
                raise
            raise ArxivError(f"ArXiv 元数据 XML 无法解析：{exc}") from exc

    def download_pdf(self, metadata: ArxivMetadata, target: Path) -> tuple[str, int]:
        # 当前网络环境下 arxiv.org 的 PDF 端点可能返回 406，export 端点提供相同内容。
        request = urllib.request.Request(
            _export_pdf_url(metadata.pdf_url),
            headers={"User-Agent": self.settings.arxiv_user_agent, "Accept": "application/pdf"},
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        total = 0
        max_bytes = self.settings.arxiv_max_download_mb * 1024 * 1024
        try:
            response = self._open(request, self.settings.arxiv_download_timeout_seconds)
            with response:
                final_url = str(getattr(response, "geturl", lambda: request.full_url)())
                _validate_arxiv_url(final_url)
                with target.open("wb") as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        total += len(chunk)
                        if total > max_bytes:
                            raise ArxivError("PDF 超过 ARXIV_MAX_DOWNLOAD_MB 限制")
                        digest.update(chunk)
                        handle.write(chunk)
        except ArxivError:
            target.unlink(missing_ok=True)
            raise
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            target.unlink(missing_ok=True)
            raise ArxivError(f"PDF 下载失败：{exc}") from exc
        with target.open("rb") as handle:
            if handle.read(5) != b"%PDF-":
                target.unlink(missing_ok=True)
                raise ArxivError("下载内容不是有效 PDF")
        return digest.hexdigest(), total

    def _read(self, request: urllib.request.Request, timeout: int, *, max_bytes: int) -> str:
        response = self._open(request, timeout)
        chunks: list[bytes] = []
        total = 0
        with response:
            final_url = str(getattr(response, "geturl", lambda: request.full_url)())
            if not final_url.startswith(self.settings.arxiv_api_base_url):
                raise ArxivError("ArXiv API 重定向到了不允许的地址")
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise ArxivError("ArXiv API 响应过大")
                chunks.append(chunk)
        return b"".join(chunks).decode("utf-8")

    def _open(self, request: urllib.request.Request, timeout: int) -> Any:
        elapsed = time.monotonic() - self._last_request_at
        delay = self.settings.arxiv_request_delay_seconds - elapsed
        if delay > 0:
            self.sleeper(delay)
        last_error: Exception | None = None
        for attempt in range(3):
            self._last_request_at = time.monotonic()
            try:
                response = self.opener(request, timeout=timeout)
                status = int(getattr(response, "status", 200))
                if status in {429, 500, 502, 503, 504}:
                    response.close()
                    raise urllib.error.HTTPError(request.full_url, status, "retryable", {}, None)
                if status >= 400:
                    response.close()
                    raise ArxivError(f"ArXiv 请求失败：HTTP {status}")
                return response
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code not in {429, 500, 502, 503, 504}:
                    detail = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
                    raise ArxivError(f"ArXiv 请求失败：HTTP {exc.code} {detail[:300]}") from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
            if attempt < 2:
                self.sleeper(2**attempt)
        raise ArxivError(f"ArXiv 请求重试失败：{last_error}") from last_error


class ArxivStore:
    """管理 ArXiv 当前版本资产和 manifest。"""

    def __init__(self, settings: Settings, client: ArxivClient | None = None):
        self.settings = settings
        self.root = settings.arxiv_data_dir
        self.client = client or ArxivClient(settings)
        self.manifest_path = self.root / "manifest.jsonl"

    def plan(self, inputs: list[str]) -> list[_Plan]:
        if not isinstance(inputs, list) or not inputs:
            raise ArxivInputError("inputs 必须是非空列表")
        local = self._load_manifest()
        plans: list[_Plan] = []
        seen: dict[str, int] = {}
        for index, raw in enumerate(inputs):
            plan = _Plan(raw_input=str(raw))
            try:
                plan.ref = normalize_arxiv_input(str(raw))
                if plan.ref.base_id in seen:
                    first = plans[seen[plan.ref.base_id]]
                    plan.duplicate_of = seen[plan.ref.base_id]
                    plan.metadata = first.metadata
                    plan.local = first.local
                    plan.status = "duplicate"
                    plans.append(plan)
                    continue
                seen[plan.ref.base_id] = index
                plan.metadata = self.client.resolve_latest(plan.ref)
                plan.local = local.get(plan.ref.base_id)
                if plan.local and not self._asset_exists(plan.local, plan.ref.base_id):
                    plan.local = None
                plan.status = compare_status(plan.ref, plan.metadata, plan.local)
            except Exception as exc:
                plan.status = "failed"
                plan.error = str(exc)
            plans.append(plan)
        return plans

    def preview(self, inputs: list[str]) -> list[dict[str, Any]]:
        return [self._preview_dict(plan) for plan in self.plan(inputs)]

    def download(self, inputs: list[str], reporter: Progress | None = None) -> ArxivBatchResult:
        plans = self.plan(inputs)
        result = ArxivBatchResult()
        completed: dict[str, ArxivItemResult] = {}
        for number, plan in enumerate(plans, start=1):
            emit(reporter, f"正在处理第 {number}/{len(plans)} 个 ArXiv 输入")
            if plan.duplicate_of is not None and plan.ref:
                previous = completed.get(plan.ref.base_id)
                if previous:
                    result.items.append(
                        ArxivItemResult(
                            input=plan.raw_input,
                            canonical_id=previous.canonical_id,
                            status="skipped",
                            pdf_path=previous.pdf_path,
                            metadata_path=previous.metadata_path,
                            sha256=previous.sha256,
                            error="批次内重复输入",
                        )
                    )
                    continue
            if plan.status == "failed" or not plan.metadata or not plan.ref:
                item = ArxivItemResult(plan.raw_input, None, "failed", error=plan.error or "未知错误")
            elif plan.status in {"already_latest", "downgrade_rejected"}:
                item = self._skipped_result(plan)
            else:
                try:
                    item = self._download_plan(plan)
                except Exception as exc:
                    item = ArxivItemResult(
                        plan.raw_input,
                        plan.metadata.canonical_id,
                        "failed",
                        error=str(exc),
                    )
            if plan.ref:
                completed[plan.ref.base_id] = item
            result.items.append(item)
        return result

    def _download_plan(self, plan: _Plan) -> ArxivItemResult:
        assert plan.ref and plan.metadata
        self.root.mkdir(parents=True, exist_ok=True)
        safe_id = safe_base_id(plan.ref.base_id)
        staging_root = Path(tempfile.mkdtemp(prefix=".arxiv-", dir=self.root))
        stage_dir = staging_root / safe_id
        stage_dir.mkdir()
        stage_pdf = stage_dir / "paper.pdf"
        try:
            digest, size = self.client.download_pdf(plan.metadata, stage_pdf)
            metadata = {
                "schema_version": 1,
                **plan.metadata.to_dict(),
                "sha256": digest,
                "size_bytes": size,
                "downloaded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
            (stage_dir / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            target_dir = self.root / safe_id
            self._promote(stage_dir, target_dir)
            record = {
                "base_id": plan.ref.base_id,
                "canonical_id": plan.metadata.canonical_id,
                "path": str(target_dir),
                **metadata,
            }
            self._update_manifest(record)
            return ArxivItemResult(
                plan.raw_input,
                plan.metadata.canonical_id,
                "downloaded",
                str(target_dir / "paper.pdf"),
                str(target_dir / "metadata.json"),
                digest,
            )
        finally:
            shutil.rmtree(staging_root, ignore_errors=True)

    def _promote(self, staged: Path, target: Path) -> None:
        backup = target.with_name(f".{target.name}.old-{uuid4().hex}")
        if target.exists():
            target.replace(backup)
        try:
            staged.replace(target)
        except Exception:
            if backup.exists() and not target.exists():
                backup.replace(target)
            raise
        shutil.rmtree(backup, ignore_errors=True)

    def _load_manifest(self) -> dict[str, dict[str, Any]]:
        records: dict[str, dict[str, Any]] = {}
        if not self.manifest_path.exists():
            return records
        for line in self.manifest_path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict) and record.get("base_id"):
                records[str(record["base_id"])] = record
        return records

    def _asset_exists(self, record: dict[str, Any], base_id: str) -> bool:
        directory = Path(record.get("path", "")) if record.get("path") else self.root / safe_base_id(base_id)
        return (directory / "paper.pdf").is_file() and (directory / "metadata.json").is_file()

    def _update_manifest(self, record: dict[str, Any]) -> None:
        records = self._load_manifest()
        records[record["base_id"]] = record
        self.root.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix="manifest-", suffix=".jsonl", dir=self.root)
        os.close(fd)
        temporary_path = Path(temporary)
        try:
            temporary_path.write_text(
                "".join(
                    json.dumps(records[key], ensure_ascii=False, sort_keys=True) + "\n"
                    for key in sorted(records)
                ),
                encoding="utf-8",
            )
            temporary_path.replace(self.manifest_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _preview_dict(self, plan: _Plan) -> dict[str, Any]:
        item: dict[str, Any] = {"input": plan.raw_input, "status": plan.status}
        if plan.ref:
            item.update(
                {
                    "base_id": plan.ref.base_id,
                    "requested_version": plan.ref.requested_version,
                    "target_dir": str(self.root / safe_base_id(plan.ref.base_id)),
                }
            )
        if plan.metadata:
            item.update(
                {
                    "canonical_id": plan.metadata.canonical_id,
                    "title": plan.metadata.title,
                    "version": plan.metadata.version,
                    "pdf_url": plan.metadata.pdf_url,
                }
            )
        if plan.local:
            item["local_version"] = plan.local.get("version")
        if plan.error:
            item["error"] = plan.error
        if plan.duplicate_of is not None:
            item["duplicate_of"] = plan.duplicate_of
        return item

    def _skipped_result(self, plan: _Plan) -> ArxivItemResult:
        assert plan.ref and plan.metadata
        local_path = plan.local.get("path") if plan.local else None
        target_dir = Path(local_path) if local_path else None
        if target_dir is None or not target_dir.exists():
            target_dir = self.root / safe_base_id(plan.ref.base_id)
        return ArxivItemResult(
            plan.raw_input,
            plan.metadata.canonical_id,
            "skipped",
            str(target_dir / "paper.pdf"),
            str(target_dir / "metadata.json"),
            plan.local.get("sha256") if plan.local else None,
            "本地已存在当前版本" if plan.status == "already_latest" else "拒绝覆盖本地较新版本",
        )


def parse_latest_metadata(xml_text: str, base_id: str) -> ArxivMetadata:
    root = ET.fromstring(xml_text)
    entries = root.findall("atom:entry", ATOM_NS)
    if not entries:
        raise ArxivError(f"ArXiv 未找到论文：{base_id}")
    for entry in entries:
        entry_id = clean_text(entry.findtext("atom:id", "", ATOM_NS))
        try:
            ref = normalize_arxiv_input(_as_https_arxiv_url(entry_id))
        except ArxivInputError:
            continue
        if ref.base_id != base_id:
            continue
        title = clean_text(entry.findtext("atom:title", "", ATOM_NS))
        summary = clean_text(entry.findtext("atom:summary", "", ATOM_NS))
        authors = [
            clean_text(author.findtext("atom:name", "", ATOM_NS))
            for author in entry.findall("atom:author", ATOM_NS)
        ]
        categories = [
            category.attrib.get("term", "").strip()
            for category in entry.findall("atom:category", ATOM_NS)
            if category.attrib.get("term", "").strip()
        ]
        pdf_url = ""
        for link in entry.findall("atom:link", ATOM_NS):
            if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                pdf_url = _normalize_pdf_url(_as_https_arxiv_url(link.attrib.get("href", "")))
                break
        version = ref.requested_version or _version_from_links(entry, ref.base_id) or 1
        canonical_id = f"{ref.base_id}v{version}"
        return ArxivMetadata(
            base_id=ref.base_id,
            version=version,
            canonical_id=canonical_id,
            title=title,
            authors=[author for author in authors if author],
            abstract=summary,
            categories=categories,
            published_at=clean_text(entry.findtext("atom:published", "", ATOM_NS)),
            updated_at=clean_text(entry.findtext("atom:updated", "", ATOM_NS)),
            abs_url=f"https://arxiv.org/abs/{canonical_id}",
            pdf_url=pdf_url or f"https://arxiv.org/pdf/{canonical_id}.pdf",
        )
    raise ArxivError(f"ArXiv 返回了不匹配的论文：{base_id}")


def compare_status(ref: ArxivRef, metadata: ArxivMetadata, local: dict[str, Any] | None) -> str:
    if not local:
        return "new"
    local_version = int(local.get("version") or 0)
    if local_version < metadata.version:
        return "upgrade_available"
    if ref.requested_version and ref.requested_version < local_version:
        return "downgrade_rejected"
    return "already_latest"


def safe_base_id(base_id: str) -> str:
    return base_id.replace("/", "__")


def _version_from_links(entry: ET.Element, base_id: str) -> int | None:
    """从 API 的 abs/pdf 链接补充 entry id 未携带的版本号。"""
    for link in entry.findall("atom:link", ATOM_NS):
        href = link.attrib.get("href", "")
        try:
            ref = normalize_arxiv_input(_as_https_arxiv_url(href))
        except ArxivInputError:
            continue
        if ref.base_id == base_id and ref.requested_version:
            return ref.requested_version
    return None


def clean_text(value: str) -> str:
    return " ".join(html.unescape(value or "").split())


def _as_https_arxiv_url(value: str) -> str:
    """只对可信的 API 响应兼容旧式 http 链接。"""
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme.lower() == "http" and (parsed.hostname or "").casefold() in ARXIV_HOSTS:
        return urllib.parse.urlunparse(("https", parsed.netloc, parsed.path, parsed.params, parsed.query, parsed.fragment))
    return value


def _normalize_pdf_url(value: str) -> str:
    parsed = urllib.parse.urlparse(value)
    if (
        parsed.scheme.lower() == "https"
        and (parsed.hostname or "").casefold() in ARXIV_HOSTS
        and parsed.path.casefold().startswith("/pdf/")
        and not parsed.path.casefold().endswith(".pdf")
    ):
        parsed = parsed._replace(path=f"{parsed.path}.pdf")
        return urllib.parse.urlunparse(parsed)
    return value


def _validate_arxiv_url(value: str) -> None:
    parsed = urllib.parse.urlparse(value)
    if (
        parsed.scheme.lower() != "https"
        or parsed.username
        or parsed.password
        or (parsed.hostname or "").casefold() not in ARXIV_HOSTS
    ):
        raise ArxivError("ArXiv 请求重定向到了不允许的地址")


def _export_pdf_url(value: str) -> str:
    """把标准 arxiv.org PDF 地址切换到同源的 export 下载端点。"""

    parsed = urllib.parse.urlparse(value)
    if (parsed.hostname or "").casefold() in {"arxiv.org", "www.arxiv.org"}:
        return urllib.parse.urlunparse(parsed._replace(netloc="export.arxiv.org"))
    return value


def emit(reporter: Progress | None, message: str) -> None:
    if reporter:
        reporter(message)


def preview_arxiv_inputs(settings: Settings, inputs: list[str]) -> list[dict[str, Any]]:
    """生成只读预览。"""
    return ArxivStore(settings).preview(inputs)


def download_arxiv_inputs(
    settings: Settings,
    inputs: list[str],
    reporter: Progress | None = None,
) -> ArxivBatchResult:
    """下载一批 ArXiv 论文，单项失败不会阻塞其他输入。"""
    return ArxivStore(settings).download(inputs, reporter=reporter)

