"""Agent 入库输入源的校验、下载、去重和原子 staging。"""

from __future__ import annotations

import ipaddress
from pathlib import Path
import re
import shutil
import socket
from typing import Any, Callable
import urllib.error
import urllib.parse
import urllib.request

from paper_rag.config import Settings
from paper_rag.utils import sha256_file, slugify_title


PDF_MAGIC = b"%PDF-"
ARXIV_HOSTS = {"arxiv.org", "www.arxiv.org", "export.arxiv.org"}
Progress = Callable[[str], None]


class SourceValidationError(ValueError):
    """入库输入不满足本地路径或 URL 安全约束。"""


def preview_sources(settings: Settings, sources: list[str]) -> list[dict[str, Any]]:
    if not isinstance(sources, list) or not sources:
        raise SourceValidationError("sources 必须是非空列表")
    previews = []
    existing = existing_pdf_hashes(settings.pdf_dir)
    for raw in sources:
        spec = parse_source(settings, raw)
        item: dict[str, Any] = {
            "source": spec["source"],
            "type": spec["type"],
            "duplicate": False,
        }
        if spec["type"] == "local_pdf":
            path = Path(spec["value"])
            max_bytes = int(getattr(settings, "mcp_max_download_mb", 100) or 100) * 1024 * 1024
            validate_pdf_file(path, max_bytes=max_bytes)
            item["size_bytes"] = path.stat().st_size
            item["sha256"] = sha256_file(path)
            item["duplicate"] = item["sha256"] in existing
        else:
            item["url"] = spec["value"]
            item["size_bytes"] = None
            item["sha256"] = None
        previews.append(item)
    return previews


def stage_sources(
    settings: Settings,
    sources: list[str],
    job_id: str,
    *,
    reporter: Progress | None = None,
) -> list[dict[str, Any]]:
    """把来源安全地放入 data/pdf，并返回实际新增文件摘要。"""
    specs = [parse_source(settings, raw) for raw in sources]
    settings.pdf_dir.mkdir(parents=True, exist_ok=True)
    staging_root = settings.data_dir / "index" / "mcp_staging" / job_id
    staging_root.mkdir(parents=True, exist_ok=True)
    existing = existing_pdf_hashes(settings.pdf_dir)
    results: list[dict[str, Any]] = []
    try:
        for index, spec in enumerate(specs, start=1):
            emit(reporter, f"正在准备第 {index}/{len(specs)} 个来源")
            temp_path = staging_root / f"source_{index:04d}.pdf"
            if spec["type"] == "local_pdf":
                shutil.copyfile(spec["value"], temp_path)
            else:
                download_pdf(spec["value"], temp_path, settings)
            validate_pdf_file(temp_path, max_bytes=int(getattr(settings, "mcp_max_download_mb", 100) or 100) * 1024 * 1024)
            file_hash = sha256_file(temp_path)
            if file_hash in existing:
                results.append({
                    "source": spec["source"],
                    "type": spec["type"],
                    "sha256": file_hash,
                    "duplicate": True,
                    "path": None,
                })
                temp_path.unlink(missing_ok=True)
                continue
            stem = slugify_title(Path(urllib.parse.urlparse(spec["value"]).path).stem if spec["type"] == "url" else Path(spec["value"]).stem)
            target = unique_target(settings.pdf_dir, f"{stem}_{file_hash[:8]}.pdf")
            target.parent.mkdir(parents=True, exist_ok=True)
            temp_path.replace(target)
            existing.add(file_hash)
            results.append({
                "source": spec["source"],
                "type": spec["type"],
                "sha256": file_hash,
                "duplicate": False,
                "path": str(target),
            })
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
    return results


def parse_source(settings: Settings, raw: str) -> dict[str, str]:
    source = str(raw or "").strip()
    if not source:
        raise SourceValidationError("sources 中不能包含空值")
    path_candidate = Path(source)
    # Windows 盘符包含冒号，不能先按 URL scheme 解析。
    is_windows_path = bool(re.match(r"^[A-Za-z]:[\\/]", source))
    if path_candidate.is_absolute() or is_windows_path:
        path = path_candidate
        if path.suffix.casefold() != ".pdf" or not path.is_file():
            raise SourceValidationError("本地来源必须是存在的 PDF 文件")
        resolved = path.resolve()
        if not any(is_within(resolved, root) for root in settings.mcp_input_roots):
            raise SourceValidationError("本地 PDF 不在 PAPER_RAG_INPUT_ROOTS 允许范围内")
        return {"source": source, "type": "local_pdf", "value": str(resolved)}
    parsed = urllib.parse.urlparse(source)
    if parsed.scheme:
        if parsed.scheme.lower() != "https":
            raise SourceValidationError("URL 只允许使用 HTTPS")
        if parsed.username or parsed.password:
            raise SourceValidationError("URL 不允许包含用户凭据")
        normalized = normalize_arxiv_url(source)
        validate_public_url(normalized)
        if not is_pdf_url(normalized):
            raise SourceValidationError("URL 必须是 PDF 直链或 arXiv PDF 地址")
        return {"source": source, "type": "url", "value": normalized}
    raise SourceValidationError("本地来源必须使用绝对路径")


def normalize_arxiv_url(value: str) -> str:
    parsed = urllib.parse.urlparse(value)
    host = (parsed.hostname or "").casefold()
    if host not in ARXIV_HOSTS:
        return value
    path = parsed.path.rstrip("/")
    match = re.fullmatch(r"/abs/(.+)", path, flags=re.I)
    if match:
        return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, f"/pdf/{match.group(1)}.pdf", "", "", ""))
    if path.casefold().startswith("/pdf/"):
        if not path.casefold().endswith(".pdf"):
            path += ".pdf"
        return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))
    return value


def is_pdf_url(value: str) -> bool:
    parsed = urllib.parse.urlparse(value)
    host = (parsed.hostname or "").casefold()
    return parsed.path.casefold().endswith(".pdf") or host in ARXIV_HOSTS and parsed.path.casefold().startswith("/pdf/")


def validate_public_url(value: str) -> None:
    parsed = urllib.parse.urlparse(value)
    host = parsed.hostname
    if not host:
        raise SourceValidationError("URL 缺少主机名")
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(host, parsed.port or 443, type=socket.SOCK_STREAM)}
    except OSError as exc:
        raise SourceValidationError("URL 主机名无法解析") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_unspecified:
            raise SourceValidationError("URL 目标地址属于本地或保留网络")


def download_pdf(url: str, target: Path, settings: Settings) -> None:
    validate_public_url(url)
    request = urllib.request.Request(url, headers={"User-Agent": "Paper_RAG/0.1 MCP"})
    max_bytes = int(getattr(settings, "mcp_max_download_mb", 100) or 100) * 1024 * 1024
    try:
        with urllib.request.urlopen(request, timeout=120) as response, target.open("wb") as handle:
            final_url = response.geturl()
            validate_public_url(final_url)
            total = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise SourceValidationError("PDF 超过 MCP_MAX_DOWNLOAD_MB 限制")
                handle.write(chunk)
    except (urllib.error.URLError, OSError) as exc:
        raise SourceValidationError("PDF 下载失败") from exc


def validate_pdf_file(path: Path, *, max_bytes: int) -> None:
    if path.stat().st_size > max_bytes:
        raise SourceValidationError("PDF 超过大小限制")
    with path.open("rb") as handle:
        if handle.read(len(PDF_MAGIC)) != PDF_MAGIC:
            raise SourceValidationError("来源文件不是有效 PDF")


def existing_pdf_hashes(pdf_dir: Path) -> set[str]:
    hashes: set[str] = set()
    for path in pdf_dir.glob("*.pdf"):
        try:
            hashes.add(sha256_file(path))
        except OSError:
            continue
    return hashes


def unique_target(directory: Path, filename: str) -> Path:
    target = directory / filename
    if not target.exists():
        return target
    stem = target.stem
    suffix = target.suffix
    counter = 2
    while (directory / f"{stem}_{counter}{suffix}").exists():
        counter += 1
    return directory / f"{stem}_{counter}{suffix}"


def is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root.resolve())
        return True
    except ValueError:
        return False


def emit(reporter: Progress | None, message: str) -> None:
    if reporter:
        reporter(message)
