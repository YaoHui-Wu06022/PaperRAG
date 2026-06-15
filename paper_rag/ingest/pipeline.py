"""PDF 入库主流程：同步文件、补全元数据、生成结构化数据。"""

from __future__ import annotations

import datetime as dt
import json
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from paper_rag.config import Settings
from paper_rag.utils import infer_title_from_pdf_name, normalize_text, replace_dir, safe_move_dir, sha256_file, slugify_title
from paper_rag.ingest.annotations import load_paper_annotations, save_paper_annotations, upsert_paper_annotation
from paper_rag.ingest.citation_graph import build_citation_graph
from paper_rag.ingest.extract import (
    extract_paper_data,
    extract_title,
    find_content_list_v2_path,
    flatten_pages,
    load_content_list_v2,
)
from paper_rag.ingest.manifest import Manifest, ManifestRecord, effective_year, normalize_year
from paper_rag.ingest.mineru import MinerUClient, MinerUError
from paper_rag.ingest.metadata_sources.arxiv import ArxivClient
from paper_rag.ingest.metadata_sources.dblp import DblpClient
from paper_rag.ingest.metadata_sources.semantic_scholar import SemanticScholarClient
from paper_rag.corpus.venues import normalize_venue_for_storage


MINERU_SOURCE_SIDECAR = "_paper_rag_source.json"


@dataclass
class MinerUOutputIndex:
    """可复用 MinerU 输出索引；hash 精确优先，旧数据只允许唯一标题精确命中。"""

    by_source_hash: dict[str, list[Path]] = field(default_factory=dict)
    by_title: dict[str, list[Path]] = field(default_factory=dict)
    titles: dict[Path, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.titles)


@dataclass(frozen=True)
class MetadataMatch:
    """外部元数据源返回的标准化命中结果。"""

    title: str
    authors: list[str]
    year: dict[str, int | None]
    venue: str | None


@dataclass
class IngestSummary:
    """一次入库同步的分类统计，用于 CLI 摘要输出。"""

    processed: list[str] = field(default_factory=list)
    reused: list[str] = field(default_factory=list)
    restored: list[str] = field(default_factory=list)
    renamed_mineru_output: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


Reporter = Callable[[str], None]


def clean_author_name(name: str) -> str:
    """清理 DBLP 作者名末尾常见的四位消歧编号。"""
    text = name.strip() if isinstance(name, str) else ""
    return re.sub(r"\s+\d{4}$", "", text).strip()


def clean_author_list(authors: list[str]) -> list[str]:
    cleaned_authors: list[str] = []
    for author in authors:
        cleaned = clean_author_name(author)
        if cleaned:
            cleaned_authors.append(cleaned)
    return cleaned_authors


def run_ingest(
    settings: Settings,
    *,
    refresh_metadata: bool = False,
    reporter: Reporter | None = None,
) -> IngestSummary:
    """执行一次全量同步，把 PDF 目录重建为本地结构化论文库。"""
    report = reporter or (lambda _: None)
    # 入库流程对目录是增量同步，对 paper_data 是按单篇论文可重建输出。
    report("[ingest] 正在准备数据目录")
    settings.pdf_dir.mkdir(parents=True, exist_ok=True)
    settings.mineru_output_dir.mkdir(parents=True, exist_ok=True)
    settings.paper_data_dir.mkdir(parents=True, exist_ok=True)
    settings.archive_dir.mkdir(parents=True, exist_ok=True)
    summary = IngestSummary()
    report(f"[ingest] 正在加载 manifest：{settings.manifest_path}")
    manifest = Manifest.load(settings.manifest_path)
    annotations = load_paper_annotations(settings)
    report(f"[ingest] 正在扫描 PDF：{settings.pdf_dir}")
    pdf_by_hash = scan_pdfs(settings.pdf_dir, summary)
    report(f"[ingest] 发现 {len(pdf_by_hash)} 个唯一 PDF")
    for duplicate in summary.duplicates:
        report(f"[ingest] 已跳过重复文件：{duplicate}")
    active_hashes = set(pdf_by_hash)

    report("[ingest] 正在同步已删除 PDF")
    # manifest 是以 file_hash 为主键的事实表；当前 PDF 目录中不存在的旧记录需要标记删除。
    for file_hash, record in list(manifest.records.items()):
        if record.status != "deleted" and record.pdf_path and file_hash not in active_hashes:
            report(f"[ingest] 正在归档已删除 PDF 的派生数据：{record.title or file_hash[:8]}")
            archive_deleted_record(settings, record)
            record.status = "deleted"
            record.pdf_path = None
            summary.deleted.append(record.title or file_hash[:8])
            manifest.records[record.file_hash] = record

    report(f"[ingest] 正在索引已有 MinerU 输出：{settings.mineru_output_dir}")
    # 先尝试复用本地 MinerU 输出，避免无谓重复上传和等待解析。
    output_index = build_existing_output_index(settings.mineru_output_dir)
    report(f"[ingest] 已索引 {len(output_index)} 个 MinerU 查找 key")
    dblp = DblpClient()
    semantic_scholar = SemanticScholarClient(settings.semantic_scholar_api_key)
    arxiv = ArxivClient()
    # 外部源的节流状态放在 run_ingest 内，避免不同命令执行之间互相影响。
    last_lookup_at = {"arxiv": 0.0, "dblp": 0.0, "semantic_scholar": 0.0}
    total = len(pdf_by_hash)
    for ordinal, (file_hash, pdf_path) in enumerate(pdf_by_hash.items(), start=1):
        report(f"[ingest] [{ordinal}/{total}] 正在处理 {pdf_path.name}")
        record = manifest.records.get(file_hash) or ManifestRecord(file_hash=file_hash, status="new")
        try:
            mineru_output = ensure_mineru_output(settings, record, pdf_path, file_hash, output_index, summary, report)
            report(f"[ingest] [{ordinal}/{total}] MinerU 输出：{mineru_output.name}")
            title = title_from_output(mineru_output)
            if not title:
                # 没有可信标题就不继续生成 paper_id，避免后续数据落到不可追踪目录。
                record.status = "title_unresolved"
                record.pdf_path = str(pdf_path)
                record.message = "未找到论文标题"
                summary.unresolved.append(f"{pdf_path.name}: 未找到标题")
                manifest.records[record.file_hash] = record
                report(f"[ingest] [{ordinal}/{total}] 标题未解析，已跳过 paper_data 生成")
                continue
            authors: list[str] = [] if refresh_metadata else clean_author_list(record.author or [])
            year = normalize_year(None if refresh_metadata else record.year)
            venue = None if refresh_metadata else normalize_venue_for_storage(settings, record.venue)
            # 非 refresh 模式下，已有完整 metadata 的记录不重复打外部 API。
            has_existing_metadata = bool(record.title and authors and effective_year(year))
            if not refresh_metadata and has_existing_metadata:
                title = record.title or title
            report(f"[ingest] [{ordinal}/{total}] 标题：{title}")
            if not (authors and effective_year(year)):
                # 三个外部源都有速率限制，闭包让每个源各自维护上次请求时间。
                match = lookup_metadata(
                    title,
                    dblp,
                    semantic_scholar,
                    arxiv,
                    dblp_candidate_limit=settings.dblp_candidate_limit,
                    dblp_retry_delay_seconds=settings.dblp_delay_seconds,
                    semantic_scholar_retry_delay_seconds=settings.semantic_scholar_delay_seconds,
                    arxiv_retry_delay_seconds=settings.arxiv_delay_seconds,
                    report=lambda message: report(f"[ingest] [{ordinal}/{total}] {message}"),
                    last_lookup_at=last_lookup_at,
                )
                if match:
                    title = match.title
                    authors = clean_author_list(match.authors)
                    year = match.year
                    venue = normalize_venue_for_storage(settings, match.venue)
                    report(
                        f"[ingest] [{ordinal}/{total}] 元数据匹配成功："
                        f"预印本年份={year.get('preprint_year')}, 正式发表年份={year.get('publish_year')}, "
                        f"期刊/会议={venue or '未解析'}"
                    )
                else:
                    summary.unresolved.append(f"{pdf_path.name}: ArXiv/DBLP/Semantic Scholar 未找到精确标题匹配")
                    report(f"[ingest] [{ordinal}/{total}] ArXiv、DBLP 和 Semantic Scholar 均未解析出元数据")
            else:
                report(f"[ingest] [{ordinal}/{total}] 元数据已有作者和有效年份，跳过外部检索")
            target_pdf = pdf_path
            rename_year = effective_year(year)
            if rename_year:
                # 只有拿到有效年份后才重命名，保持 PDF、MinerU 输出和 paper_data 命名一致。
                try:
                    renamed = rename_pdf_if_needed(settings.pdf_dir, pdf_path, rename_year, title, file_hash)
                except OSError as exc:
                    renamed = pdf_path
                    summary.errors.append(f"无法重命名 {pdf_path.name}，已保留原名：{exc}")
                    report(f"[ingest] [{ordinal}/{total}] PDF 重命名失败，已保留原名：{exc}")
                if renamed is None:
                    summary.errors.append(f"{pdf_path.name} 存在命名冲突，已跳过重命名")
                    report(f"[ingest] [{ordinal}/{total}] PDF 重命名冲突，已保留原名")
                else:
                    target_pdf = renamed
                    if renamed != pdf_path:
                        report(f"[ingest] [{ordinal}/{total}] 已重命名 PDF -> {renamed.name}")
                renamed_output = rename_mineru_output_if_needed(
                    settings.mineru_output_dir,
                    mineru_output,
                    rename_year,
                    title,
                )
                if renamed_output is None:
                    summary.errors.append(f"{title} 的 MinerU 输出存在命名冲突，已保留 {mineru_output.name}")
                    report(f"[ingest] [{ordinal}/{total}] MinerU 输出重命名冲突，已保留 {mineru_output.name}")
                else:
                    if renamed_output != mineru_output:
                        summary.renamed_mineru_output.append(f"{mineru_output.name} -> {renamed_output.name}")
                        report(f"[ingest] [{ordinal}/{total}] 已重命名 MinerU 输出 -> {renamed_output.name}")
                    mineru_output = renamed_output
            hash8 = file_hash[:8]
            paper_key = f"{slugify_title(title)}_{hash8}"
            paper_data_dir = settings.paper_data_dir / paper_key
            report(f"[ingest] [{ordinal}/{total}] 正在构建 paper_data：{paper_data_dir.name}")
            # extract 阶段只消费 MinerU 原始输出和规范 metadata，不再访问外部服务。
            result = extract_paper_data(
                mineru_output,
                paper_data_dir,
                {
                    "title": title,
                    "author": authors,
                    "year": year,
                    "venue": venue,
                    "pdf_path": str(target_pdf),
                },
                chunk_target_chars=settings.chunk_target_chars,
                chunk_overlap_chars=settings.chunk_overlap_chars,
            )
            record.status = "active" if authors and effective_year(year) else "metadata_unresolved"
            # manifest 只保存当前可重建所需的稳定路径和规范化 metadata。
            record.pdf_path = str(target_pdf)
            record.title = result.title
            record.author = authors
            record.year = year
            record.venue = venue
            record.mineru_output_path = str(mineru_output)
            record.archived_mineru_output_path = None
            record.paper_data_path = str(result.paper_data_dir)
            record.message = "; ".join(result.warnings) if result.warnings else None
            upsert_paper_annotation(annotations, file_hash, result.title)
            manifest.records[record.file_hash] = record
            for warning in result.warnings:
                report(f"[ingest] [{ordinal}/{total}] 警告：{warning}")
            summary.processed.append(
                f"{result.title}（{result.block_count} 个 block，{result.reference_count} 条引用，{result.chunk_count} 个 chunk）"
            )
            report(
                f"[ingest] [{ordinal}/{total}] 完成："
                f"{result.block_count} 个 block，{result.reference_count} 条引用，{result.chunk_count} 个 chunk"
            )
        except Exception as exc:
            summary.errors.append(f"{pdf_path.name}: {exc}")
            record.status = "error"
            record.pdf_path = str(pdf_path)
            record.message = str(exc)
            manifest.records[record.file_hash] = record
            report(f"[ingest] [{ordinal}/{total}] 错误：{exc}")

    report(f"[ingest] 正在保存 manifest：{settings.manifest_path}")
    manifest.save()
    save_paper_annotations(settings, annotations)
    try:
        # citation graph 是派生索引，放在所有论文都同步完成后统一构建。
        graph_result = build_citation_graph(settings, manifest)
        report(
            "[ingest] 引用图："
            f"{graph_result.node_count} 个节点，{graph_result.edge_count} 条边 -> {graph_result.path}"
        )
    except Exception as exc:
        summary.errors.append(f"引用图构建失败：{exc}")
        report(f"[ingest] 引用图构建失败：{exc}")
    report("[ingest] 完成")
    return summary


def lookup_metadata(
    title: str,
    dblp: DblpClient,
    semantic_scholar: SemanticScholarClient,
    arxiv: ArxivClient,
    dblp_candidate_limit: int = 20,
    dblp_retry_delay_seconds: float = 1.0,
    semantic_scholar_retry_delay_seconds: float = 1.0,
    arxiv_retry_delay_seconds: float = 1.0,
    report: Reporter | None = None,
    last_lookup_at: dict[str, float] | None = None,
) -> MetadataMatch | None:
    """按 ArXiv -> DBLP -> Semantic Scholar 的顺序补全论文元数据。"""
    emit = report or (lambda _: None)
    preprint_year: int | None = None

    def wait_for_lookup(source_key: str, label: str, delay_seconds: float) -> None:
        if last_lookup_at is None:
            return
        previous = last_lookup_at.get(source_key, 0.0)
        elapsed = time.monotonic() - previous
        if previous and elapsed < delay_seconds:
            wait_seconds = delay_seconds - elapsed
            emit(f"等待 {wait_seconds:.1f}s 后再查询 {label}")
            time.sleep(wait_seconds)

    def mark_lookup(source_key: str) -> None:
        if last_lookup_at is not None:
            last_lookup_at[source_key] = time.monotonic()

    emit("正在查询 ArXiv")
    wait_for_lookup("arxiv", "ArXiv", arxiv_retry_delay_seconds)
    try:
        arxiv_match = arxiv.lookup_exact_title(title, retry_delay_seconds=arxiv_retry_delay_seconds)
    except Exception as exc:
        arxiv_match = None
        emit(f"ArXiv 查询失败：{exc}")
    finally:
        mark_lookup("arxiv")
    if arxiv_match:
        preprint_year = arxiv_match.preprint_year
        emit(f"ArXiv 命中预印本年份：{preprint_year}")
    else:
        emit("ArXiv 未找到精确标题匹配")

    # DBLP 通常能给出更稳定的正式会议/期刊信息，所以优先于 Semantic Scholar。
    emit("正在查询 DBLP")
    wait_for_lookup("dblp", "DBLP", dblp_retry_delay_seconds)
    try:
        dblp_match = dblp.lookup_exact_title(
            title,
            limit=dblp_candidate_limit,
            retry_delay_seconds=dblp_retry_delay_seconds,
        )
    except Exception as exc:
        dblp_match = None
        emit(f"DBLP 查询失败：{exc}")
    finally:
        mark_lookup("dblp")
    if dblp_match and is_formal_venue(dblp_match.venue):
        # ArXiv 提供预印本年份，正式发表信息优先来自 DBLP。
        return MetadataMatch(
            title=dblp_match.title,
            authors=clean_author_list(dblp_match.authors),
            year={"preprint_year": preprint_year, "publish_year": formal_publish_year(dblp_match.year, dblp_match.venue)},
            venue=dblp_match.venue,
        )
    if dblp_match:
        emit(f"DBLP 命中非正式 venue，已忽略：{dblp_match.venue}")

    emit("DBLP 未找到精确标题匹配，继续查询 Semantic Scholar")
    wait_for_lookup("semantic_scholar", "Semantic Scholar", semantic_scholar_retry_delay_seconds)
    try:
        semantic_scholar_match = semantic_scholar.lookup_exact_title(
            title,
            retry_delay_seconds=semantic_scholar_retry_delay_seconds,
        )
    except Exception as exc:
        semantic_scholar_match = None
        emit(f"Semantic Scholar 查询失败：{exc}")
    finally:
        mark_lookup("semantic_scholar")
    if semantic_scholar_match and is_formal_venue(semantic_scholar_match.venue):
        # DBLP 没有正式 venue 时，再用 Semantic Scholar 的正式发表信息兜底。
        return MetadataMatch(
            title=semantic_scholar_match.title,
            authors=clean_author_list(semantic_scholar_match.authors),
            year={
                "preprint_year": preprint_year,
                "publish_year": formal_publish_year(semantic_scholar_match.year, semantic_scholar_match.venue),
            },
            venue=semantic_scholar_match.venue,
        )
    if semantic_scholar_match:
        emit(f"Semantic Scholar 命中非正式 venue，已忽略：{semantic_scholar_match.venue}")

    if arxiv_match:
        # 没有正式发表命中时，仍保留 ArXiv 的 title/authors/preprint_year。
        return MetadataMatch(
            title=arxiv_match.title,
            authors=clean_author_list(arxiv_match.authors),
            year={"preprint_year": preprint_year, "publish_year": None},
            venue=arxiv_match.venue,
        )

    emit("未找到正式发表元数据的精确标题匹配")
    return None


def is_formal_venue(venue: str | None) -> bool:
    normalized = normalize_text(str(venue or ""))
    return bool(normalized) and normalized not in {"arxiv", "corr"}


def formal_publish_year(source_year: int, venue: str | None) -> int:
    venue_years = re.findall(r"\b(?:19|20)\d{2}\b", str(venue or ""))
    if venue_years:
        return int(venue_years[0])
    return source_year


def scan_pdfs(pdf_dir: Path, summary: IngestSummary) -> dict[str, Path]:
    pdf_by_hash: dict[str, Path] = {}
    for path in sorted(pdf_dir.glob("*.pdf"), key=lambda p: p.name.lower()):
        file_hash = sha256_file(path)
        if file_hash in pdf_by_hash:
            summary.duplicates.append(f"{path.name} 与 {pdf_by_hash[file_hash].name} 重复")
            continue
        pdf_by_hash[file_hash] = path
    return pdf_by_hash


def archive_deleted_record(settings: Settings, record: ManifestRecord) -> None:
    """删除本地 PDF 后，移除派生 paper_data 并归档可复用的 MinerU 输出。"""
    if record.paper_data_path:
        shutil.rmtree(record.paper_data_path, ignore_errors=True)
        record.paper_data_path = None
    if record.mineru_output_path:
        src = Path(record.mineru_output_path)
        if src.exists():
            dst = settings.archive_dir / src.name
            archived = safe_move_dir(src, dst)
            record.archived_mineru_output_path = str(archived)
            record.mineru_output_path = None


def build_existing_output_index(mineru_output_dir: Path) -> MinerUOutputIndex:
    """按目录名和解析出的标题建立 MinerU 输出复用索引。"""
    index = MinerUOutputIndex()
    for directory in sorted(mineru_output_dir.iterdir() if mineru_output_dir.exists() else []):
        if not directory.is_dir():
            continue
        try:
            find_content_list_v2_path(directory)
        except FileNotFoundError:
            continue
        sidecar = load_mineru_source_sidecar(directory)
        file_hash = str(sidecar.get("file_hash") or "").strip() if sidecar else ""
        if file_hash:
            index.by_source_hash.setdefault(file_hash, []).append(directory)
        try:
            title = title_from_output(directory)
        except Exception:
            title = None
        if title:
            index.titles[directory] = title
            index.by_title.setdefault(normalize_text(title), []).append(directory)
    return index


def title_from_output(mineru_output_dir: Path) -> str | None:
    """只从 MinerU 输出中取标题，避免文件名推断覆盖解析出的真实标题。"""
    pages = load_content_list_v2(find_content_list_v2_path(mineru_output_dir))
    return extract_title(flatten_pages(pages))


def ensure_mineru_output(
    settings: Settings,
    record: ManifestRecord,
    pdf_path: Path,
    file_hash: str,
    output_index: MinerUOutputIndex,
    summary: IngestSummary,
    report: Reporter | None = None,
) -> Path:
    """优先复用 manifest/归档/现有输出，最后才调用 MinerU API。"""
    emit = report or (lambda _: None)
    if record.mineru_output_path and Path(record.mineru_output_path).exists():
        summary.reused.append(pdf_path.name)
        emit("[ingest] 正在复用 manifest 中的 MinerU 输出")
        output_dir = Path(record.mineru_output_path)
        write_mineru_source_sidecar(output_dir, file_hash, pdf_path)
        return output_dir
    if record.archived_mineru_output_path and Path(record.archived_mineru_output_path).exists():
        src = Path(record.archived_mineru_output_path)
        dst = settings.mineru_output_dir / src.name
        replace_dir(src, dst)
        summary.restored.append(pdf_path.name)
        emit("[ingest] 已从归档恢复 MinerU 输出")
        write_mineru_source_sidecar(dst, file_hash, pdf_path)
        return dst
    inferred_title = infer_title_from_pdf_name(pdf_path)
    hash_matches = output_index.by_source_hash.get(file_hash) or []
    if hash_matches:
        output_dir = unique_mineru_match(hash_matches, f"file_hash={file_hash}")
        summary.reused.append(pdf_path.name)
        emit("[ingest] 通过 source sidecar 精确复用 MinerU 输出")
        return output_dir
    title_matches = exact_title_matches(output_index, pdf_path)
    if title_matches:
        output_dir = unique_mineru_match(title_matches, inferred_title)
        summary.reused.append(pdf_path.name)
        emit("[ingest] 通过唯一标题精确匹配复用已有 MinerU 输出")
        write_mineru_source_sidecar(output_dir, file_hash, pdf_path)
        return output_dir
    fuzzy_matches = fuzzy_title_matches(output_index, pdf_path)
    if fuzzy_matches:
        names = ", ".join(path.name for path in fuzzy_matches[:3])
        raise MinerUError(f"找到可能匹配但不精确的 MinerU 输出，已拒绝自动复用：{names}")
    if not settings.mineru_api_key:
        raise MinerUError(".env 中缺少 MINERU_API_KEY，且没有找到可复用的 MinerU 输出")
    title_slug = slugify_title(inferred_title)
    output_dir = settings.mineru_output_dir / f"{title_slug}_{file_hash[:8]}"
    client = MinerUClient(
        settings.mineru_api_key,
        settings.mineru_api_base_url,
        settings.mineru_model_version,
        settings.mineru_language,
    )
    emit("[ingest] 正在调用 MinerU API")
    parsed_output = client.parse_local_pdf(pdf_path, output_dir, file_hash)
    write_mineru_source_sidecar(parsed_output, file_hash, pdf_path)
    return parsed_output


def unique_mineru_match(matches: list[Path], label: str) -> Path:
    unique = sorted({path.resolve(): path for path in matches}.values(), key=lambda path: path.name.lower())
    if len(unique) != 1:
        names = ", ".join(path.name for path in unique[:5])
        raise MinerUError(f"MinerU 输出匹配不唯一：{label} -> {names}")
    return unique[0]


def exact_title_matches(index: MinerUOutputIndex, pdf_path: Path) -> list[Path]:
    matches: list[Path] = []
    for key in inferred_title_keys(pdf_path):
        matches.extend(index.by_title.get(key) or [])
    return dedupe_paths(matches)


def fuzzy_title_matches(index: MinerUOutputIndex, pdf_path: Path) -> list[Path]:
    keys = inferred_title_keys(pdf_path)
    matches: list[Path] = []
    for directory, title in index.titles.items():
        title_key = normalize_text(title)
        if any(key and title_key and key != title_key and (key in title_key or title_key in key) for key in keys):
            matches.append(directory)
    return dedupe_paths(matches)


def inferred_title_keys(pdf_path: Path) -> list[str]:
    title = infer_title_from_pdf_name(pdf_path)
    candidates = [title, strip_leading_year_from_title(title)]
    return [
        key
        for index, key in enumerate(normalize_text(value) for value in candidates)
        if key and key not in list(normalize_text(item) for item in candidates[:index])
    ]


def strip_leading_year_from_title(title: str) -> str:
    return re.sub(r"^\s*(?:19|20)\d{2}[\s_\-.]+", "", title).strip()


def dedupe_paths(paths: list[Path]) -> list[Path]:
    output: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        output.append(path)
    return output


def load_mineru_source_sidecar(output_dir: Path) -> dict[str, Any] | None:
    path = output_dir / MINERU_SOURCE_SIDECAR
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def write_mineru_source_sidecar(output_dir: Path, file_hash: str, pdf_path: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "file_hash": file_hash,
        "pdf_name": pdf_path.name,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    (output_dir / MINERU_SOURCE_SIDECAR).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def rename_pdf_if_needed(pdf_dir: Path, pdf_path: Path, year: int, title: str, file_hash: str) -> Path | None:
    """把 PDF 重命名为 year_title.pdf，遇到不同文件同名时返回 None。"""
    target = pdf_dir / f"{year}_{slugify_title(title)}.pdf"
    if pdf_path.resolve() == target.resolve():
        return pdf_path
    if target.exists():
        if sha256_file(target) == file_hash:
            if pdf_path.exists() and pdf_path.resolve() != target.resolve():
                pdf_path.unlink()
            return target
        return None
    pdf_path.rename(target)
    return target


def rename_mineru_output_if_needed(mineru_output_dir: Path, mineru_output: Path, year: int, title: str) -> Path | None:
    """让 MinerU 输出目录跟 PDF 命名保持一致，方便人工排查。"""
    target = mineru_output_dir / f"{year}_{slugify_title(title)}"
    if mineru_output.resolve() == target.resolve():
        return mineru_output
    if target.exists():
        return None
    mineru_output.rename(target)
    return target
