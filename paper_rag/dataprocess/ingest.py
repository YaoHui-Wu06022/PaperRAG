from __future__ import annotations

import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..config import Settings
from ..utils import infer_title_from_pdf_name, normalize_text, replace_dir, safe_move_dir, sha256_file, slugify_title
from .annotations import load_paper_annotations, save_paper_annotations, upsert_paper_annotation
from .citation_graph import build_citation_graph
from .extract import extract_paper_data, extract_title, flatten_pages, load_content_list_v2
from .manifest import Manifest, ManifestRecord, effective_year, normalize_year
from .mineru import MinerUClient, MinerUError
from .metadata.arxiv import ArxivClient
from .metadata.dblp import DblpClient
from .metadata.semantic_scholar import SemanticScholarClient


@dataclass(frozen=True)
class MetadataMatch:
    title: str
    authors: list[str]
    year: dict[str, int | None]
    venue: str | None
    source: str


@dataclass
class IngestSummary:
    processed: list[str] = field(default_factory=list)
    reused: list[str] = field(default_factory=list)
    restored: list[str] = field(default_factory=list)
    renamed_mineru_output: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def has_errors(self) -> bool:
        return bool(self.errors)


Reporter = Callable[[str], None]


def noop_reporter(_: str) -> None:
    return None


def clean_author_name(name: str) -> str:
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
    report = reporter or noop_reporter
    report("[ingest] Preparing data directories")
    settings.pdf_dir.mkdir(parents=True, exist_ok=True)
    settings.mineru_output_dir.mkdir(parents=True, exist_ok=True)
    settings.paper_data_dir.mkdir(parents=True, exist_ok=True)
    settings.archive_dir.mkdir(parents=True, exist_ok=True)
    summary = IngestSummary()
    report(f"[ingest] Loading manifest: {settings.manifest_path}")
    manifest = Manifest.load(settings.manifest_path)
    annotations = load_paper_annotations(settings)
    report(f"[ingest] Scanning PDFs: {settings.pdf_dir}")
    pdf_by_hash = scan_pdfs(settings.pdf_dir, summary)
    report(f"[ingest] Found {len(pdf_by_hash)} unique PDF(s)")
    for duplicate in summary.duplicates:
        report(f"[ingest] Duplicate skipped: {duplicate}")
    active_hashes = set(pdf_by_hash)

    report("[ingest] Synchronizing deleted PDFs")
    for file_hash, record in list(manifest.records.items()):
        if record.status != "deleted" and record.pdf_path and file_hash not in active_hashes:
            report(f"[ingest] Archiving deleted PDF data: {record.title or file_hash[:8]}")
            archive_deleted_record(settings, record)
            record.status = "deleted"
            record.pdf_path = None
            summary.deleted.append(record.title or file_hash[:8])
            manifest.upsert(record)

    report(f"[ingest] Indexing existing MinerU output: {settings.mineru_output_dir}")
    output_index = build_existing_output_index(settings.mineru_output_dir)
    report(f"[ingest] Indexed {len(output_index)} MinerU lookup key(s)")
    dblp = DblpClient()
    semantic_scholar = SemanticScholarClient(settings.semantic_scholar_api_key)
    arxiv = ArxivClient()
    last_dblp_request = 0.0
    last_semantic_scholar_request = 0.0
    last_arxiv_request = 0.0
    total = len(pdf_by_hash)
    for ordinal, (file_hash, pdf_path) in enumerate(pdf_by_hash.items(), start=1):
        report(f"[ingest] [{ordinal}/{total}] Processing {pdf_path.name}")
        record = manifest.get(file_hash) or ManifestRecord(file_hash=file_hash, status="new")
        try:
            mineru_output = ensure_mineru_output(settings, record, pdf_path, file_hash, output_index, summary, report)
            report(f"[ingest] [{ordinal}/{total}] MinerU output: {mineru_output.name}")
            title = title_from_output(mineru_output)
            if not title:
                record.status = "title_unresolved"
                record.pdf_path = str(pdf_path)
                record.message = "No paper title found"
                summary.unresolved.append(f"{pdf_path.name}: no title")
                manifest.upsert(record)
                report(f"[ingest] [{ordinal}/{total}] Title unresolved; skipped paper_data generation")
                continue
            authors: list[str] = [] if refresh_metadata else clean_author_list(record.author or [])
            year = normalize_year(None if refresh_metadata else record.year)
            venue = None if refresh_metadata else record.venue
            if venue and not is_formal_venue(venue):
                venue = None
            has_existing_metadata = bool(record.title and authors and effective_year(year))
            if not refresh_metadata and has_existing_metadata:
                title = record.title or title
            report(f"[ingest] [{ordinal}/{total}] Title: {title}")
            if not (authors and effective_year(year)):
                def wait_before_semantic_scholar_lookup() -> None:
                    nonlocal last_semantic_scholar_request
                    elapsed = time.monotonic() - last_semantic_scholar_request
                    if last_semantic_scholar_request and elapsed < settings.semantic_scholar_delay_seconds:
                        wait_seconds = settings.semantic_scholar_delay_seconds - elapsed
                        report(f"[ingest] [{ordinal}/{total}] Waiting {wait_seconds:.1f}s before Semantic Scholar lookup")
                        time.sleep(wait_seconds)

                def mark_semantic_scholar_lookup() -> None:
                    nonlocal last_semantic_scholar_request
                    last_semantic_scholar_request = time.monotonic()

                def wait_before_arxiv_lookup() -> None:
                    nonlocal last_arxiv_request
                    elapsed = time.monotonic() - last_arxiv_request
                    if last_arxiv_request and elapsed < settings.arxiv_delay_seconds:
                        wait_seconds = settings.arxiv_delay_seconds - elapsed
                        report(f"[ingest] [{ordinal}/{total}] Waiting {wait_seconds:.1f}s before ArXiv lookup")
                        time.sleep(wait_seconds)

                def mark_arxiv_lookup() -> None:
                    nonlocal last_arxiv_request
                    last_arxiv_request = time.monotonic()

                def wait_before_dblp_lookup() -> None:
                    nonlocal last_dblp_request
                    elapsed = time.monotonic() - last_dblp_request
                    if last_dblp_request and elapsed < settings.dblp_delay_seconds:
                        wait_seconds = settings.dblp_delay_seconds - elapsed
                        report(f"[ingest] [{ordinal}/{total}] Waiting {wait_seconds:.1f}s before DBLP lookup")
                        time.sleep(wait_seconds)

                def mark_dblp_lookup() -> None:
                    nonlocal last_dblp_request
                    last_dblp_request = time.monotonic()

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
                    before_dblp_lookup=wait_before_dblp_lookup,
                    after_dblp_lookup=mark_dblp_lookup,
                    before_semantic_scholar_lookup=wait_before_semantic_scholar_lookup,
                    after_semantic_scholar_lookup=mark_semantic_scholar_lookup,
                    before_arxiv_lookup=wait_before_arxiv_lookup,
                    after_arxiv_lookup=mark_arxiv_lookup,
                )
                if match:
                    title = match.title
                    authors = clean_author_list(match.authors)
                    year = match.year
                    venue = match.venue
                    report(f"[ingest] [{ordinal}/{total}] {match.source} matched: {format_year_for_log(year)}, venue={venue or 'unresolved'}")
                else:
                    summary.unresolved.append(f"{pdf_path.name}: ArXiv/DBLP/Semantic Scholar exact title not found")
                    report(f"[ingest] [{ordinal}/{total}] Metadata unresolved after ArXiv, DBLP, and Semantic Scholar")
            else:
                report(f"[ingest] [{ordinal}/{total}] Metadata already has authors/effective year; skipping external lookup")
            target_pdf = pdf_path
            rename_year = effective_year(year)
            if rename_year:
                try:
                    renamed = rename_pdf_if_needed(settings.pdf_dir, pdf_path, rename_year, title, file_hash)
                except OSError as exc:
                    renamed = pdf_path
                    summary.errors.append(f"Could not rename {pdf_path.name}; kept original name: {exc}")
                    report(f"[ingest] [{ordinal}/{total}] PDF rename failed; kept original name: {exc}")
                if renamed is None:
                    summary.errors.append(f"Name conflict for {pdf_path.name}; skipped rename")
                    report(f"[ingest] [{ordinal}/{total}] PDF rename conflict; kept original name")
                else:
                    target_pdf = renamed
                    if renamed != pdf_path:
                        report(f"[ingest] [{ordinal}/{total}] Renamed PDF -> {renamed.name}")
                renamed_output = rename_mineru_output_if_needed(
                    settings.mineru_output_dir,
                    mineru_output,
                    rename_year,
                    title,
                )
                if renamed_output is None:
                    summary.errors.append(f"MinerU output name conflict for {title}; kept {mineru_output.name}")
                    report(f"[ingest] [{ordinal}/{total}] MinerU output rename conflict; kept {mineru_output.name}")
                else:
                    if renamed_output != mineru_output:
                        summary.renamed_mineru_output.append(f"{mineru_output.name} -> {renamed_output.name}")
                        report(f"[ingest] [{ordinal}/{total}] Renamed MinerU output -> {renamed_output.name}")
                    mineru_output = renamed_output
            hash8 = file_hash[:8]
            paper_key = f"{slugify_title(title)}_{hash8}"
            paper_data_dir = settings.paper_data_dir / paper_key
            report(f"[ingest] [{ordinal}/{total}] Building paper_data: {paper_data_dir.name}")
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
            manifest.upsert(record)
            for warning in result.warnings:
                report(f"[ingest] [{ordinal}/{total}] Warning: {warning}")
            summary.processed.append(
                f"{result.title} ({result.block_count} blocks, {result.reference_count} refs, {result.chunk_count} chunks)"
            )
            report(
                f"[ingest] [{ordinal}/{total}] Done: "
                f"{result.block_count} block(s), {result.reference_count} reference(s), {result.chunk_count} chunk(s)"
            )
        except Exception as exc:
            summary.errors.append(f"{pdf_path.name}: {exc}")
            record.status = "error"
            record.pdf_path = str(pdf_path)
            record.message = str(exc)
            manifest.upsert(record)
            report(f"[ingest] [{ordinal}/{total}] ERROR: {exc}")

    report(f"[ingest] Saving manifest: {settings.manifest_path}")
    manifest.save()
    save_paper_annotations(settings, annotations)
    try:
        graph_result = build_citation_graph(settings, manifest)
        report(
            "[ingest] Citation graph: "
            f"{graph_result.node_count} node(s), {graph_result.edge_count} edge(s) -> {graph_result.path}"
        )
    except Exception as exc:
        summary.errors.append(f"Citation graph build failed: {exc}")
        report(f"[ingest] Citation graph build failed: {exc}")
    report("[ingest] Complete")
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
    report: Reporter = noop_reporter,
    before_dblp_lookup: Callable[[], None] | None = None,
    after_dblp_lookup: Callable[[], None] | None = None,
    before_semantic_scholar_lookup: Callable[[], None] | None = None,
    after_semantic_scholar_lookup: Callable[[], None] | None = None,
    before_arxiv_lookup: Callable[[], None] | None = None,
    after_arxiv_lookup: Callable[[], None] | None = None,
) -> MetadataMatch | None:
    arxiv_title: str | None = None
    arxiv_authors: list[str] = []
    preprint_year: int | None = None

    report("Querying ArXiv")
    if before_arxiv_lookup:
        before_arxiv_lookup()
    try:
        arxiv_match = arxiv.lookup_exact_title(title, retry_delay_seconds=arxiv_retry_delay_seconds)
    except Exception as exc:
        arxiv_match = None
        report(f"ArXiv lookup failed: {exc}")
    finally:
        if after_arxiv_lookup:
            after_arxiv_lookup()
    if arxiv_match:
        arxiv_title = arxiv_match.title
        arxiv_authors = clean_author_list(arxiv_match.authors)
        preprint_year = arxiv_match.preprint_year
        report(f"ArXiv matched preprint year: {preprint_year}")
    else:
        report("ArXiv exact title not found")

    report("Querying DBLP")
    if before_dblp_lookup:
        before_dblp_lookup()
    try:
        dblp_match = dblp.lookup_exact_title(
            title,
            limit=dblp_candidate_limit,
            retry_delay_seconds=dblp_retry_delay_seconds,
        )
    except Exception as exc:
        dblp_match = None
        report(f"DBLP lookup failed: {exc}")
    finally:
        if after_dblp_lookup:
            after_dblp_lookup()
    if dblp_match and is_formal_venue(dblp_match.venue):
        return MetadataMatch(
            title=dblp_match.title,
            authors=clean_author_list(dblp_match.authors),
            year={"preprint_year": preprint_year, "publish_year": formal_publish_year(dblp_match.year, dblp_match.venue)},
            venue=dblp_match.venue,
            source="ArXiv+DBLP" if preprint_year else "DBLP",
        )
    if dblp_match:
        report(f"DBLP matched non-formal venue; ignored: {dblp_match.venue}")

    report("DBLP exact title not found; querying Semantic Scholar")
    if before_semantic_scholar_lookup:
        before_semantic_scholar_lookup()
    try:
        semantic_scholar_match = semantic_scholar.lookup_exact_title(
            title,
            retry_delay_seconds=semantic_scholar_retry_delay_seconds,
        )
    except Exception as exc:
        semantic_scholar_match = None
        report(f"Semantic Scholar lookup failed: {exc}")
    finally:
        if after_semantic_scholar_lookup:
            after_semantic_scholar_lookup()
    if semantic_scholar_match and is_formal_venue(semantic_scholar_match.venue):
        return MetadataMatch(
            title=semantic_scholar_match.title,
            authors=clean_author_list(semantic_scholar_match.authors),
            year={
                "preprint_year": preprint_year,
                "publish_year": formal_publish_year(semantic_scholar_match.year, semantic_scholar_match.venue),
            },
            venue=semantic_scholar_match.venue,
            source="ArXiv+Semantic Scholar" if preprint_year else "Semantic Scholar",
        )
    if semantic_scholar_match:
        report(f"Semantic Scholar matched non-formal venue; ignored: {semantic_scholar_match.venue}")

    if arxiv_match:
        return MetadataMatch(
            title=arxiv_title or title,
            authors=clean_author_list(arxiv_authors),
            year={"preprint_year": preprint_year, "publish_year": None},
            venue=None,
            source="ArXiv",
        )

    report("Formal metadata exact title not found")
    return None


def is_formal_venue(venue: str | None) -> bool:
    normalized = normalize_text(str(venue or ""))
    return bool(normalized) and normalized not in {"arxiv", "corr"}


def formal_publish_year(source_year: int, venue: str | None) -> int:
    venue_years = re.findall(r"\b(?:19|20)\d{2}\b", str(venue or ""))
    if venue_years:
        return int(venue_years[0])
    return source_year


def format_year_for_log(year: dict[str, int | None]) -> str:
    return f"preprint={year.get('preprint_year')}, publish={year.get('publish_year')}"


def scan_pdfs(pdf_dir: Path, summary: IngestSummary) -> dict[str, Path]:
    pdf_by_hash: dict[str, Path] = {}
    for path in sorted(pdf_dir.glob("*.pdf"), key=lambda p: p.name.lower()):
        file_hash = sha256_file(path)
        if file_hash in pdf_by_hash:
            summary.duplicates.append(f"{path.name} duplicates {pdf_by_hash[file_hash].name}")
            continue
        pdf_by_hash[file_hash] = path
    return pdf_by_hash


def archive_deleted_record(settings: Settings, record: ManifestRecord) -> None:
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


def build_existing_output_index(mineru_output_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for directory in sorted(mineru_output_dir.iterdir() if mineru_output_dir.exists() else []):
        if not directory.is_dir() or not (directory / "content_list_v2.json").exists():
            continue
        index[normalize_text(directory.name)] = directory
        try:
            title = title_from_output(directory)
        except Exception:
            title = None
        if title:
            index[normalize_text(title)] = directory
            index[normalize_text(slugify_title(title))] = directory
    return index


def title_from_output(mineru_output_dir: Path) -> str | None:
    pages = load_content_list_v2(mineru_output_dir / "content_list_v2.json")
    return extract_title(flatten_pages(pages))


def ensure_mineru_output(
    settings: Settings,
    record: ManifestRecord,
    pdf_path: Path,
    file_hash: str,
    output_index: dict[str, Path],
    summary: IngestSummary,
    report: Reporter = noop_reporter,
) -> Path:
    if record.mineru_output_path and Path(record.mineru_output_path).exists():
        summary.reused.append(pdf_path.name)
        report("[ingest] Reusing MinerU output from manifest")
        return Path(record.mineru_output_path)
    if record.archived_mineru_output_path and Path(record.archived_mineru_output_path).exists():
        src = Path(record.archived_mineru_output_path)
        dst = settings.mineru_output_dir / src.name
        replace_dir(src, dst)
        summary.restored.append(pdf_path.name)
        report("[ingest] Restored MinerU output from archive")
        return dst
    inferred_title = infer_title_from_pdf_name(pdf_path)
    inferred_norm = normalize_text(inferred_title)
    for key, directory in output_index.items():
        if inferred_norm and (inferred_norm == key or inferred_norm in key or key in inferred_norm):
            summary.reused.append(pdf_path.name)
            report("[ingest] Reusing existing MinerU output by title match")
            return directory
    if not settings.mineru_api_key:
        raise MinerUError("MINERU_API_KEY is missing in .env and no reusable MinerU output was found")
    title_slug = slugify_title(inferred_title)
    output_dir = settings.mineru_output_dir / f"{title_slug}_{file_hash[:8]}"
    client = MinerUClient(
        settings.mineru_api_key,
        settings.mineru_api_base_url,
        settings.mineru_model_version,
        settings.mineru_language,
    )
    report("[ingest] Calling MinerU API")
    return client.parse_local_pdf(pdf_path, output_dir, file_hash)


def rename_pdf_if_needed(pdf_dir: Path, pdf_path: Path, year: int, title: str, file_hash: str) -> Path | None:
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
    target = mineru_output_dir / f"{year}_{slugify_title(title)}"
    if mineru_output.resolve() == target.resolve():
        return mineru_output
    if target.exists():
        return None
    mineru_output.rename(target)
    return target
