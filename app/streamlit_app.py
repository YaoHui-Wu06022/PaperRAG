from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import io
import re
import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from config import load_config
from ingestion.embedding import build_embedding_model
from pipeline import answer_question, ingest_documents
from services.health import build_startup_health_report
from services.local_cache_store import chunk_corpus_path, reference_chunk_corpus_path
from services.sync_transaction import (
    has_pending_sync_operation,
    recover_pending_sync_operation,
)


_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def _sanitize_uploaded_filename(filename: str) -> str:
    name = Path(str(filename or "")).name.strip()
    name = re.sub(r'[<>:"/\\\\|?*\x00-\x1f]', "_", name)
    name = re.sub(r"\s+", "_", name)
    name = name.lstrip(".")
    stem = Path(name).stem or "uploaded"
    suffix = Path(name).suffix.lower()
    if stem.upper() in _WINDOWS_RESERVED_NAMES:
        stem = f"upload_{stem}"
    if suffix != ".pdf":
        suffix = ".pdf"
    return f"{stem}{suffix}"


def _dedupe_upload_path(target_dir: Path, filename: str) -> Path:
    candidate = target_dir / filename
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix or ".pdf"
    for index in range(1, 10_000):
        alternative = target_dir / f"{stem}_{index}{suffix}"
        if not alternative.exists():
            return alternative
    raise RuntimeError("Too many files with the same uploaded name.")


def save_uploads(uploaded_files, target_dir: Path) -> int:
    target_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for file in uploaded_files:
        safe_name = _sanitize_uploaded_filename(file.name)
        file_path = _dedupe_upload_path(target_dir, safe_name)
        file_path.write_bytes(file.read())
        count += 1
    return count


@st.cache_resource(show_spinner=False)
def _task_executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag_streamlit")


@st.cache_data(ttl=30, show_spinner=False)
def _startup_health_snapshot() -> list[dict[str, object]]:
    config = load_config()
    report = build_startup_health_report(
        config,
        require_mineru=True,
        require_llm=True,
        require_local_cache=False,
        require_milvus=True,
    )
    return [item.__dict__ for item in report]


def _recover_pending_sync_if_needed(config) -> str | None:
    if not has_pending_sync_operation(config):
        return None
    embeddings = build_embedding_model(
        config.embedding_provider,
        config.embedding_model,
        openai_api_key=config.openai_api_key,
        openai_base_url=config.openai_base_url,
        aihubmix_api_key=config.aihubmix_api_key,
        aihubmix_base_url=config.aihubmix_base_url,
    )
    return recover_pending_sync_operation(config, embeddings)


def _init_task_state() -> None:
    st.session_state.setdefault("tasks", {})


def _submit_task(task_key: str, label: str, fn, *args, **kwargs) -> bool:
    task = st.session_state["tasks"].get(task_key)
    if task and task["status"] == "running":
        return False

    future = _task_executor().submit(fn, *args, **kwargs)
    st.session_state["tasks"][task_key] = {
        "label": label,
        "status": "running",
        "future": future,
        "submitted_at": time.time(),
        "completed_at": None,
        "result": None,
        "error": "",
    }
    return True


def _refresh_tasks() -> None:
    for task in st.session_state.get("tasks", {}).values():
        if task["status"] != "running":
            continue
        future: Future = task["future"]
        if not future.done():
            continue
        try:
            task["result"] = future.result()
            task["status"] = "succeeded"
        except Exception as exc:
            task["error"] = str(exc)
            task["status"] = "failed"
        task["completed_at"] = time.time()


def _clear_completed_tasks() -> None:
    tasks = st.session_state.get("tasks", {})
    completed = [
        task_key
        for task_key, task in tasks.items()
        if task["status"] in {"failed", "succeeded"}
    ]
    for task_key in completed:
        tasks.pop(task_key, None)


def _has_running_tasks() -> bool:
    return any(
        task.get("status") == "running"
        for task in st.session_state.get("tasks", {}).values()
    )


def _render_task_panel(config) -> None:
    tasks = st.session_state.get("tasks", {})
    if not tasks:
        return

    st.subheader("Tasks")
    controls = st.columns(2)
    if controls[0].button("Refresh Tasks"):
        st.rerun()
    if controls[1].button("Clear Completed"):
        _clear_completed_tasks()
        st.rerun()

    if _has_running_tasks():
        st.caption("Task panel auto-refreshes every 2 seconds while tasks are running.")

    for task_key, task in tasks.items():
        elapsed = time.time() - float(task["submitted_at"])
        if task["status"] == "running":
            st.info(f"{task['label']}: running ({elapsed:.1f}s)")
        elif task["status"] == "failed":
            st.error(f"{task['label']}: failed - {task['error']}")
        else:
            st.success(f"{task['label']}: completed")

        if task_key == "ingest" and task["status"] == "succeeded":
            result = task["result"]
            if result.skipped:
                if result.raw_documents == 0 and result.chunks == 0:
                    st.info("No new doc_id found. Use force rebuild to overwrite existing documents.")
                else:
                    st.info("Ingest cache hit. Index build skipped.")
            else:
                st.success(f"Indexed {result.raw_documents} pages into {result.chunks} chunks.")
            if result.reference_chunks or result.suspicious_reference_chunks:
                st.caption(
                    "Reference purity: "
                    f"{result.reference_chunks} reference chunks, "
                    f"{result.suspicious_reference_chunks} suspicious."
                )

        if task_key == "ask" and task["status"] == "succeeded":
            _render_qa_result(task["result"], config)


def _render_qa_result(result, config) -> None:
    st.subheader("Answer")
    st.caption(f"Retrieval scope: {result.retrieval_scope}")
    st.write(result.answer)
    st.subheader("Sources")

    if result.evidences:
        for idx, evidence in enumerate(result.evidences, start=1):
            st.markdown(f"{idx}. {evidence.citation_text}")
            expander_key = f"{idx}_{evidence.doc_id}_{evidence.block_id}"
            with st.expander(f"Expand Evidence {idx}", expanded=False):
                st.markdown(f"**Citation**: `{evidence.citation_text}`")
                st.markdown(f"**Tag**: `{evidence.citation_tag}`")
                st.markdown("**Snippet**")
                st.write(evidence.snippet)
                st.markdown("**Trace Fields**")
                st.json(
                    {
                        "doc_id": evidence.doc_id,
                        "block_id": evidence.block_id,
                        "page": evidence.page,
                        "bbox": evidence.bbox,
                        "source": evidence.source,
                        "section_path": evidence.section_path,
                    }
                )

                target_pdf = config.data_pdf_dir / evidence.source
                if target_pdf.exists():
                    show_page = st.checkbox(
                        "Show PDF page and highlight bbox",
                        key=f"show_pdf_{expander_key}",
                    )
                    if show_page:
                        page_number = _safe_int(evidence.page, 0)
                        image_bytes, err = _render_pdf_page_with_bbox(
                            str(target_pdf),
                            page_number,
                            evidence.bbox,
                        )
                        if err:
                            st.warning(err)
                        elif image_bytes is not None:
                            st.image(
                                image_bytes,
                                caption=f"{evidence.source} - page {page_number}",
                                use_container_width=True,
                            )
                else:
                    st.caption(f"PDF file not found: {target_pdf}")
    else:
        for citation in result.citations:
            st.markdown(f"- `{citation}`")

    with st.expander("Retrieved Contexts"):
        for idx, context in enumerate(result.contexts, start=1):
            st.markdown(f"**Context {idx}**")
            st.write(context)


@st.fragment(run_every=2)
def _render_live_task_panel(config) -> None:
    _refresh_tasks()
    _render_task_panel(config)


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_bbox(bbox: list[float] | None) -> list[float]:
    if not isinstance(bbox, list) or len(bbox) < 4:
        return []
    normalized: list[float] = []
    for item in bbox[:4]:
        try:
            normalized.append(float(item))
        except (TypeError, ValueError):
            return []
    x0, y0, x1, y1 = normalized
    if x0 > x1:
        x0, x1 = x1, x0
    if y0 > y1:
        y0, y1 = y1, y0
    return [x0, y0, x1, y1]


def _map_bbox_to_pdf_space(
    bbox: list[float],
    *,
    page_width: float,
    page_height: float,
) -> tuple[float, float, float, float] | None:
    values = _normalize_bbox(bbox)
    if not values:
        return None
    x0, y0, x1, y1 = values

    if max(abs(v) for v in values) <= 1.5:
        x0 *= page_width
        x1 *= page_width
        y0 *= page_height
        y1 *= page_height
    else:
        max_x = max(x0, x1)
        max_y = max(y0, y1)
        if max_x > page_width * 1.2 or max_y > page_height * 1.2:
            sx = page_width / max(max_x, page_width)
            sy = page_height / max(max_y, page_height)
            x0 *= sx
            x1 *= sx
            y0 *= sy
            y1 *= sy

    x0 = max(0.0, min(x0, page_width))
    x1 = max(0.0, min(x1, page_width))
    y0 = max(0.0, min(y0, page_height))
    y1 = max(0.0, min(y1, page_height))
    if x1 - x0 <= 1e-3 or y1 - y0 <= 1e-3:
        return None
    return x0, y0, x1, y1


@st.cache_data(show_spinner=False)
def _render_pdf_page_with_bbox(
    pdf_path: str,
    page_number: int,
    bbox: list[float] | None,
) -> tuple[bytes | None, str | None]:
    if page_number <= 0:
        return None, "Invalid page number."

    try:
        import fitz
    except ImportError:
        return None, "PyMuPDF is not installed. Run `pip install pymupdf`."

    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None, "Pillow is not installed. Run `pip install pillow`."

    target = Path(pdf_path)
    if not target.exists():
        return None, f"PDF file not found: {target}"

    try:
        with fitz.open(str(target)) as doc:
            if page_number > doc.page_count:
                return None, f"Page number out of range: {doc.page_count}"
            page = doc.load_page(page_number - 1)
            page_rect = page.rect
            zoom = 2.0
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)

        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        draw = ImageDraw.Draw(image)
        mapped = _map_bbox_to_pdf_space(
            bbox or [],
            page_width=float(page_rect.width),
            page_height=float(page_rect.height),
        )
        if mapped:
            x0, y0, x1, y1 = mapped
            draw.rectangle(
                [x0 * zoom, y0 * zoom, x1 * zoom, y1 * zoom],
                outline=(255, 0, 0),
                width=max(2, int(2 * zoom)),
            )

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue(), None
    except Exception as exc:
        return None, f"Failed to render PDF page: {exc}"


def main() -> None:
    st.set_page_config(page_title="RAG PDF QA", page_icon="?", layout="wide")
    config = load_config()
    _init_task_state()
    recovered_status = _recover_pending_sync_if_needed(config)
    health_snapshot = _startup_health_snapshot()
    fatal_failures = [
        item for item in health_snapshot if item.get("fatal") and not item.get("ok")
    ]
    main_cache_ready = chunk_corpus_path(config).exists()
    reference_cache_ready = reference_chunk_corpus_path(config).exists()
    ingest_running = (
        st.session_state.get("tasks", {}).get("ingest", {}).get("status") == "running"
    )
    ask_running = (
        st.session_state.get("tasks", {}).get("ask", {}).get("status") == "running"
    )

    st.title("RAG-based PDF Knowledge Base QA")
    st.caption("Upload PDFs, queue indexing tasks, ask questions with citations.")
    if recovered_status is not None:
        st.warning(
            f"Recovered pending sync journal on startup (previous status: {recovered_status})."
        )
    if fatal_failures:
        st.error(
            "Startup health checks have fatal failures. Save/upload still works, "
            "but ingest or QA tasks may fail until the configuration is fixed."
        )

    with st.sidebar:
        st.subheader("Settings")
        st.write("PDF parser: `mineru_cloud`")
        st.write(f"MinerU API: `{config.mineru_api_base_url}`")
        st.write(f"MinerU cloud model: `{config.mineru_cloud_model_version}`")
        st.write(f"Embedding: `{config.embedding_provider}/{config.embedding_model}`")
        st.write(f"Chunk: `{config.chunk_size}` / Overlap: `{config.chunk_overlap}`")
        st.write(f"Chunk tokenizer: `{config.chunk_tokenizer_model}`")
        st.write(f"Structure split: `{config.chunk_use_structure_split}`")
        st.write(f"Min block chars: `{config.chunk_min_block_chars}`")
        st.write(f"Retriever top-k: `{config.retriever_top_k}`")
        st.write(f"Retrieval mode: `{config.retrieval_mode}`")
        st.write(f"Hybrid dense top-k: `{config.hybrid_dense_top_k}`")
        st.write(f"Hybrid BM25 top-k: `{config.hybrid_bm25_top_k}`")
        st.write(f"Hybrid RRF k: `{config.hybrid_rrf_k}`")
        st.write(f"Use reranker: `{config.use_reranker}`")
        st.write(f"Reranker: `{config.reranker_model}`")
        st.write(f"LLM: `{config.llm_provider}/{config.llm_model}`")
        st.divider()
        st.subheader("Health")
        for item in health_snapshot:
            prefix = "OK" if item["ok"] else "FAIL"
            st.caption(f"[{prefix}] {item['name']}: {item['message']}")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )
    force_rebuild = st.checkbox("Force rebuild index", value=False)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("1) Save Uploaded PDFs", disabled=ingest_running):
            if not uploaded_files:
                st.warning("Please upload at least one PDF.")
            else:
                count = save_uploads(uploaded_files, config.data_pdf_dir)
                st.success(f"Saved {count} PDF files to data/pdf.")

    with col2:
        if st.button("2) Build / Refresh Index", disabled=ingest_running):
            if _submit_task(
                "ingest",
                "Build / Refresh Index",
                ingest_documents,
                config,
                force=force_rebuild,
            ):
                st.info("Ingestion task submitted.")
                st.rerun()
            else:
                st.warning("An ingestion task is already running.")

    st.divider()
    question = st.text_input("Ask a question about your PDFs")
    ask_scope = st.radio(
        "Knowledge Scope",
        options=["main", "references"],
        format_func=lambda value: (
            "Main Content" if value == "main" else "References Only"
        ),
        horizontal=True,
    )
    ask_cache_ready = main_cache_ready if ask_scope == "main" else reference_cache_ready
    if not ask_cache_ready:
        if ask_scope == "references":
            st.caption(
                "Reference cache not found yet. Run ingestion before asking reference questions."
            )
        else:
            st.caption("Index cache not found yet. Run ingestion before asking questions.")
    if st.button("3) Ask", disabled=ask_running or not ask_cache_ready):
        if not question.strip():
            st.warning("Please input a question.")
        else:
            ask_label = (
                "Ask Question (References)"
                if ask_scope == "references"
                else "Ask Question"
            )
            if _submit_task(
                "ask",
                ask_label,
                answer_question,
                config,
                question.strip(),
                scope=ask_scope,
            ):
                st.info("QA task submitted.")
                st.rerun()
            else:
                st.warning("A QA task is already running.")

    _render_live_task_panel(config)


if __name__ == "__main__":
    main()
