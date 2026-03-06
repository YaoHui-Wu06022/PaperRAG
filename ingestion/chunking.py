from __future__ import annotations

import re
from typing import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _is_heading(text: str) -> bool:
    line = text.strip()
    if not line:
        return False
    lower = line.lower()
    patterns = [
        r"^(chapter|section|appendix)\b",
        r"^\d+(\.\d+){0,4}\s+",
        r"^[ivxlcdm]+[\.\)]\s+",
        r"^第[一二三四五六七八九十百0-9]+[章节部分]\s*",
    ]
    if any(re.match(pattern, lower) for pattern in patterns):
        return True
    return line.isupper() and len(line) <= 80


def _is_list_item(text: str) -> bool:
    line = text.strip()
    if not line:
        return False
    patterns = [
        r"^[-*•]\s+",
        r"^\d+[\.\)]\s+",
        r"^[a-zA-Z][\.\)]\s+",
        r"^（\d+）",
        r"^\(\d+\)",
    ]
    return any(re.match(pattern, line) for pattern in patterns)


def _structure_split_text(text: str, min_block_chars: int) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    paragraphs = [item.strip() for item in re.split(r"\n{2,}", normalized) if item.strip()]
    if not paragraphs:
        return []

    blocks: list[str] = []
    current = ""
    for para in paragraphs:
        first_line = para.split("\n", 1)[0].strip()
        heading_like = _is_heading(first_line)
        list_like = _is_list_item(first_line)

        if heading_like:
            if current:
                blocks.append(current.strip())
            current = para
            continue

        if not current:
            current = para
            continue

        if len(current) < min_block_chars or list_like:
            sep = "\n" if list_like else "\n\n"
            current = f"{current}{sep}{para}"
        else:
            blocks.append(current.strip())
            current = para

    if current:
        blocks.append(current.strip())
    return blocks


def _build_token_splitter(
    chunk_size: int,
    chunk_overlap: int,
    tokenizer_model: str,
) -> RecursiveCharacterTextSplitter:
    separators = [
        "\n\n",
        "\n",
        "。 ",
        "。",
        "；",
        "，",
        ". ",
        "; ",
        ", ",
        " ",
        "",
    ]
    try:
        from transformers import AutoTokenizer

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_model,
                local_files_only=True,
            )
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
        )
    except Exception:
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
        )


def _iter_semantic_parent_docs(documents: list[Document]) -> Iterable[Document]:
    for doc_idx, doc in enumerate(documents):
        content = doc.page_content.strip()
        if not content:
            continue
        metadata = dict(doc.metadata or {})
        if "structure_block_id" not in metadata:
            metadata["structure_block_id"] = 0
        if "parent_id" not in metadata:
            source = str(metadata.get("source", "unknown"))
            page = str(metadata.get("page", "?"))
            metadata["parent_id"] = f"{source}::p{page}::d{doc_idx}::b0"
        metadata["chunk_strategy"] = "semantic_paper"
        yield Document(page_content=content, metadata=metadata)


def _build_parent_documents(
    documents: list[Document],
    *,
    use_structure_split: bool,
    min_block_chars: int,
) -> list[Document]:
    parent_docs: list[Document] = []
    for doc_idx, doc in enumerate(documents):
        content = doc.page_content.strip()
        if not content:
            continue

        if use_structure_split:
            blocks = _structure_split_text(content, min_block_chars=min_block_chars) or [content]
        else:
            blocks = [content]

        for block_id, block_text in enumerate(blocks):
            metadata = dict(doc.metadata or {})
            metadata["structure_block_id"] = block_id
            source = str(metadata.get("source", "unknown"))
            page = str(metadata.get("page", "?"))
            metadata["parent_id"] = f"{source}::p{page}::d{doc_idx}::b{block_id}"
            metadata["chunk_strategy"] = "token"
            parent_docs.append(Document(page_content=block_text, metadata=metadata))
    return parent_docs


def _split_semantic_parents(
    parent_docs: list[Document],
    *,
    splitter: RecursiveCharacterTextSplitter,
    semantic_hard_max_chars: int,
) -> list[Document]:
    chunks: list[Document] = []
    hard_limit = max(semantic_hard_max_chars, 800)

    for parent in parent_docs:
        content = parent.page_content.strip()
        if not content:
            continue
        if len(content) <= hard_limit:
            chunks.append(parent)
            continue

        # 极长语义块做兜底切分，避免超出生成上下文限制。
        sub_chunks = splitter.split_documents([parent])
        if not sub_chunks:
            chunks.append(parent)
            continue
        for sub_idx, sub in enumerate(sub_chunks):
            metadata = dict(sub.metadata or {})
            metadata["semantic_overflow_split"] = True
            metadata["semantic_child_id"] = sub_idx
            chunks.append(Document(page_content=sub.page_content, metadata=metadata))
    return chunks


def _assign_chunk_ids(chunks: list[Document]) -> list[Document]:
    finalized: list[Document] = []
    for idx, chunk in enumerate(chunks):
        metadata = dict(chunk.metadata or {})
        metadata["chunk_id"] = idx
        parent_id = str(metadata.get("parent_id", "")).strip()
        metadata["chunk_uid"] = f"{parent_id}::c{idx}" if parent_id else f"chunk::{idx}"
        finalized.append(Document(page_content=chunk.page_content, metadata=metadata))
    return finalized


def split_documents_with_parents(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
    tokenizer_model: str,
    *,
    use_structure_split: bool = True,
    min_block_chars: int = 120,
    chunk_strategy: str = "semantic_paper",
    semantic_hard_max_chars: int = 2400,
) -> tuple[list[Document], list[Document]]:
    strategy = chunk_strategy.strip().lower()
    if strategy not in {"semantic_paper", "token"}:
        strategy = "semantic_paper"

    splitter = _build_token_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        tokenizer_model=tokenizer_model,
    )

    if strategy == "semantic_paper":
        parent_docs = list(_iter_semantic_parent_docs(documents))
        chunks = _split_semantic_parents(
            parent_docs,
            splitter=splitter,
            semantic_hard_max_chars=semantic_hard_max_chars,
        )
        return _assign_chunk_ids(chunks), parent_docs

    parent_docs = _build_parent_documents(
        documents,
        use_structure_split=use_structure_split,
        min_block_chars=min_block_chars,
    )
    chunks = splitter.split_documents(parent_docs)
    return _assign_chunk_ids(chunks), parent_docs


def split_documents(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
    tokenizer_model: str,
    *,
    use_structure_split: bool = True,
    min_block_chars: int = 120,
    chunk_strategy: str = "semantic_paper",
    semantic_hard_max_chars: int = 2400,
) -> list[Document]:
    chunks, _ = split_documents_with_parents(
        documents=documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        tokenizer_model=tokenizer_model,
        use_structure_split=use_structure_split,
        min_block_chars=min_block_chars,
        chunk_strategy=chunk_strategy,
        semantic_hard_max_chars=semantic_hard_max_chars,
    )
    return chunks

