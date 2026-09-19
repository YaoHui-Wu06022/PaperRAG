from __future__ import annotations

import logging
import re
from functools import lru_cache

from .common import digest, dumps

TOKENIZER_VERSION = "jieba-0.42.1-unicode-v1"
CHUNKER_VERSION = "cl100k-512-768-span-v2"


@lru_cache(maxsize=1)
def segmenter():
    import jieba

    jieba.setLogLevel(logging.ERROR)
    tokenizer = jieba.Tokenizer()
    tokenizer.initialize()
    return tokenizer


def lexical(text: str) -> str:
    terms = []
    for part in re.findall(
        r"[\u3400-\u9fff]+|[a-zA-Z0-9]+(?:[-_.][a-zA-Z0-9]+)*", text.casefold()
    ):
        terms.extend(
            segmenter().cut(part, HMM=False)
            if re.search(r"[\u3400-\u9fff]", part)
            else [part]
        )
    return " ".join(terms)


def match_expression(text: str) -> str:
    return " OR ".join(
        '"' + term.replace('"', '""') + '"'
        for term in dict.fromkeys(lexical(text).split())
    )


@lru_cache(maxsize=1)
def encoding():
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def token_count(text: str) -> int:
    return len(encoding().encode(text, disallowed_special=()))


def split_span(text: str, limit: int = 512):
    """Exact character spans; prefer row/sentence boundaries, never alter originals."""
    start = 0
    while start < len(text):
        if token_count(text[start:]) <= limit:
            yield start, len(text)
            break
        low, high = start + 1, min(len(text), start + limit * 8)
        while low < high:
            mid = (low + high + 1) // 2
            if token_count(text[start:mid]) <= limit:
                low = mid
            else:
                high = mid - 1
        end = low
        matches = list(re.finditer(r"\n|[.!?。！？](?:\s|$)", text[start:end]))
        if matches and matches[-1].end() >= (end - start) // 2:
            end = start + matches[-1].end()
        yield start, end
        start = end


def chunks_for(
    document_id: str,
    revision_id: str,
    title: str,
    blocks: list[dict],
    sections: list[dict],
):
    paths = {
        s["section_id"]: " > ".join(s.get("path") or [s["title"]]) for s in sections
    }
    current, current_text, key = [], [], None

    def build():
        text = "\n\n".join(current_text)
        section, region = key
        prefix = f"Paper: {title}\nSection: {paths.get(section, '')}\n\n"
        return {
            "chunk_id": "c_"
            + digest([document_id, revision_id, CHUNKER_VERSION, current]),
            "document_id": document_id,
            "revision_id": revision_id,
            "section_id": section,
            "region": region,
            "text": text,
            "embedding_text": prefix + text,
            "content_hash": digest(prefix + text),
            "spans": dumps(current),
        }

    for block in blocks:
        if block["region"] not in {"abstract", "body", "appendix"} or not block["text"]:
            continue
        next_key = (block.get("section_id"), block["region"])
        for start, end in split_span(block["text"]):
            part = block["text"][start:end]
            if current and (
                key != next_key or token_count("\n\n".join(current_text + [part])) > 512
            ):
                yield build()
                current, current_text = [], []
            key = next_key
            current.append({"block_id": block["block_id"], "start": start, "end": end})
            current_text.append(part)
    if current:
        yield build()
