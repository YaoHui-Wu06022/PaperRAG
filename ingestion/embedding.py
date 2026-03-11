from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
import os
from pathlib import Path
import time
from typing import Iterable

from sklearn.feature_extraction.text import HashingVectorizer


@dataclass
class LocalHashEmbeddings:
    """
    本地 Hash Embedding：实现简单、无需下载外部模型。
    适合离线或网络不稳定时快速跑通 RAG 链路。
    """

    n_features: int = 1024

    def __post_init__(self) -> None:
        self._vectorizer = HashingVectorizer(
            n_features=self.n_features,
            alternate_sign=False,
            norm="l2",
        )

    def _encode(self, texts: list[str]) -> list[list[float]]:
        matrix = self._vectorizer.transform(texts)
        return matrix.astype(float).toarray().tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        safe_texts = [text or "" for text in texts]
        return self._encode(safe_texts)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text or ""])[0]


@dataclass
class OpenAICompatibleEmbeddings:
    """
    直接使用 OpenAI 官方 client 调兼容接口。

    这样可以避开部分第三方 embedding 服务与 `langchain_openai.OpenAIEmbeddings`
    在分词/请求体上的兼容问题；对 Jina 这类兼容 OpenAI Embeddings API 的服务更稳。
    """

    model: str
    api_key: str
    base_url: str = ""
    batch_size: int = 64
    max_retries: int = 6
    retry_delay_sec: int = 20

    def __post_init__(self) -> None:
        from openai import OpenAI

        self._client = OpenAI(
            api_key=self.api_key or None,
            base_url=self.base_url or None,
        )

    def _sanitize_text(self, text: str | None) -> str:
        normalized = str(text or "").strip()
        # 兼容部分服务端对空串更严格的校验，避免发出空输入。
        return normalized or " "

    def _iter_batches(self, texts: list[str]) -> Iterable[list[str]]:
        size = max(int(self.batch_size or 64), 1)
        for start in range(0, len(texts), size):
            yield texts[start : start + size]

    def _create_embeddings(self, input_value):
        for attempt in range(self.max_retries + 1):
            try:
                return self._client.embeddings.create(
                    model=self.model,
                    input=input_value,
                )
            except Exception as exc:
                status_code = getattr(exc, "status_code", None)
                message = str(exc)
                is_rate_limit = status_code == 429 or "RATE_TOKEN_LIMIT_EXCEEDED" in message
                if not is_rate_limit or attempt >= self.max_retries:
                    raise
                sleep_seconds = self.retry_delay_sec * (attempt + 1)
                time.sleep(sleep_seconds)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        sanitized = [self._sanitize_text(text) for text in texts]
        if not sanitized:
            return []
        vectors: list[list[float]] = []
        for batch in self._iter_batches(sanitized):
            response = self._create_embeddings(batch)
            vectors.extend(item.embedding for item in response.data)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        response = self._create_embeddings(self._sanitize_text(text))
        return list(response.data[0].embedding)


def _build_hf_embeddings(model_name: str):
    from langchain_huggingface import HuggingFaceEmbeddings

    hf_home = os.getenv("HF_HOME", "./data/hf_cache")
    cache_folder = str(Path(hf_home).resolve())
    Path(cache_folder).mkdir(parents=True, exist_ok=True)
    device = _resolve_hf_device()
    batch_size = _resolve_hf_batch_size(device)
    model_kwargs = {"device": device}
    encode_kwargs = {
        "normalize_embeddings": True,
        "batch_size": batch_size,
    }

    # 优先走本地缓存，避免每次启动都触发远程 HEAD/下载请求。
    try:
        return HuggingFaceEmbeddings(
            model_name=model_name,
            cache_folder=cache_folder,
            model_kwargs={
                **model_kwargs,
                "local_files_only": True,
            },
            encode_kwargs=encode_kwargs,
        )
    except Exception:
        # 本地缺模型时，再回退到联网下载。
        return HuggingFaceEmbeddings(
            model_name=model_name,
            cache_folder=cache_folder,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )


def _resolve_hf_device() -> str:
    requested = os.getenv("HF_EMBED_DEVICE", "auto").strip().lower()
    if requested and requested != "auto":
        return requested

    try:
        import torch
    except Exception:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"

    mps = getattr(torch.backends, "mps", None)
    if mps is not None:
        try:
            if mps.is_available():
                return "mps"
        except Exception:
            pass
    return "cpu"


def _resolve_hf_batch_size(device: str) -> int:
    requested = os.getenv("HF_EMBED_BATCH_SIZE", "").strip()
    if requested:
        try:
            value = int(requested)
            if value > 0:
                return value
        except ValueError:
            pass
    return 64 if device == "cuda" else 16


def build_embedding_model(
    provider: str,
    model_name: str,
    *,
    api_key: str = "",
    base_url: str = "",
):
    """根据配置构建 Embedding 模型。"""
    provider = provider.strip().lower()

    if provider == "huggingface":
        # 降低终端噪声，避免大量进度条和日志刷屏。
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        try:
            from huggingface_hub.utils import disable_progress_bars

            disable_progress_bars()
        except Exception:
            pass
        try:
            from transformers.utils import logging as hf_logging

            hf_logging.set_verbosity_error()
        except Exception:
            pass

        with open(os.devnull, "w", encoding="utf-8") as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                return _build_hf_embeddings(model_name)

    if provider == "openai":
        batch_size = os.getenv("EMBEDDING_BATCH_SIZE", "").strip()
        try:
            if batch_size:
                resolved_batch_size = max(int(batch_size), 1)
            elif "jina.ai" in (base_url or "").lower():
                resolved_batch_size = 16
            else:
                resolved_batch_size = 64
        except ValueError:
            resolved_batch_size = 16 if "jina.ai" in (base_url or "").lower() else 64
        max_retries = os.getenv("EMBEDDING_MAX_RETRIES", "").strip()
        retry_delay_sec = os.getenv("EMBEDDING_RETRY_DELAY_SEC", "").strip()
        return OpenAICompatibleEmbeddings(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            batch_size=resolved_batch_size,
            max_retries=max(int(max_retries), 0) if max_retries.isdigit() else 6,
            retry_delay_sec=max(int(retry_delay_sec), 1) if retry_delay_sec.isdigit() else 20,
        )

    if provider in {"localhash", "local"}:
        return LocalHashEmbeddings()

    raise ValueError(
        "不支持的 embedding provider："
        f"{provider}。可选：huggingface / openai / localhash"
    )
