from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
import os
from pathlib import Path

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


def _build_hf_embeddings(model_name: str):
    from langchain_huggingface import HuggingFaceEmbeddings

    hf_home = os.getenv("HF_HOME", "./data/hf_cache")
    cache_folder = str(Path(hf_home).resolve())
    Path(cache_folder).mkdir(parents=True, exist_ok=True)

    # 优先走本地缓存，避免每次启动都触发远程 HEAD/下载请求。
    try:
        return HuggingFaceEmbeddings(
            model_name=model_name,
            cache_folder=cache_folder,
            model_kwargs={
                "device": "cpu",
                "local_files_only": True,
            },
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception:
        # 本地缺模型时，再回退到联网下载。
        return HuggingFaceEmbeddings(
            model_name=model_name,
            cache_folder=cache_folder,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )


def build_embedding_model(
    provider: str,
    model_name: str,
    *,
    openai_api_key: str = "",
    openai_base_url: str = "",
    aihubmix_api_key: str = "",
    aihubmix_base_url: str = "https://aihubmix.com/v1",
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
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=model_name,
            api_key=openai_api_key or None,
            base_url=openai_base_url or None,
        )

    if provider == "aihubmix":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=model_name,
            api_key=aihubmix_api_key or None,
            base_url=aihubmix_base_url or None,
        )

    if provider in {"localhash", "local"}:
        return LocalHashEmbeddings()

    raise ValueError(
        "不支持的 embedding provider："
        f"{provider}。可选：huggingface / openai / aihubmix / localhash"
    )
