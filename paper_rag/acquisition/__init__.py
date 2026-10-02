"""原始资料获取模块。"""

from paper_rag.acquisition.arxiv import (
    ArxivBatchResult,
    ArxivError,
    ArxivItemResult,
    ArxivMetadata,
    ArxivRef,
    download_arxiv_inputs,
    normalize_arxiv_input,
    preview_arxiv_inputs,
)

__all__ = [
    "ArxivBatchResult",
    "ArxivError",
    "ArxivItemResult",
    "ArxivMetadata",
    "ArxivRef",
    "download_arxiv_inputs",
    "normalize_arxiv_input",
    "preview_arxiv_inputs",
]

