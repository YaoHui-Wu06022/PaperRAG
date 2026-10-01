"""DeepSeek 结构化查询抽取。"""

from paper_rag.extraction.deepseek import (
    EXTRACTION_PROMPT_VERSION,
    DeepSeekExtractionClient,
    ExtractionError,
    extract_query_with_cache,
    extract_with_cache,
)
from paper_rag.extraction.schema import QueryExtraction

__all__ = [
    "EXTRACTION_PROMPT_VERSION",
    "DeepSeekExtractionClient",
    "ExtractionError",
    "extract_query_with_cache",
    "extract_with_cache",
    "QueryExtraction",
]
