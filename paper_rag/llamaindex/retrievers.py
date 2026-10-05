"""LlamaIndex Retriever 适配层和确定性的 RRF 混合检索。"""

from __future__ import annotations

from typing import Any, Iterable

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle

from paper_rag.catalog.service import search_chunks
from paper_rag.config import Settings
from paper_rag.llamaindex.nodes import chunk_row_to_node
from paper_rag.llamaindex.translation import prepare_lexical_query


class SQLiteLexicalRetriever(BaseRetriever):
    """把既有 SQLite FTS5 检索暴露为 LlamaIndex Retriever。"""

    def __init__(
        self,
        settings: Settings,
        *,
        paper_ids: Iterable[str] | None = None,
        regions: Iterable[str] | None = None,
        top_k: int = 50,
    ) -> None:
        super().__init__()
        self.settings = settings
        self.paper_ids = list(paper_ids or ())
        self.regions = set(str(value) for value in (regions or ()) if str(value))
        self.top_k = max(1, int(top_k))
        self.debug: dict[str, Any] = {}
        self.warnings: list[str] = []

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        prepared = prepare_lexical_query(query_bundle.query_str, self.settings)
        self.debug = prepared.debug()
        self.warnings = list(prepared.warnings)
        items = search_chunks(
            self.settings,
            prepared.query,
            self.paper_ids or None,
            self.top_k,
            tuple(self.regions),
            fts_query=prepared.fts_query,
        )
        result: list[NodeWithScore] = []
        for rank, item in enumerate(items, start=1):
            node = chunk_row_to_node(_item_to_row(item))
            node.metadata["lexical_rank"] = rank
            node.metadata["lexical_query"] = prepared.query
            result.append(NodeWithScore(node=node, score=1.0 / rank))
        return result


class HybridRetriever(BaseRetriever):
    """组合 lexical 和 semantic Retriever，并按 chunk_id 做 RRF。"""

    def __init__(
        self,
        settings: Settings,
        semantic_retriever: BaseRetriever | None = None,
        *,
        paper_ids: Iterable[str] | None = None,
        regions: Iterable[str] | None = None,
        mode: str = "hybrid",
        lexical_top_k: int = 50,
        semantic_top_k: int = 50,
        rrf_k: int = 60,
    ) -> None:
        super().__init__()
        self.settings = settings
        self.semantic_retriever = semantic_retriever
        self.paper_ids = list(paper_ids or ())
        self.regions = set(str(value) for value in (regions or ()) if str(value))
        self.mode = mode.casefold()
        self.lexical_top_k = max(1, int(lexical_top_k))
        self.semantic_top_k = max(1, int(semantic_top_k))
        self.rrf_k = max(1, int(rrf_k))
        self.warnings: list[str] = []
        self.lexical_debug: dict[str, Any] = {
            "lexical_query": "",
            "translation_used": False,
            "translation_provider": None,
            "translation_fallback": False,
            "stopwords_removed": [],
            "rewriter_used": False,
            "rewriter_fallback": False,
            "core_terms": [],
        }

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        self.warnings = []
        self.lexical_debug = {
            "lexical_query": "",
            "translation_used": False,
            "translation_provider": None,
            "translation_fallback": False,
            "stopwords_removed": [],
            "rewriter_used": False,
            "rewriter_fallback": False,
            "core_terms": [],
        }
        lexical: list[NodeWithScore] = []
        semantic: list[NodeWithScore] = []
        if self.mode in {"lexical", "hybrid"}:
            lexical_retriever = SQLiteLexicalRetriever(
                self.settings,
                paper_ids=self.paper_ids,
                regions=self.regions,
                top_k=self.lexical_top_k,
            )
            lexical = lexical_retriever.retrieve(query_bundle)
            self.lexical_debug = lexical_retriever.debug
            self.warnings.extend(lexical_retriever.warnings)
        if self.mode in {"semantic", "hybrid"} and self.semantic_retriever is not None:
            try:
                semantic = self.semantic_retriever.retrieve(query_bundle)
                semantic = [item for item in semantic if self._allowed(item)]
            except Exception as exc:
                self.warnings.append(f"semantic_unavailable:{type(exc).__name__}")
        if self.mode == "semantic" and not semantic:
            self.warnings.append("lexical_fallback")
            if not lexical:
                lexical_retriever = SQLiteLexicalRetriever(
                    self.settings,
                    paper_ids=self.paper_ids,
                    regions=self.regions,
                    top_k=self.lexical_top_k,
                )
                lexical = lexical_retriever.retrieve(query_bundle)
                self.lexical_debug = lexical_retriever.debug
                self.warnings.extend(lexical_retriever.warnings)
        merged: dict[str, dict[str, Any]] = {}
        for rank, item in enumerate(lexical, start=1):
            key = item.node.node_id
            entry = merged.setdefault(key, {"node": item.node, "lexical_rank": rank, "semantic_rank": None})
            entry["lexical_rank"] = min(entry["lexical_rank"], rank)
        for rank, item in enumerate(semantic, start=1):
            key = item.node.node_id
            entry = merged.setdefault(key, {"node": item.node, "lexical_rank": None, "semantic_rank": rank})
            entry["semantic_rank"] = rank
            entry["semantic_score"] = item.score
        ranked: list[NodeWithScore] = []
        for entry in merged.values():
            score = 0.0
            if entry["lexical_rank"]:
                score += 1 / (self.rrf_k + entry["lexical_rank"])
            if entry["semantic_rank"]:
                score += 1 / (self.rrf_k + entry["semantic_rank"])
            entry["node"].metadata.update(
                {
                    "lexical_rank": entry["lexical_rank"],
                    "semantic_rank": entry["semantic_rank"],
                    "semantic_score": entry.get("semantic_score"),
                    "rrf_score": score,
                }
            )
            ranked.append(NodeWithScore(node=entry["node"], score=score))
        ranked.sort(key=lambda item: item.score or 0.0, reverse=True)
        return ranked

    def _allowed(self, item: NodeWithScore) -> bool:
        metadata = item.node.metadata
        if self.paper_ids and str(metadata.get("paper_id")) not in self.paper_ids:
            return False
        return not self.regions or str(metadata.get("region")) in self.regions


def _item_to_row(item: dict[str, Any]) -> tuple[Any, ...]:
    """将 Catalog 查询结果还原为 chunk_row_to_node 所需的列顺序。"""

    return (
        item.get("chunk_id"),
        item.get("paper_id"),
        item.get("canonical_id"),
        item.get("ordinal", 0),
        item.get("region", ""),
        item.get("chapter_number"),
        item.get("chapter_title"),
        item.get("section_path", []),
        item.get("section_label"),
        item.get("type", "text"),
        item.get("text", ""),
        item.get("retrieval_text", item.get("text", "")),
        item.get("page_start"),
        item.get("page_end"),
        item.get("source_blocks", []),
        item.get("asset_refs", []),
        item.get("content_hash", ""),
        item.get("retrieval_text_hash", ""),
    )


__all__ = ["HybridRetriever", "SQLiteLexicalRetriever"]
