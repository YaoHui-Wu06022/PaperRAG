"""为 Agent 答案生成和引用校验保存一次检索上下文。"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
import re
from threading import RLock
import time
from typing import Any, Iterable, Mapping
from uuid import uuid4


ANSWER_STATUS = frozenset({"answered", "insufficient_evidence"})
_INLINE_CITATION_RE = re.compile(r"\[(S\d+)\]")
_REGISTRY_FIELDS = (
    "paper_id",
    "canonical_id",
    "chunk_id",
    "section_path",
    "section_label",
    "page_start",
    "page_end",
    "type",
    "evidence_role",
    "window_id",
    "continuity_status",
)


@dataclass(frozen=True)
class AnswerContext:
    """一次正文检索对应的真实证据和引用注册表。"""

    context_id: str
    created_at: float
    query: str
    task: str
    mode: str
    filters: dict[str, Any]
    regions: tuple[str, ...]
    items: tuple[dict[str, Any], ...]
    context_text: str
    truncated: bool
    citation_registry: dict[str, dict[str, Any]]


class AnswerContextStore:
    """进程内、有上限和 TTL 的 Agent 答案上下文缓存。"""

    def __init__(self, *, ttl_seconds: int = 1800, max_contexts: int = 128) -> None:
        self.ttl_seconds = max(1, int(ttl_seconds))
        self.max_contexts = max(1, int(max_contexts))
        self._contexts: OrderedDict[str, AnswerContext] = OrderedDict()
        self._lock = RLock()

    def create(
        self,
        *,
        query: str,
        task: str,
        mode: str,
        filters: Mapping[str, Any] | None,
        regions: Iterable[str],
        items: Iterable[Mapping[str, Any]],
        context_text: str,
        truncated: bool,
    ) -> AnswerContext:
        """保存检索结果，并从真实返回项生成引用注册表。"""

        copied_items = tuple(deepcopy(dict(item)) for item in items)
        registry = build_citation_registry(copied_items)
        context = AnswerContext(
            context_id=f"ctx-{uuid4().hex}",
            created_at=time.monotonic(),
            query=str(query),
            task=str(task),
            mode=str(mode),
            filters=deepcopy(dict(filters or {})),
            regions=tuple(str(value) for value in regions),
            items=copied_items,
            context_text=str(context_text or ""),
            truncated=bool(truncated),
            citation_registry=registry,
        )
        with self._lock:
            self._purge_locked()
            self._contexts[context.context_id] = context
            self._contexts.move_to_end(context.context_id)
            while len(self._contexts) > self.max_contexts:
                self._contexts.popitem(last=False)
        return context

    def get(self, context_id: str) -> AnswerContext | None:
        """读取未过期上下文；过期上下文按 miss 处理。"""

        wanted = str(context_id or "").strip()
        if not wanted:
            return None
        with self._lock:
            self._purge_locked()
            context = self._contexts.get(wanted)
            if context is not None:
                self._contexts.move_to_end(wanted)
            return context

    def clear(self) -> None:
        """清空上下文，供测试和进程关闭时使用。"""

        with self._lock:
            self._contexts.clear()

    def _purge_locked(self) -> None:
        now = time.monotonic()
        expired = [
            key
            for key, value in self._contexts.items()
            if now - value.created_at >= self.ttl_seconds
        ]
        for key in expired:
            self._contexts.pop(key, None)


def build_citation_registry(items: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """只复制检索结果中已有的来源定位字段，不生成论文事实。"""

    registry: dict[str, dict[str, Any]] = {}
    for item in items:
        source_id = str(item.get("source_id") or "").strip()
        if not source_id:
            continue
        entry = {key: deepcopy(item.get(key)) for key in _REGISTRY_FIELDS}
        chunk_id = str(item.get("chunk_id") or "").strip()
        source_chunk_ids = item.get("source_chunk_ids")
        if source_chunk_ids:
            entry["source_chunk_ids"] = deepcopy(source_chunk_ids)
        elif chunk_id:
            entry["source_chunk_ids"] = [chunk_id]
        registry[source_id] = entry
    return registry


def answer_contract() -> dict[str, Any]:
    """返回 Agent 组织可溯源答案时必须遵守的静态契约。"""

    return {
        "answer_status": ["answered", "insufficient_evidence"],
        "required_fields": ["answer_status", "answer", "claims", "citations"],
        "citation_syntax": "[S1]",
        "claim_fields": ["claim_id", "text", "citation_ids"],
        "rules": [
            "每条事实性 claim 至少包含一个 citation_id",
            "只能引用当前 citation_registry 中存在的 source_id",
            "不得补充检索结果中没有的论文、页码或实验数据",
            "证据不足时使用 insufficient_evidence",
        ],
    }


def validate_answer_payload(
    context: AnswerContext,
    *,
    answer_status: str,
    answer: str,
    claims: list[dict[str, Any]],
    citations: list[str],
) -> list[dict[str, str]]:
    """校验 Agent 答案是否只引用当前检索上下文中的来源。"""

    errors: list[dict[str, str]] = []
    status = str(answer_status or "")
    if status not in ANSWER_STATUS:
        errors.append({"code": "invalid_answer_status", "message": "answer_status 不受支持"})
    if not isinstance(answer, str) or not answer.strip():
        errors.append({"code": "empty_answer", "message": "answer 必须是非空字符串"})
    if not isinstance(claims, list):
        errors.append({"code": "invalid_claims", "message": "claims 必须是数组"})
        claims = []
    if not isinstance(citations, list):
        errors.append({"code": "invalid_citations", "message": "citations 必须是数组"})
        citations = []

    registry_ids = set(context.citation_registry)
    inline_ids = set(_INLINE_CITATION_RE.findall(answer if isinstance(answer, str) else ""))
    unknown_inline = sorted(inline_ids - registry_ids)
    for source_id in unknown_inline:
        errors.append({"code": "unknown_inline_citation", "message": f"未知内联引用 {source_id}"})

    claim_ids: set[str] = set()
    used_ids: set[str] = set()
    if status == "answered" and not claims:
        errors.append({"code": "claims_required", "message": "answered 状态至少需要一条 claim"})
    for claim in claims:
        if not isinstance(claim, dict):
            errors.append({"code": "invalid_claim", "message": "claim 必须是对象"})
            continue
        unsupported = sorted(set(claim) - {"claim_id", "text", "citation_ids"})
        if unsupported:
            errors.append({"code": "unsupported_claim_field", "message": "claim 含有未定义字段"})
        claim_id = str(claim.get("claim_id") or "").strip()
        if not claim_id:
            errors.append({"code": "empty_claim_id", "message": "claim_id 不能为空"})
        elif claim_id in claim_ids:
            errors.append({"code": "duplicate_claim_id", "message": f"重复 claim_id {claim_id}"})
        else:
            claim_ids.add(claim_id)
        if not isinstance(claim.get("text"), str) or not claim["text"].strip():
            errors.append({"code": "empty_claim_text", "message": f"claim {claim_id or '?'} 缺少文本"})
        citation_ids = claim.get("citation_ids")
        if not isinstance(citation_ids, list):
            errors.append({"code": "invalid_claim_citations", "message": f"claim {claim_id or '?'} 的 citation_ids 必须是数组"})
            citation_ids = []
        if status == "answered" and not citation_ids:
            errors.append({"code": "missing_claim_citation", "message": f"claim {claim_id or '?'} 没有引用"})
        for source_id in citation_ids:
            if not isinstance(source_id, str) or not source_id:
                errors.append({"code": "invalid_citation_id", "message": f"claim {claim_id or '?'} 含有无效引用"})
                continue
            used_ids.add(source_id)
            if source_id not in registry_ids:
                errors.append({"code": "unknown_citation_id", "message": f"未知引用 {source_id}"})

    citation_values = [value for value in citations if isinstance(value, str)]
    if len(citation_values) != len(citations):
        errors.append({"code": "invalid_citation_id", "message": "顶层 citations 含有无效引用"})
    if len(citation_values) != len(set(citation_values)):
        errors.append({"code": "duplicate_citation_id", "message": "顶层 citations 含有重复引用"})
    top_level_ids = set(citation_values)
    if top_level_ids != used_ids:
        errors.append({"code": "citation_set_mismatch", "message": "顶层 citations 与 claims 使用的引用不一致"})
    for source_id in sorted(top_level_ids - registry_ids):
        errors.append({"code": "unknown_citation_id", "message": f"未知引用 {source_id}"})
    for source_id in sorted(inline_ids - used_ids):
        errors.append({"code": "unlinked_inline_citation", "message": f"内联引用 {source_id} 未绑定到 claim"})
    for source_id in sorted(used_ids - inline_ids):
        errors.append({"code": "missing_inline_citation", "message": f"答案缺少内联引用 {source_id}"})
    return errors


ANSWER_CONTEXTS = AnswerContextStore()


__all__ = [
    "ANSWER_CONTEXTS",
    "ANSWER_STATUS",
    "AnswerContext",
    "AnswerContextStore",
    "answer_contract",
    "build_citation_registry",
    "validate_answer_payload",
]
