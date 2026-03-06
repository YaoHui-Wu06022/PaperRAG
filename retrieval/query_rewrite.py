from __future__ import annotations

import re


ZH_TO_EN_TERMS = {
    "综述": "survey review",
    "总结": "summary",
    "方法": "method",
    "模型": "model",
    "算法": "algorithm",
    "实验": "experiment",
    "结果": "results",
    "性能": "performance",
    "指标": "metrics",
    "数据集": "dataset",
    "基准": "benchmark",
    "训练": "training",
    "推理": "inference",
    "多模态": "multimodal",
    "视觉": "vision",
    "图像": "image",
    "文本": "text",
    "检索": "retrieval",
    "重排序": "rerank",
    "问答": "question answering",
    "生成": "generation",
    "知识库": "knowledge base",
    "医学": "medical",
    "临床": "clinical",
    "药物": "drug",
    "蛋白": "protein",
    "基因": "gene",
}


EN_TO_ZH_TERMS = {
    "survey": "综述",
    "review": "综述",
    "method": "方法",
    "model": "模型",
    "algorithm": "算法",
    "experiment": "实验",
    "result": "结果",
    "performance": "性能",
    "metric": "指标",
    "dataset": "数据集",
    "benchmark": "基准",
    "training": "训练",
    "inference": "推理",
    "multimodal": "多模态",
    "vision": "视觉",
    "image": "图像",
    "text": "文本",
    "retrieval": "检索",
    "rerank": "重排序",
    "question answering": "问答",
    "generation": "生成",
    "knowledge base": "知识库",
    "medical": "医学",
    "clinical": "临床",
    "drug": "药物",
    "protein": "蛋白",
    "gene": "基因",
}


def _contains_zh(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _contains_en(text: str) -> bool:
    return bool(re.search(r"[a-zA-Z]", text))


def _normalize_spaces(text: str) -> str:
    return " ".join(text.strip().split())


def _strip_boilerplate(text: str) -> str:
    cleaned = text
    patterns = [
        r"^(请问|请帮我|帮我|请你|麻烦你)\s*",
        r"(可以吗|谢谢|好吗)\s*$",
        r"^what is\s+",
        r"^can you\s+",
    ]
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    return _normalize_spaces(cleaned)


def _collect_mapped_terms(text: str, mapping: dict[str, str]) -> list[str]:
    low = text.lower()
    terms: list[str] = []
    for key, value in mapping.items():
        if key.lower() in low:
            terms.extend(value.split())
    dedup: list[str] = []
    seen = set()
    for term in terms:
        t = term.strip().lower()
        if not t or t in seen:
            continue
        dedup.append(term)
        seen.add(t)
    return dedup


def build_query_variants(
    query: str,
    *,
    enabled: bool = True,
    max_variants: int = 3,
) -> list[str]:
    if not enabled:
        return [query]

    variants: list[str] = []
    base = _normalize_spaces(query)
    if not base:
        return []
    variants.append(base)

    stripped = _strip_boilerplate(base)
    if stripped and stripped != base:
        variants.append(stripped)

    if _contains_zh(base):
        en_terms = _collect_mapped_terms(base, ZH_TO_EN_TERMS)
        if en_terms:
            variants.append(_normalize_spaces(f"{stripped or base} {' '.join(en_terms)}"))

    if _contains_en(base):
        zh_terms = _collect_mapped_terms(base, EN_TO_ZH_TERMS)
        if zh_terms:
            variants.append(_normalize_spaces(f"{stripped or base} {' '.join(zh_terms)}"))

    dedup: list[str] = []
    seen = set()
    for item in variants:
        key = item.lower().strip()
        if not key or key in seen:
            continue
        dedup.append(item)
        seen.add(key)
        if len(dedup) >= max(1, max_variants):
            break
    return dedup
