from __future__ import annotations

import re


ZH_TO_EN_TERMS = {
    "综述": "survey review overview",
    "总结": "summary overview",
    "方法": "method approach",
    "模型": "model architecture",
    "算法": "algorithm method",
    "实验": "experiment evaluation",
    "结果": "results performance",
    "性能": "performance results",
    "指标": "metrics score",
    "数据集": "dataset benchmark",
    "基准": "benchmark evaluation",
    "训练": "training optimization",
    "推理": "inference decoding",
    "多模态": "multimodal",
    "视觉": "vision visual",
    "图像": "image visual",
    "文本": "text language",
    "检索": "retrieval search",
    "重排序": "rerank reranking",
    "问答": "question answering qa",
    "生成": "generation generative",
    "知识库": "knowledge base",
    "医学": "medical biomedical",
    "临床": "clinical medical",
    "药物": "drug molecule",
    "蛋白": "protein biology",
    "基因": "gene genomics",
    "消融": "ablation study",
    "局限": "limitation weakness",
    "贡献": "contribution novelty",
}


EN_TO_ZH_TERMS = {
    "survey": "综述",
    "review": "综述",
    "overview": "综述",
    "method": "方法",
    "approach": "方法",
    "model": "模型",
    "architecture": "模型",
    "algorithm": "算法",
    "experiment": "实验",
    "evaluation": "评测",
    "result": "结果",
    "results": "结果",
    "performance": "性能",
    "metric": "指标",
    "metrics": "指标",
    "dataset": "数据集",
    "benchmark": "基准",
    "training": "训练",
    "inference": "推理",
    "multimodal": "多模态",
    "vision": "视觉",
    "visual": "视觉",
    "image": "图像",
    "text": "文本",
    "retrieval": "检索",
    "search": "检索",
    "rerank": "重排序",
    "reranking": "重排序",
    "question answering": "问答",
    "qa": "问答",
    "generation": "生成",
    "knowledge base": "知识库",
    "medical": "医学",
    "biomedical": "医学",
    "clinical": "临床",
    "drug": "药物",
    "molecule": "药物",
    "protein": "蛋白",
    "gene": "基因",
    "ablation": "消融",
    "limitation": "局限",
    "contribution": "贡献",
    "novelty": "创新点",
}


ACRONYM_EXPANSIONS = {
    "rag": "retrieval augmented generation",
    "llm": "large language model",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "vqa": "visual question answering",
    "qa": "question answering",
    "kg": "knowledge graph",
    "gnn": "graph neural network",
    "mlm": "masked language modeling",
    "asr": "automatic speech recognition",
}


MODEL_ALIAS_EXPANSIONS = {
    "resnet": "deep residual learning for image recognition residual network kaiming he xiangyu zhang shaoqing ren jian sun",
    "残差网络": "deep residual learning for image recognition resnet kaiming he xiangyu zhang shaoqing ren jian sun",
    "resnext": "aggregated residual transformations for deep neural networks saining xie ross girshick piotr dollar zhuowen tu kaiming he",
    "transformer": "attention is all you need ashish vaswani noam shazeer niki parmar jakob uszkoreit llion jones",
    "inception": "going deeper with convolutions christian szegedy wei liu yangqing jia pierre sermanet",
    "efficientnet": "efficientnet rethinking model scaling for convolutional neural networks mingxing tan quoc v le",
    "alexnet": "imagenet classification with deep convolutional neural networks alex krizhevsky ilya sutskever geoffrey hinton",
    "senet": "squeeze-and-excitation networks jie hu li shen gang sun",
    "se-net": "squeeze-and-excitation networks jie hu li shen gang sun",
    "eca-net": "eca-net efficient channel attention for deep convolutional neural networks qilong wang banggu wu",
    "normface": "normface l2 hypersphere embedding for face verification jian cheng feng wang alan loddon yuille",
}


MODEL_ALIAS_TITLE_HINTS = {
    "resnet": "deep residual learning for image recognition",
    "残差网络": "deep residual learning for image recognition",
    "resnext": "aggregated residual transformations for deep neural networks",
    "transformer": "attention is all you need",
    "inception": "going deeper with convolutions",
    "efficientnet": "efficientnet: rethinking model scaling for convolutional neural networks",
    "alexnet": "imagenet classification with deep convolutional neural networks",
    "senet": "squeeze-and-excitation networks",
    "se-net": "squeeze-and-excitation networks",
    "eca-net": "eca-net efficient channel attention for deep convolutional neural networks",
    "normface": "normface: l2 hypersphere embedding for face verification",
}


INTENT_HINTS = {
    "method": {
        "triggers": ("方法", "模型", "架构", "原理", "how", "method", "approach", "architecture"),
        "terms": "method approach architecture framework pipeline",
    },
    "evaluation": {
        "triggers": ("评测", "指标", "实验", "结果", "性能", "dataset", "benchmark", "metric", "evaluation", "result", "performance"),
        "terms": "dataset benchmark evaluation metrics results performance",
    },
    "summary": {
        "triggers": ("综述", "总结", "贡献", "创新", "overview", "summary", "contribution", "novelty"),
        "terms": "overview summary contribution novelty motivation",
    },
    "analysis": {
        "triggers": ("局限", "缺点", "消融", "ablation", "limitation", "weakness", "failure"),
        "terms": "ablation limitation weakness failure case analysis",
    },
}


EN_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "could",
    "do",
    "does",
    "for",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "of",
    "on",
    "please",
    "show",
    "tell",
    "the",
    "to",
    "used",
    "uses",
    "using",
    "what",
    "which",
}


ZH_FILLER_WORDS = {
    "请问",
    "帮我",
    "告诉我",
    "介绍下",
    "介绍一下",
    "解释下",
    "解释一下",
    "看看",
    "这个",
    "这篇",
    "论文",
    "文章",
    "一下",
    "一下子",
    "哪些",
    "什么",
    "怎么",
    "如何",
    "一下吧",
}


def _contains_zh(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _contains_en(text: str) -> bool:
    return bool(re.search(r"[a-zA-Z]", text))


def _normalize_spaces(text: str) -> str:
    return " ".join(text.strip().split())


def _strip_boilerplate(text: str) -> str:
    cleaned = _normalize_spaces(text)
    patterns = [
        r"^(?:请问|请帮我|帮我|麻烦你|麻烦|可以帮我|能不能帮我)\s*",
        r"^(?:what is|what are|how does|how do|can you explain|can you summarize|please explain|please summarize|tell me about)\s+",
        r"(?:可以吗|好吗|呢|呀|啊|\?+|？+)\s*$",
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
    return _dedup_terms(terms)


def _dedup_terms(terms: list[str]) -> list[str]:
    dedup: list[str] = []
    seen = set()
    for term in terms:
        item = term.strip()
        key = item.lower()
        if not item or key in seen:
            continue
        dedup.append(item)
        seen.add(key)
    return dedup


def _extract_keyword_tokens(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9._+-]*|\d{4}|[\u4e00-\u9fff]{2,}", text)
    keywords: list[str] = []
    for token in tokens:
        low = token.lower()
        if re.fullmatch(r"\d{4}", token):
            keywords.append(token)
            continue
        if _contains_zh(token):
            if token in ZH_FILLER_WORDS:
                continue
            keywords.append(token)
            continue
        if low in EN_STOPWORDS:
            continue
        keywords.append(token)
    return _dedup_terms(keywords)


def _expand_acronyms(text: str) -> list[str]:
    low = text.lower()
    terms: list[str] = []
    for acronym, expansion in ACRONYM_EXPANSIONS.items():
        if re.search(rf"\b{re.escape(acronym)}\b", low):
            terms.extend(expansion.split())
        if expansion in low:
            terms.append(acronym.upper())
    return _dedup_terms(terms)


def _expand_intent_terms(text: str) -> list[str]:
    low = text.lower()
    terms: list[str] = []
    for spec in INTENT_HINTS.values():
        if any(trigger.lower() in low for trigger in spec["triggers"]):
            terms.extend(spec["terms"].split())
    return _dedup_terms(terms)


def _expand_model_alias_terms(text: str) -> list[str]:
    low = text.lower()
    terms: list[str] = []
    for alias, expansion in MODEL_ALIAS_EXPANSIONS.items():
        alias_low = alias.lower()
        if alias in text or alias_low in low:
            terms.extend(expansion.split())
    return _dedup_terms(terms)


def extract_canonical_title_hints(text: str) -> list[str]:
    low = str(text or "").lower()
    hints: list[str] = []
    for alias, title in MODEL_ALIAS_TITLE_HINTS.items():
        alias_low = alias.lower()
        if alias in text or alias_low in low:
            hints.append(title)
    return _dedup_terms(hints)


def _build_keyword_variant(text: str) -> str:
    terms = _extract_keyword_tokens(text)
    return _normalize_spaces(" ".join(terms))


def _build_cross_lingual_variant(text: str) -> str:
    terms: list[str] = []
    if _contains_zh(text):
        terms.extend(_collect_mapped_terms(text, ZH_TO_EN_TERMS))
    if _contains_en(text):
        terms.extend(_collect_mapped_terms(text, EN_TO_ZH_TERMS))
    if not terms:
        return ""
    return _normalize_spaces(f"{text} {' '.join(_dedup_terms(terms))}")


def _build_academic_variant(text: str) -> str:
    keyword_terms = _extract_keyword_tokens(text)
    intent_terms = _expand_intent_terms(text)
    acronym_terms = _expand_acronyms(text)
    alias_terms = _expand_model_alias_terms(text)
    merged = _dedup_terms(keyword_terms + intent_terms + acronym_terms + alias_terms)
    return _normalize_spaces(" ".join(merged))


def _build_alias_variant(text: str) -> str:
    alias_terms = _expand_model_alias_terms(text)
    if not alias_terms:
        return ""
    return _normalize_spaces(f"{text} {' '.join(alias_terms)}")


def build_query_variants(
    query: str,
    *,
    enabled: bool = True,
    max_variants: int = 3,
) -> list[str]:
    normalized = _normalize_spaces(query)
    if not enabled:
        return [normalized] if normalized else []
    if not normalized:
        return []

    stripped = _strip_boilerplate(normalized)
    candidates = [
        normalized,
        stripped,
        _build_alias_variant(stripped or normalized),
        _build_keyword_variant(stripped or normalized),
        _build_cross_lingual_variant(stripped or normalized),
        _build_academic_variant(stripped or normalized),
    ]

    variants: list[str] = []
    seen = set()
    for item in candidates:
        text = _normalize_spaces(item)
        key = text.lower()
        if not text or key in seen:
            continue
        variants.append(text)
        seen.add(key)
        if len(variants) >= max(1, max_variants):
            break
    return variants
