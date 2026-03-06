from __future__ import annotations

from langchain_core.documents import Document


SYSTEM_RULES = """你是论文知识库问答助手，请严格遵守以下规则：
1. 仅基于给定上下文回答，不要编造。
2. 无论问题语言是什么，回答统一使用中文。
3. 只输出一个 JSON 对象，不要输出 Markdown、解释、前后缀文本或代码块标记。
4. 回答正文不要包含任何引用标签，引用由系统统一展示在 Sources。
5. JSON 必须包含以下字段：
{
  "conclusion": "字符串，1-3句总结",
  "evidence_points": ["字符串", "..."],
  "uncertainties": ["字符串", "..."]
}
6. 若证据不足，conclusion 要明确说明“不足以确定”，并在 uncertainties 中说明缺失信息。
"""


def _doc_citation_tag(doc: Document) -> str:
    metadata = dict(doc.metadata or {})
    doc_id = str(metadata.get("doc_id", "")).strip() or "unknown_doc"
    source = str(metadata.get("source", "unknown")).strip() or "unknown"
    page = str(metadata.get("page", "?")).strip() or "?"
    section = str(metadata.get("section_path", "正文")).strip() or "正文"

    block_ids = metadata.get("mineru_block_ids")
    block_id = ""
    if isinstance(block_ids, list) and block_ids:
        block_id = str(block_ids[0])
    elif isinstance(block_ids, str) and block_ids.strip():
        block_id = block_ids.strip()
    if not block_id:
        block_id = str(metadata.get("parent_id", metadata.get("chunk_id", "?")))

    return f"[doc:{doc_id}|source:{source}|p:{page}|section:{section}|block:{block_id}]"


def _render_context(documents: list[Document]) -> str:
    blocks: list[str] = []
    for idx, doc in enumerate(documents, start=1):
        citation = _doc_citation_tag(doc)
        content = doc.page_content.strip()
        blocks.append(f"[上下文{idx}] {citation}\n{content}")
    return "\n\n".join(blocks)


def build_qa_prompt(
    question: str,
    documents: list[Document],
    *,
    context_label: str = "上下文",
    scope_hint: str = "",
) -> str:
    context = _render_context(documents)
    extra_hint = f"{scope_hint}\n\n" if scope_hint.strip() else ""
    return (
        f"{SYSTEM_RULES}\n\n"
        f"{extra_hint}"
        "请按严格 JSON 输出，不要添加多余字段。\n\n"
        f"{context_label}：\n{context}\n\n"
        f"问题：\n{question}\n"
    )
