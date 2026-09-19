"""Publish four host-authored, source-reviewed starter pages.

Facts below were checked against the stored source spans. These are deliberately
scoped reading notes, not claims that the underlying experiments were replicated.
"""

import json
from pathlib import Path
from paper_rag.library.catalog import Catalog
from paper_rag.library.common import evidence_id
from paper_rag.library.contracts import WikiApply
from paper_rag.library.settings import LibrarySettings
from paper_rag.library.wiki import apply, lint


def main():
    with Catalog(LibrarySettings.load(Path.cwd()), writable=True) as cat:
        docs = cat.rows("SELECT * FROM documents")

        def ref(selector, quote):
            doc = next(
                d
                for d in docs
                if json.loads(d["metadata"])["title"].startswith(selector)
            )
            blocks = cat.rows(
                "SELECT * FROM blocks WHERE revision_id=? ORDER BY ordinal",
                (doc["active_revision"],),
            )
            block = next(b for b in blocks if quote in b["text"])
            start = block["text"].index(quote)
            return evidence_id(
                doc["document_id"],
                doc["active_revision"],
                block["block_id"],
                start,
                start + len(quote),
            )

        definitions = [
            (
                "gdp-rag",
                "paper",
                "GDP-RAG：按信息缺口规划检索",
                [
                    (
                        "贡献与方法（作者报告）：GDP-RAG 先检索以支撑规划，再针对缺失信息制定计划，并以携带初始证据的轨迹连接子问题。",
                        "Only Ask",
                        "(1) preliminary retrieval to ground planning before execution, (2) a gap-conditioned planning prompt that asks only for missing information, and (3) a skeletal trajectory",
                    ),
                    (
                        "实验（作者报告）：摘要报告准确率 60.63%、cost-of-pass 为 0.51；这些数字是论文报告值。",
                        "Only Ask",
                        "accuracy (60.63%) among all compared systems while maintaining a cost-of-pass of 0.51",
                    ),
                ],
            ),
            (
                "scaled-attention",
                "concept",
                "Scaled Dot-Product Attention",
                [
                    (
                        "定义：原文公式为 Attention(Q,K,V) = softmax(QKᵀ/√dₖ)V。",
                        "Attention is All",
                        r"\operatorname{Attention} (Q, K, V) = \operatorname{softmax} (\frac {Q K ^ {T}}{\sqrt {d _ {k}}}) V\tag{1}",
                    ),
                    (
                        "结构：MultiHead 将各 head 拼接后通过 Wᴼ 投影。",
                        "Attention is All",
                        r"\mathrm{Concat} (\mathrm{head} _ {1},..., \mathrm{head} _ {\mathrm{h}}) W ^ {O}",
                    ),
                ],
            ),
            (
                "channel-attention",
                "comparison",
                "SE 与 ECA 的通道建模比较",
                [
                    (
                        "SE：通过显式建模通道之间的依赖，调整通道特征响应。",
                        "Squeeze-and-Excitation",
                        "adaptively recalibrates channel-wise feature responses by explicitly modelling interdependencies between channels",
                    ),
                    (
                        "ECA：使用不降维的局部跨通道交互，通过一维卷积实现。",
                        "ECA-Net",
                        "local crosschannel interaction strategy without dimensionality reduction, which can be efficiently implemented via 1D convolution",
                    ),
                ],
            ),
            (
                "agent-research-memory",
                "topic",
                "Agent 研究中的证据、记忆与检索规划",
                [
                    (
                        "持久研究资料：Reconcile Once 的 librarian 维护证据卡片、权威指标账本和 claim graph。",
                        "Reconcile Once",
                        "evidence cards, an authoritative metric ledger, and a claim graph",
                    ),
                    (
                        "经验记忆：Learning on the Job 将反馈提炼为可检索的自然语言规则，模型权重保持冻结。",
                        "Learning on the Job",
                        "external memory that distils each episode into retrievable natural-language rules",
                    ),
                    (
                        "证据约束：Citation-Enforced RAG 保存文档标识和页级来源。",
                        "Citation-Enforced",
                        "document identifiers and page-level provenance",
                    ),
                ],
            ),
        ]
        results = []
        for page_id, kind, title, statements in definitions:
            if cat.one("SELECT 1 FROM wiki_pages WHERE page_id=?", (page_id,)):
                results.append(
                    {"page_id": page_id, "status": "already_exists_preserved"}
                )
                continue
            body, evidence, quotes = [], [], {}
            for statement, selector, quote in statements:
                value = ref(selector, quote)
                body.append(statement + f" [原文]({value})")
                evidence.append(value)
                quotes[value] = quote
            body.append(
                "核读范围与局限：这是宿主 Agent 按以上明确原文范围整理的初始页面；未据此独立复现实验，也未完成全文局限性综述。"
            )
            results.append(
                apply(
                    cat,
                    WikiApply(
                        page_id=page_id,
                        kind=kind,
                        title=title,
                        expected_revision=0,
                        body="\n\n".join(body),
                        evidence_ids=evidence,
                        quotes=quotes,
                    ),
                )
            )
        print(json.dumps({"pages": results, "lint": lint(cat)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
