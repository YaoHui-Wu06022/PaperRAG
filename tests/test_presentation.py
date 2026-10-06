from paper_rag.presentation import citation_presentation, retrieve_presentation, search_presentation


def test_search_presentation_omits_state():
    result = search_presentation(
        {
            "count": 1,
            "items": [
                {
                    "title": "Attention",
                    "paper_id": "1706.03762",
                    "authors": ["Alice", "Bob"],
                    "published_at": "2017-06-12T00:00:00Z",
                    "categories": ["cs.CL"],
                    "state": "ingested",
                }
            ],
        }
    )

    assert result["render_policy"] == "verbatim"
    assert "状态" not in result["answer_text"]
    assert "Attention" in result["answer_text"]


def test_references_presentation_lists_only_first_ten_local_matches():
    items = [
        {
            "ordinal": number,
            "resolution": "local" if number <= 12 else "external",
            "matched_paper_id": f"local-{number}" if number <= 12 else None,
        }
        for number in range(1, 15)
    ]

    result = citation_presentation(
        {"paper_id": "1706.03762", "items": items},
        "references",
        title_or_paper_id="Attention Is All You Need",
        title_lookup={"local-1": "Deep Residual Learning for Image Recognition"},
    )
    text = result["answer_text"]

    assert "论文《Attention Is All You Need》共有 14 条参考文献。" in text
    assert "本地匹配：12" in text
    assert "本地匹配：\n" not in text
    assert "1. Deep Residual Learning for Image Recognition" in text
    assert "10. local-10" in text
    assert "11. local-11" not in text
    assert "local-13" not in text


def test_citations_presentation_lists_only_first_ten_ids():
    result = citation_presentation(
        {
            "paper_id": "1706.03762",
            "items": [{"source_paper_id": f"paper-{number}"} for number in range(1, 13)],
        },
        "citations",
        title_lookup={"1706.03762": "Attention Is All You Need", "paper-1": "First Paper"},
    )

    text = result["answer_text"]
    assert "共有 12 篇论文引用了 Attention Is All You Need" in text
    assert "1. First Paper" in text
    assert "10. paper-10" in text
    assert "paper-11" not in text
    assert "relation" not in text


def test_graph_presentation_contains_only_requested_statistics():
    result = citation_presentation(
        {
            "paper_id": "1706.03762",
            "direction": "both",
            "depth": 1,
            "nodes": ["1706.03762", "2205.14135", "1512.03385"],
            "edges": [
                {"source_paper_id": "1706.03762", "target_arxiv_id": "1512.03385", "relation": "cites"},
                {"source_paper_id": "2205.14135", "target_arxiv_id": "1706.03762", "relation": "cites"},
            ],
            "scope": "local_catalog",
        },
        "graph",
        title_lookup={
            "1706.03762": "Attention Is All You Need",
            "2205.14135": "FlashAttention",
            "1512.03385": "Deep Residual Learning",
        },
    )

    text = result["answer_text"]
    assert "目标论文：Attention Is All You Need" in text
    assert "引用：1 篇" in text
    assert "直接被引用：1 篇" in text
    assert "引用（前10条）：" in text
    assert "1. Deep Residual Learning" in text
    assert "被引用（前10条）：" in text
    assert "1. FlashAttention" in text
    assert "local_catalog" not in text


def test_graph_presentation_explains_multihop_scope_without_changing_direct_counts():
    result = citation_presentation(
        {
            "paper_id": "1706.03762",
            "direction": "both",
            "depth": 2,
            "nodes": ["1706.03762", "1512.03385", "1404.5997"],
            "edges": [
                {"source_paper_id": "1706.03762", "target_arxiv_id": "1512.03385", "depth": 1},
                {"source_paper_id": "1811.06965", "target_arxiv_id": "1706.03762", "depth": 1},
                {"source_paper_id": "1512.03385", "target_arxiv_id": "1404.5997", "depth": 2},
            ],
        },
        "graph",
    )

    text = result["answer_text"]
    assert "直接引用：1 篇" in text
    assert "直接被引用：1 篇" in text
    assert "查询深度：2" in text
    assert "间接引用：1 篇" in text
    assert "间接被引用：0 篇" in text
    assert "────────" in text


def test_retrieve_presentation_only_hybrid_allows_composition():
    data = {"context_text": "[S1] evidence"}
    assert retrieve_presentation(data, "hybrid")["render_policy"] == "compose"
    assert retrieve_presentation(data, "lexical") == {
        "template_version": "library-answer-v1",
        "answer_type": "rag_evidence",
        "render_policy": "verbatim",
        "answer_text": "[S1] evidence",
    }
