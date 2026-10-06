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


def test_retrieve_presentation_always_requires_composition():
    data = {"task": "fact", "evidence": [{"source_id": "S1", "text": "evidence"}]}
    for mode in ("hybrid", "lexical", "semantic"):
        result = retrieve_presentation(data, mode)
        assert result["render_policy"] == "compose"
        assert result["answer_text"] == ""


def test_retrieve_presentation_returns_structured_agent_instruction():
    result = retrieve_presentation({"task": "reason", "evidence": []}, "hybrid")
    instruction = result["agent_instruction"]

    assert set(instruction) == {"version", "task", "system_prompt"}
    assert instruction["version"] == "rag-agent-v1"
    assert instruction["task"] == "reason"
    assert "data.evidence" in instruction["system_prompt"]
    assert "中文" in instruction["system_prompt"]
    assert "[S#]" in instruction["system_prompt"]
    assert "为什么、如何或机制" in instruction["system_prompt"]
    assert "language" not in instruction
    assert "citation_syntax" not in instruction
    assert "evidence_field" not in instruction
    assert "evidence_field" not in instruction["system_prompt"]


def test_retrieve_presentation_has_task_specific_instructions():
    prompts = {
        task: retrieve_presentation({"task": task}, "hybrid")["agent_instruction"]["system_prompt"]
        for task in ("fact", "reason", "summary", "comparison")
    }

    assert len(set(prompts.values())) == 4
    assert "直接回答问题" in prompts["fact"]
    assert "解释问题中的为什么" not in prompts["fact"]
    assert "按共同维度比较" not in prompts["fact"]
    assert "解释问题中的为什么" in prompts["reason"]
    assert "按共同维度比较" not in prompts["reason"]
    assert "组织摘要" in prompts["summary"]
    assert "解释问题中的为什么" not in prompts["summary"]
    assert "按共同维度比较" in prompts["comparison"]
    assert "组织摘要" not in prompts["comparison"]


def test_insufficient_evidence_instruction_forbids_fake_citations():
    result = retrieve_presentation({"task": "fact", "evidence": []}, "hybrid", status="insufficient_evidence")
    prompt = result["agent_instruction"]["system_prompt"]

    assert "没有足够的正文证据" in prompt
    assert "无法可靠回答该问题" in prompt
    assert "不存在的 [S#] 引用" in prompt
