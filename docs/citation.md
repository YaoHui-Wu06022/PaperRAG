# 本地引用匹配、方向与两跳查询

引用是独立于正文 RAG 的 SQLite 查询能力。实现位于 `catalog/references.py`、`catalog/service.py`、`mcp/tools/catalog.py` 和 `presentation.py`。Reference 内容不进入正文 Chunk，查询引用关系不调用 JEV、Query Rewriter、翻译、Embedding 或 Milvus。

## 1. 引用从哪里来

Catalog 同步读取 MinerU `content_list.json`，遇到 References/Reference/Bibliography 标题后提取 `ref_text`、reference、text、paragraph 等条目。Reference 之后再遇到新的章节标题，引用解析结束，避免把后续附录当作参考文献。

支持 `[1]`、`1.`、`1)` 等编号起点；编号后的无编号连续文本并入前一条。未出现编号时，按解析 Block 保存条目。条目解析依赖 MinerU 的标题和文本顺序，并不保证与 PDF 肉眼看到的参考文献条数完全一致。

每条保留 reference_id、源论文/版本、ordinal、raw_text、页码、识别出的 ArXiv ID/DOI、匹配状态，以及 matched_paper_id、match_method、match_score、duplicate_of_reference_id。原始外部、未解析和歧义条目均不删除。

## 2. 本地混合匹配

| 顺序 | 条件 | 匹配方法 |
|---|---|---|
| 1 | 提取到的 ArXiv base ID 在本地存在 | arxiv_exact，分数 1 |
| 2 | DOI 规范化后匹配本地元数据 DOI | doi_exact，分数 1 |
| 3 | 严格标题相似度，并得到作者或年份支持 | title_author_year |

ArXiv 规范化只去除结尾版本号，不使用 `split("v")`。DOI 去除 doi:/doi.org URL 前缀、URL 查询/fragment、尾部标点和括号，转小写。本地 DOI 来自 metadata.json 的 doi 字段；原始元数据没有 DOI 就无法参与 DOI 精确匹配。没有调用 Crossref 或 Semantic Scholar 补全身份。

标题和作者做 Unicode NFKC、大小写、标点、空白和简单 LaTeX 清理。当前标题匹配的数值规则是：

```text
overlap = 引用文本覆盖的标题 token 数 / 标题 token 数
sequence = SequenceMatcher(标题, 整条引用文本).ratio()
title_score = 标题完整出现时为 1，否则 0.7*overlap + 0.3*sequence
author_score = 作者姓氏命中数 / max(1, min(2, 唯一作者姓氏数))，计分时上限 1
year_score = 发表年份一致或相差一年时为 1，否则为 0
score = 0.72*title_score + 0.18*author_score + 0.10*year_score
```

只有 title_score≥0.78、作者或年份至少一项命中、score≥0.72 的候选才进入排序。前两名分差小于 0.08 时标记 ambiguous，不建立引用边。match_score 是这套身份匹配的启发式分数，不是正文语义评分，也不是经校准的正确概率。

本地精确匹配失败后仍会尝试标题匹配；识别了外部 ArXiv/DOI 不会提前结束。有标识符但无本地目标时为 external；无高置信候选且无外部标识符时为 unresolved。

## 3. 重复条目与引用边

同一源论文的多个参考文献条目如果匹配到同一目标，保留全部 raw_text，但选择一个主条目：ArXiv 精确优先于 DOI，DOI 优先于标题；同方法优先分数高者，再优先 ordinal 较小者。其他条目填写 duplicate_of_reference_id。

只有本地匹配的主条目写入 citation_edges，SQL `INSERT OR IGNORE` 进一步去重。关系为有向 `source_paper_id → target_arxiv_id`，表示源论文引用目标论文，relation=cites、resolution=local。两端都在本地库，外部论文不伪装成本地图节点。

注意：references 模式 JSON 仍保留重复原始条目。当前 presentation 的本地匹配数量和前十条按所有 local 条目计算，没有排除 duplicate_of_reference_id。因此图中的目标去重不等于 references 展示列表必然去重；人工核验可检查该字段。

## 4. 可查看的图 JSON

Catalog 同步生成与 SQLite 同目录的 `citation_graph.json`，默认路径：

```text
data/index/citation_graph.json
```

其中包含 schema_version、generated_at、scope、paper_count、reference_count、edge_count、citation_match_stats、nodes（论文 ID/标题/作者/年份）和 edges。nodes 包括本地论文记录，不仅是有连接的论文；edges 只包含本地匹配连接。SQLite 是查询权威来源，JSON 是检查副本，不由查询请求修改。

同步 issues 中 external、ambiguous、unresolved、duplicates 的数量分别表示外部目标、歧义、未解析和重复条目。external 不意味着条目错误，只表示目标不在当前本地匹配范围。总引用数可能远多于本地图连接数。

## 5. 直接按论文题目调用 MCP

Agent 从问题选择 library_citation，不必先调用 library_search：

```json
{
  "paper_title": "Attention Is All You Need",
  "mode": "graph",
  "direction": "both"
}
```

paper_id 与 paper_title 必须且只能提供一个。题目做 NFKC、大小写和标点规范化后精确匹配本地标题；不做题目翻译、简称消歧或模糊第一名选择。无匹配返回 not_found，多匹配返回 invalid_input。Agent 只有不知道完整题目或身份不确定时才需要先用元数据工具发现对象。

| mode | 查询内容 | 参数生效情况 |
|---|---|---|
| references | 原始参考文献全部条目，按 ordinal | 不使用 direction/depth/filters 扩展 |
| citations | 直接引用目标的本地论文 | filters 约束源论文；不展开多跳 |
| graph | 独立有向 BFS 后合并 | direction、depth 和 filters 生效 |

direction 只接受 out/in/both，depth 只允许 1 或 2。省略 depth 时，both 默认 2，out/in 默认 1。因此“引用和被引用关系”默认两跳；只问直接被哪些论文引用，可选 citations 或 graph/in 的默认一跳。查询间接关系必须用 graph 并设置 depth=2。

## 6. 两跳是什么

```text
出向：A → B → C
A 直接引用 B，间接引用 C。

入向：E → D → A
A 直接被 D 引用，间接被 E 引用。
```

both 分别从根运行出向和入向 BFS，再按源/目标/relation 去重。不会从 A→B 再反向寻找 D→B，然后把 D 当作 A 的间接引用或被引用。多个路径到达同一论文时展示数量按目标身份去重，已直接相关的论文不再计入间接组，根论文不计入自己的间接关系。

JSON edges 仍是实际的直接引用边，depth 是本次从根走到该边的层数，path 保存单向路线。A→B→C 的第二跳边是 B→C，不会人为增加 A→C 的直接引用边。节点/边数可以大于展示中的去重论文数量，不能将 edge_count 等同于目标论文的被引用篇数。

## 7. 过滤范围的实现边界

citations 用同一论文 SQL 编译器筛源论文。graph 每条边只在“两端都不满足 filters”时剔除，即至少一端命中便保留，并沿保留边继续 BFS。**它不是仅过滤最终间接论文的硬条件**；例如根本身满足年份条件时，直接连到根的另一端未必满足年份条件。references 当前不应用 filters。

citation 工具没有复用 search/retrieve 的未知键与年份输入校验，底层 SQL 只读取受支持键。调用时应仅使用文档字段，不依赖未知键报错。若需要严格筛最终论文年份，当前行为必须在客户端明确核验，不能把自然语言条件视为已完全执行。

## 8. 确定性回答格式

references 返回参考文献总条目数、本地匹配条目数，以及最多十条本地目标题目；external/unresolved/ambiguous 不进入 answer_text，但完整 items 保留。

citations 返回直接被引数量和最多十篇源论文题目，不添加解析状态或 scope。

graph/both/depth=2 返回：

```text
目标论文：<标题>
直接引用：<篇数> 篇
直接被引用：<篇数> 篇
查询深度：2
间接引用：<去重篇数> 篇
间接被引用：<去重篇数> 篇

引用（前10条）：
1. <直接引用目标>
────────
2. <间接引用目标>

被引用（前10条）：
1. <直接引用源>
────────
2. <间接引用源>
```

引用和被引用各有十条展示名额：先直接再间接，直接不足十条时才由间接补位；两类均实际展示时才有分割线，编号连续。直接已经满十条时不显示间接列表，但间接统计仍保留。标题缺失时回退 ID。单向两跳只显示该方向的统计和列表；一跳不显示间接数量，出向一跳标签当前为“引用”。

presentation 为 verbatim，Agent 原样输出 answer_text。完整 nodes/edges 保留在 JSON 中；文本的前十条不是图查询 LIMIT。

## 9. 更新关系是否需要重建向量

引用匹配、图遍历或展示模板改变不改变正文 retrieval_text，因此不需要重新生成正文 Embedding。更新匹配逻辑或原始 Reference 数据后，要同步 Catalog 才能得到新的 references/citation_edges/图 JSON。

不过当前 Catalog 全量同步总会刷新 indexed_at，即使正文没有变化，也可能令 index_status 出现 catalog_timestamp_mismatch。随后 auto 增量同步可更新状态，未变 Chunk 不调用 Embedding；这与“重新计算所有向量”不同。增量行为和安全边界见 [index.md](index.md)。

## 10. CLI 与核验

```powershell
conda run -n RAG_project python -X utf8 -m paper_rag citation graph --paper-id 1706.03762 --json
conda run -n RAG_project python -X utf8 -m paper_rag citation graph --paper-id 1706.03762 --direction out --depth 2 --json
```

CLI 目前只加载根论文题目，邻居的展示可能仍为 ID；MCP 会补载展示列表中的邻居题目。真实 MCP 请求和模板回答见 [record.md](record.md)，其中的数量属于执行时快照，后续入库可改变结果。
