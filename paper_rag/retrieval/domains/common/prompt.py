from __future__ import annotations


PAPER_FILTER_SCHEMA = """    
{
  "field": "paper|author|year|venue|title",
  "op": "=|contains|interval|in|follow|prior",
  "value": "",
  "negated": false
}
"""


PAPER_FILTER_RULES = """
Filter 合法组合:
- paper:
  - "=": 绑定单篇论文
  - "follow": 当前论文位于 value 的后续集合中
  - "prior": 当前论文位于 value 的前期集合中
- year: "=" | "interval"
- venue: "=" | "in"
- author: "contains"
- title: "contains"
禁止组合:
- paper in
- year contains
- author =
- title =
- venue contains
- follow/prior 用在 paper 以外字段
"""


FILTER_BOOLEAN_RULES = """
Filter 组合语义:
- 数组内的多个条件默认是 AND
- A 或 B 这种 OR 不能拆成多个 filters
- 如果同一字段支持集合 op，则用集合 op，例如 venue in ["ACL","EMNLP"]
- 如果同一字段不支持集合 op，例如 title contains / author contains，则按 OR 分组表达
"""

CITATION_GRAPH_RULES = """
citation graph 语义:
- edge(A, B) 表示 A 引用了 B
- paper follow X: 当前候选论文 P 满足 edge(P, X)，即 P 引用了 X
- paper prior X: 当前候选论文 P 满足 edge(X, P)，即 X 引用了 P
- follow/prior 是 citation graph 关系过滤，不要额外生成 year filter
"""


SINGLE_SIDE_SCOPE_EXTRACTION_RULES = """
venue:
- 单个 venue:
  - {"field":"venue","op":"=","value":"VENUE","negated":false}
- 多个候选 venue:
  例: 发表在 VENUE1 或 VENUE2
  - {"field":"venue","op":"in","value":["VENUE1","VENUE2"],"negated":false}
- “不是 / 不在 / 非 VENUE”:
  - 单个: {"field":"venue","op":"=","value":"VENUE","negated":true}
  - 多个: {"field":"venue","op":"in","value":["VENUE1","VENUE2"],"negated":true}
  - 不在多个不要用多个"="来表示
  
author:
- “X 写的论文 / 作者是 X 的论文”:
  - {"field":"author","op":"contains","value":"X","negated":false}
- “不是 X 写的论文 / 作者不是 X 的论文”:
  - {"field":"author","op":"contains","value":"X","negated":true}

title:
- “标题包含 X / 题目包含 X / title 包含 X 的论文”:
  - {"field":"title","op":"contains","value":"X","negated":false}
- “标题不包含 X / 题目不包含 X / title 不含 X 的论文”:
  - {"field":"title","op":"contains","value":"X","negated":true} 
- “标题包含 X 或 Y / 题目包含 X 或 Y”:
  - 错误输出: filters=[title contains X, title contains Y]
  - 必须按 OR 分组表达
  - 将两个 title contains 条件分别放入当前作用域对应的 groups
  - 将当前作用域对应的 mode 设为 "or"
  - groups=[
      {"semantic":"","filters":[title contains X]},
      {"semantic":"","filters":[title contains Y]}
    ]

semantic:
- semantic 只保存无法结构化的论文主题语义
- semantic 不保存普通动作词
- semantic 不保存已经进入 filters / groups 的结构化条件
- 如果移除结构化条件后只剩“论文 / 工作 / 研究 / 相关论文”等泛称，semantic=""
- 如果泛称前还有主题修饰词，则保留完整主题短语:
  - “目标检测论文” -> semantic="目标检测论文"
  - “使用 CNN 的论文” -> semantic="使用 CNN 的论文"
- “Transformer 后续的目标检测论文” -> filters=[paper follow Transformer], semantic="目标检测论文"
  
semantic 反向校验:
- semantic 中不得残留标题、年份、venue、作者、paper follow/prior 等可结构化条件
- 这些内容必须进入 filters 或 groups
"""
