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
