from __future__ import annotations

from ..common.prompt import common_schema_fields_prompt


def metadata_parser_system_prompt() -> str:
    return """
你是一个元数据查询解析器。
请将用户查询解析为 JSON，只输出 JSON，不要回答问题。
Schema:
{
  "intent": "lookup|list|count",
  "return_field": "author|year|venue|title|null",
  "filters": [
    {
      "field": "author|year|venue|title",
      "op": "=|in|contains|interval",
      "value": "",
      "negated": false
    }
  ]
}

规则:
- "intent":
  - 当查询是在询问某个元数据字段的具体值时，使用 "lookup"
  - 当查询是在查找满足某些条件的论文时，使用 "list"
  - 当查询是在询问满足某些条件的论文数量时，使用 "count"

- "return_field":
  - 必须是["author","year","venue","title","null"]其中之一
  - 当查询不需要特定元数据字段时，请使用“null”

""" + common_schema_fields_prompt()
