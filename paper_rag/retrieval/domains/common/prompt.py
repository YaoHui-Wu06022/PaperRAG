from __future__ import annotations


def common_schema_fields_prompt() -> str:
    return """
- "field":
  - "author"：作者
  - "year"：年份
  - "venue"：会议、期刊、发表来源，"value"不会是"本地数据库"等表述
  - "title"：论文标题

- "anchors":
  - 字符串列表，只存论文标题、别名或缩写
  - 问某篇论文的数据时用 anchors
  - 按标题关键词筛选时用 filters.title

- "op":
  - "="：精确匹配作者/年份/期刊/论文标题字段
    示例："作者是 He Kaiming 的论文" -> {"field":"author","op":"=","value":"He Kaiming","negated":false}
  - "in"：字段匹配多个候选值，value 必须是数组
    示例："发表在 CVPR 或 ICCV 的论文" -> {"field":"venue","op":"in","value":["CVPR","ICCV"],"negated":false}
  - "contains"：作者/期刊/论文标题字段中包含部分内容
    示例："题目中带有 BERT 的论文" -> {"field":"title","op":"contains","value":"BERT","negated":false}
  - "interval"：年份区间，value 必须有两个边界
  - 年份范围不要使用 "contains"
  
- 年份区间:
  - 普通年份边界使用数字，不使用字符串
  - 开放边界使用字符串 "-inf" 或 "inf"

- 相对锚点年份:
  - 区间边界相对于锚点论文，不要推断具体年份，用字符串 "anchor" 占位
  - "锚点以后" -> {"field":"year","op":"interval","value":["anchor","inf"],"negated":false}
  - "锚点以前" -> {"field":"year","op":"interval","value":["-inf","anchor"],"negated":false}
  - "锚点和锚点之间""锚点之前锚点之后" -> {"field":"year","op":"interval","value":["anchor","anchor"],"negated":false}
  示例：
  - "ResNet和BERT之间有哪些论文" -> {"field":"year","op":"interval","value":["anchor","anchor"],"negated":false}

- 某个过滤条件前带"不"，表示否定语义，使用"negated": true
  示例
  - "不在2015年之前" -> {"field":"year","op":"interval","value":["-inf",2015],"negated":true}
  - "不在2015年到2018年" -> {"field":"year","op":"interval","value":[2015,2018],"negated":true}
  - "2015到2017不在CVPR" -> {"field":"year","op":"interval","value":[2015,2017],"negated":false},{"field":"venue","op":"=","value":"CVPR","negated":true}
"""
