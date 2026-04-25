from __future__ import annotations


def common_schema_fields_prompt() -> str:
    return """
- "filters":
  - "field": 被约束的元数据字段，只能是 ["author", "year", "venue", "title"]
    - "author": 作者
    - "year": 发表年份
    - "venue": 会议、期刊等发表来源，不会是"本地数据库"等表述
    - "title": 论文标题
  - "op":
    - "="：精确匹配作者/年份/期刊/论文标题字段
      示例："作者是 He Kaiming 的论文" -> {"field":"author","op":"=","value":"He Kaiming","negated":false}
    - "in"：字段匹配多个候选值，value 必须是数组
      示例："发表在 CVPR 或 ICCV 的论文" -> {"field":"venue","op":"in","value":["CVPR","ICCV"],"negated":false}
    - "contains"：作者/期刊/论文标题字段中包含部分内容
      示例："题目中带有 BERT 的论文" -> {"field":"title","op":"contains","value":"BERT","negated":false}
    - "interval"：年份区间，value 必须有两个边界
      - 年份范围不要使用 "contains"
      - 普通年份边界使用数字，不使用字符串
      - 开放边界使用字符串 "-inf" 或 "inf"

- 相对年份:
  - 区间边界用具体论文标题、简称、别名或缩写填充，并且必须使用字符串
  - 不要把相对年份锚点额外转成 title filter
  - "ResNet以后" -> {"field":"year","op":"interval","value":["ResNet","inf"],"negated":false}
  - "ResNet以前" -> {"field":"year","op":"interval","value":["-inf","ResNet"],"negated":false}
  - "ResNet和BERT之间" -> {"field":"year","op":"interval","value":["ResNet","BERT"],"negated":false}
  
- 某个过滤条件前带"不"，表示否定语义，使用"negated": true
  示例
  - "不在2015年之前" -> {"field":"year","op":"interval","value":["-inf",2015],"negated":true}
  - "不在2015年到2018年" -> {"field":"year","op":"interval","value":[2015,2018],"negated":true}
  - "2015到2017不在CVPR" -> {"field":"year","op":"interval","value":[2015,2017],"negated":false},{"field":"venue","op":"=","value":"CVPR","negated":true}
"""
