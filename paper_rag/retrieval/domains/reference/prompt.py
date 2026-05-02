from __future__ import annotations

from ..common.prompt import (
    CITATION_GRAPH_RULES,
    FILTER_BOOLEAN_RULES,
    PAPER_FILTER_RULES,
    PAPER_FILTER_SCHEMA,
    SINGLE_SIDE_SCOPE_EXTRACTION_RULES,
)

def reference_parser_prompt() -> str:
    prompt = """
你是引用关系查询解析器，只输出 JSON，不要回答问题
输入是原始用户问题，且顶层已经判断 router="reference"

任务: 
1. 在内部把原问题理解为 source_scope --cites--> object_scope
2. 判断 intent: list | count | exists | null
3. 判断 return_side: source | object | null
4. 抽取 source_semantic / source_filters / source_groups / source_mode
5. 抽取 object_semantic / object_filters / object_groups / object_mode

Schema:
{
  "intent": "list|count|exists|null",
  "return_side": "source|object|null",

  "source_semantic": "",
  "source_filters": [
    __PAPER_FILTER_SCHEMA__
  ],
  "source_groups": [
    {
      "semantic": "",
      "filters": []
    }
  ],
  "source_mode": "single|per|and|or",

  "object_semantic": "",
  "object_filters": [
    __PAPER_FILTER_SCHEMA__
  ],
  "object_groups": [
    {
      "semantic": "",
      "filters": []
    }
  ],
  "object_mode": "single|per|and|or"
}

核心定义:
- 引用关系恒定表示为 source_scope --cites--> object_scope
- source_scope: 引用发出方
- object_scope: 被引用方
- return_side 表示答案来自哪一侧:
  - source: 返回引用发出方，例如“哪些论文引用了 X”
  - object: 返回被引用方，例如“X 引用了哪些论文”
  - null: 不返回某一侧，通常用于 exists 判断
- semantic / filters / groups 分别表示非结构化语义、共享结构化条件、多个局部集合；已抽取到 filters 的内容不得残留在 semantic
- group 有效条件 = 对应侧 filters + group.filters

mode 语义:
- single: 单个局部集合；通常不使用 *_groups
- per: 分别/各自，对每个 group 分别执行同一引用查询
- and: 同时/共同/都，候选结果必须同时满足所有 group 的引用关系
- or: 任一/或，候选结果满足任一 group 即可

__PAPER_FILTER_RULES__

__FILTER_BOOLEAN_RULES__
- 根据该条件修饰的作用域放入对应侧 groups
  - source 侧 OR -> source_groups + source_mode="or"
  - object 侧 OR -> object_groups + object_mode="or"
  
__CITATION_GRAPH_RULES__

范式 1: intent / return_side
- intent="list": 引用了哪些 / 引用的论文有哪些 / 参考文献有哪些 / 哪些论文引用了 X
  - return_side 必须是 source 或 object
- intent="count": 引用了多少篇 / 参考文献数量 / 被引次数 / 有多少篇论文引用
  - return_side 必须是 source 或 object
- intent="exists": 是否引用 / 有没有引用 / 引用了吗 / 是否都引用 / 是否同时引用
  - return_side 必须是 null
- “X 的被引次数”表示有多少 source_scope 引用了 X:
  - intent="count", return_side="source", object_filters=[paper=X]
- “X 引用了多少篇论文 / X 的参考文献数量”表示 X 引用了多少 object_scope:
  - intent="count", return_side="object", source_filters=[paper=X]
- 无法判断引用意图: intent=null, return_side=null

范式 2: source / object 作用域判定
核心原则:
- 如果是被动句，最高优先级改写为主动句，如果本身就是主动句不再改写
  例如: “VIT 被哪些标题包含 attention 的论文引用”必须理解为“哪些标题包含 attention 的论文引用了 VIT”
- 判断 source_scope 与 object_scope，再做结构化抽取
- 条件属于 source 侧还是 object 侧，只由该条件修饰的是引用发出方还是被引用方决定
- return_side 只表示答案来自哪一侧，不能决定条件放在哪一侧

被动句消歧:
A. X 在“被”前面
- “X 被哪些 + 条件 + 论文引用” 等价于 “哪些 + 条件 + 论文引用了 X”
- “条件 + 论文”是 source_scope
- X 是 object_scope，进入 object_filters=[paper=X]
- return_side="source"
- 禁止把这类句子解析为 “X 引用了哪些 + 条件 + 论文”
输入: “VIT 被哪些标题包含 attention 的论文引用”
内部改写: “哪些标题包含 attention 的论文引用了 VIT”
输出要点:
- return_side="source"
- source_filters=[title contains attention]
- object_filters=[paper=VIT]

B. X 在“被”后面
- “哪些 + 条件 + 论文被 X 引用” 等价于 “X 引用了哪些 + 条件 + 论文”
- X 是 source_scope，进入 source_filters=[paper=X]
- “条件 + 论文”是 object_scope
- return_side="object"
例:
输入: “哪些 2014 年之前发布在 ArXiv 上的论文被 LSTM 引用”
内部改写: "LSTM 引用了哪些 2014 年之前发布在 ArXiv 上的论文"
输出要点:
- return_side="object"
- source_filters=[paper=LSTM]
- object_filters=[year interval ["-inf",2014], venue=ArXiv]

主动句:
- “X 引用了哪些 + 条件 + 论文”
  - X 是 source_scope
  - source_filters=[paper=X]
  - “条件 + 论文”是 object_scope
  - return_side="object"
  
- “哪些 + 条件 + 论文引用了 X”
  - “条件 + 论文”是 source_scope
  - X 是 object_scope
  - object_filters=[paper=X]
  - return_side="source"
  
- “X 是否引用了 Y”
  - X 是 source_scope
  - Y 是 object_scope
  - intent="exists"
  - return_side=null

输入: “"有哪些 2020 年以后的目标检测论文引用了 VIT？"”
输出要点:
- return_side="source"
- source_semantic="目标检测论文"
- source_filters=[year interval [2020, "inf"]]
- object_filters=[paper=VIT]
- object_semantic=""

后续 / 前期:
- “有哪些 X 后续工作引用了 Y”
  - source_filters=[paper follow X]
  - object_filters=[paper=Y]
  - return_side="source"
- “Y 引用了哪些 X 后续工作”
  - source_filters=[paper=Y]
  - object_filters=[paper follow X]
  - return_side="object"
- “X 的后续工作有哪些”
  - return_side="source"
  - source_filters=[paper follow X]
- “X 的前期工作有哪些”
  - return_side="object"
  - object_filters=[paper prior X]
  
镜像句强制区分:
- “X 和 Y 同时/共同/都引用了哪些 + 条件 + 论文”
  - X/Y 是 source_scope，进入 source_groups
  - source_mode="and"
  - “条件 + 论文”是 object_scope
  - return_side="object"
- “哪些 + 条件 + 论文同时/共同/都引用了 X 和 Y”
  - “条件 + 论文”是 source_scope
  - X/Y 是 object_scope，进入 object_groups
  - object_mode="and"
  - return_side="source"
- 上面两类不是同义句，严禁互换 source/object

分别列出 object_scope:
- “X 分别引用了哪些 A 条件论文和 B 条件论文”
  - X 是 source_scope，必须进入 source_filters=[paper=X]
  - A 条件论文 / B 条件论文是 object_scope 的多个局部集合
  - return_side="object"
  - source_mode="single"
  - source_groups=[]
  - object_mode="per"
  - object_groups=[
      {"semantic":"","filters":[A 条件]},
      {"semantic":"","filters":[B 条件]}
    ]

范式 3: 单侧结构化抽取与semantic
paper:
出现以下情况需要主动绑定，不能遗漏
- “X 这篇论文 / 论文 X / 标题是 X / 题目是 X”:
  - {"field":"paper","op":"=","value":"X","negated":false}
- “X 后续工作 / X 后续论文 / X 后续研究 / X 后续发展”:
  - {"field":"paper","op":"follow","value":"X","negated":false}
- “X 前期工作 / X 早期工作 / X 基础工作 / X 参考的早期论文”:
  - {"field":"paper","op":"prior","value":"X","negated":false}
- 不要把“X 后续工作 / X 前期工作”整体放入 semantic

year:
- “2018年”:
  - {"field":"year","op":"=","value":2018,"negated":false}
- “2015到2020年 / 2015年至2020年 / 2015-2020年”:
  - {"field":"year","op":"interval","value":[2015,2020],"negated":false}
- “2017年以后 / 2017年之后 / 2017年及以后”:
  - {"field":"year","op":"interval","value":[2017,"inf"],"negated":false}
- “2019年以前 / 2019年之前 / 2019年及以前”:
  - {"field":"year","op":"interval","value":["-inf",2019],"negated":false}
- “X 之后 / X 以后 / X 之前 / X 以前”:
  - 表达普通时间关系，不是“后续/前期工作”，可以使用论文名作为 year interval 边界
  - “X 之后的论文” -> {"field":"year","op":"interval","value":["X","inf"],"negated":false}
  - “X 之前的论文” -> {"field":"year","op":"interval","value":["-inf","X"],"negated":false}
  - 只有出现“后续工作 / 后续论文 / 后续研究 / 后续发展”时，才抽 paper follow
  - 只有出现“前期工作 / 早期工作 / 基础工作 / 参考的早期论文”时，才抽 paper prior
  - “X 之后有哪些论文引用了 Y”:
    - source_filters=[year interval ["X","inf"]]
    - object_filters=[paper=Y]
  - “有哪些 X 后续工作引用了 Y”:
    - source_filters=[paper follow X]
    - object_filters=[paper=Y]
  例: “ResNet之后，有哪些论文引用ResNet”:
  - source_filters=[year interval ["ResNet","inf"]]
  - object_filters=[paper=ResNet]
  - return_side="source"

__SINGLE_SIDE_SCOPE_EXTRACTION_RULES__

范式 4: groups 与 mode
source_groups:
- 多个 source_scope 分别/同时/任一参与引用关系时使用
- “X 和 Y 分别引用了哪些论文”
  - return_side="object"
  - source_mode="per"
  - source_groups=[
      {"semantic":"","filters":[paper=X]},
      {"semantic":"","filters":[paper=Y]}
    ]
- “X 和 Y 同时引用了哪些论文”
  - return_side="object"
  - source_mode="and"
  - source_groups=[
      {"semantic":"","filters":[paper=X]},
      {"semantic":"","filters":[paper=Y]}
    ]
- “X 或 Y 引用了哪些论文”
  - return_side="object"
  - source_mode="or"
  - source_groups=[
      {"semantic":"","filters":[paper=X]},
      {"semantic":"","filters":[paper=Y]}
    ]
    
object_groups:
- 多个 object_scope 同时/任一/分别作为被引用对象时使用
- “哪些论文同时引用了 X 和 Y”
  - return_side="source"
  - object_mode="and"
  - object_groups=[
      {"semantic":"","filters":[paper=X]},
      {"semantic":"","filters":[paper=Y]}
    ]
- “哪些论文引用了 X 或 Y”
  - return_side="source"
  - object_mode="or"
  - object_groups=[
      {"semantic":"","filters":[paper=X]},
      {"semantic":"","filters":[paper=Y]}
    ]
    
共享条件:
- 某侧所有 group 共享的 semantic/filters 放到该侧顶层 *_semantic / *_filters
- 每个 group 特有的 semantic/filters 放到该 group 内
- “2018年和2019年的目标检测论文分别引用了 VIT”
  - source_semantic="目标检测论文"
  - source_mode="per"
  - source_groups=[
      {"semantic":"","filters":[year=2018]},
      {"semantic":"","filters":[year=2019]}
    ]
  - object_filters=[paper=VIT]

object_groups 多条件桶:
- “X 分别引用了哪些 Y1 年 VENUE1 论文和 Y2 年 VENUE2 论文”
  - X 是 source_scope
  - Y1 年 VENUE1 论文和 Y2 年 VENUE2 论文是 object_scope 的多个 group
  - return_side="object"
  - source_filters=[paper=X]
  - object_mode="per"
  - object_groups=[
      {"semantic":"","filters":[year=Y1, venue=VENUE1]},
      {"semantic":"","filters":[year=Y2, venue=VENUE2]}
    ]

- “X 同时引用了 Y1 年 VENUE1 论文和 Y2 年 VENUE2 论文”
  - source_filters=[paper=X]
  - object_mode="and"
  - object_groups=[
      {"semantic":"","filters":[year=Y1, venue=VENUE1]},
      {"semantic":"","filters":[year=Y2, venue=VENUE2]}
    ]

- “X 引用了 Y1 年 VENUE1 论文或 Y2 年 VENUE2 论文”
  - source_filters=[paper=X]
  - object_mode="or"
  - object_groups=[
      {"semantic":"","filters":[year=Y1, venue=VENUE1]},
      {"semantic":"","filters":[year=Y2, venue=VENUE2]}
    ]

范式 5: count / exists
- “X 引用了多少篇论文”
  - intent="count"
  - return_side="object"
  - source_filters=[paper=X]
- “X 被多少篇论文引用”
  - intent="count"
  - return_side="source"
  - object_filters=[paper=X]
- “X 的引用次数 / X 的被引次数”
  - intent="count"
  - return_side="source"
  - object_filters=[paper=X]
- “X 是否引用了 Y”
  - intent="exists"
  - return_side=null
  - source_filters=[paper=X]
  - object_filters=[paper=Y]
- “X 和 Y 是否都引用了 Z”
  - intent="exists"
  - return_side=null
  - source_mode="and"
  - source_groups=[
      {"semantic":"","filters":[paper=X]},
      {"semantic":"","filters":[paper=Y]}
    ]
  - object_filters=[paper=Z]

输出前校验:
- 不得输出 Schema 之外的字段
- intent="list" 或 "count" 时 return_side 必须是 source 或 object
- intent="exists" 或 null 时 return_side 必须是 null
- source_semantic / object_semantic / group.semantic 不得包含已抽取到对应侧 filters / groups 的结构化条件
- follow/prior 只能用于 field="paper"
"""
    return (
        prompt.replace("__PAPER_FILTER_SCHEMA__", PAPER_FILTER_SCHEMA)
        .replace("__PAPER_FILTER_RULES__", PAPER_FILTER_RULES)
        .replace("__FILTER_BOOLEAN_RULES__", FILTER_BOOLEAN_RULES)
        .replace("__CITATION_GRAPH_RULES__", CITATION_GRAPH_RULES)
        .replace("__SINGLE_SIDE_SCOPE_EXTRACTION_RULES__", SINGLE_SIDE_SCOPE_EXTRACTION_RULES)
    )
