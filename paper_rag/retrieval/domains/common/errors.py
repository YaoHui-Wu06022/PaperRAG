"""domain parser/schema 共用异常。"""

from __future__ import annotations


class PlanParseError(RuntimeError):
    """parser 输出不符合 schema 或请求失败时抛出的错误。"""

    pass
