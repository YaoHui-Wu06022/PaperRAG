"""reference parser 封装：query -> source/object citation schema。"""

from __future__ import annotations

from typing import Any

from paper_rag.config import Settings
from paper_rag.retrieval.routes.common.parser_client import PlanParserClient
from paper_rag.retrieval.routes.reference.prompt import reference_parser_prompt
from paper_rag.retrieval.routes.reference.schema import validate_reference_parse


class ReferenceParserClient:
    """调用 reference prompt，并返回已校验的 parser result。"""

    def __init__(self, client: PlanParserClient) -> None:
        self.client = client

    @classmethod
    def from_settings(cls, settings: Settings) -> "ReferenceParserClient":
        """从 Settings 创建 reference parser client。"""
        return cls(PlanParserClient.from_settings(settings))

    def parse_reference(self, query: str) -> dict[str, Any]:
        """解析引用关系中的 source/object 两侧范围。"""
        content = self.client.complete_json(reference_parser_prompt(), query)
        return validate_reference_parse(content, query)
