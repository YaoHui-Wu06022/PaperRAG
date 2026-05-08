"""top route parser 封装：query -> router。"""

from __future__ import annotations

from typing import Any

from paper_rag.config import Settings
from paper_rag.retrieval.routes.common.parser_client import PlanParserClient
from paper_rag.retrieval.routes.top.prompt import top_route_prompt
from paper_rag.retrieval.routes.top.schema import validate_top_parse


class TopParserClient:
    """只负责调用 top prompt 并校验 router 字段。"""

    def __init__(self, client: PlanParserClient) -> None:
        self.client = client

    @classmethod
    def from_settings(cls, settings: Settings) -> "TopParserClient":
        """从 Settings 创建 top parser client。"""
        return cls(PlanParserClient.from_settings(settings))

    def parse_top(self, query: str) -> dict[str, Any]:
        """解析 query 所属顶层 route。"""
        content = self.client.complete_json(top_route_prompt(), query)
        return validate_top_parse(content, query)
