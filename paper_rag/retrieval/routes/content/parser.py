"""content parser 封装：query -> 正文检索语义 schema。"""

from __future__ import annotations

from typing import Any

from paper_rag.config import Settings
from paper_rag.retrieval.routes.common.parser_client import PlanParserClient
from paper_rag.retrieval.routes.content.prompt import content_parser_system_prompt
from paper_rag.retrieval.routes.content.schema import validate_content_parse


class ContentParserClient:
    """调用 content prompt，并返回已校验的 parser result。"""

    def __init__(self, client: PlanParserClient) -> None:
        self.client = client

    @classmethod
    def from_settings(cls, settings: Settings) -> "ContentParserClient":
        """从 Settings 创建 content parser client。"""
        return cls(PlanParserClient.from_settings(settings))

    def parse_content(self, query: str) -> dict[str, Any]:
        """解析正文问题的 intent、paper scope 和 content objects。"""
        content = self.client.complete_json(content_parser_system_prompt(), query)
        return validate_content_parse(content, query)
