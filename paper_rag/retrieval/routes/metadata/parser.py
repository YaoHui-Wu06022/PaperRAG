"""metadata parser 封装：query -> metadata schema。"""

from __future__ import annotations

from typing import Any

from ....config import Settings
from ..common.parser_client import PlanParserClient
from .prompt import metadata_parser_system_prompt
from .schema import validate_metadata_parse


class MetadataParserClient:
    """调用 metadata prompt，并返回已校验的 parser result。"""

    def __init__(self, client: PlanParserClient) -> None:
        self.client = client

    @classmethod
    def from_settings(cls, settings: Settings) -> "MetadataParserClient":
        """从 Settings 创建 metadata parser client。"""
        return cls(PlanParserClient.from_settings(settings))

    def parse_metadata(self, query: str) -> dict[str, Any]:
        """解析 metadata 查询意图、返回字段和论文 scope。"""
        content = self.client.complete_json(metadata_parser_system_prompt(), query)
        return validate_metadata_parse(content, query)
