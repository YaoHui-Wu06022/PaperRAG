from __future__ import annotations

from typing import Any

from ....config import Settings
from ..common.parser_client import PlanParserClient
from .prompt import content_parser_system_prompt
from .schema import validate_content_parse


class ContentParserClient:
    def __init__(self, client: PlanParserClient) -> None:
        self.client = client

    @classmethod
    def from_settings(cls, settings: Settings) -> "ContentParserClient":
        return cls(PlanParserClient.from_settings(settings))

    def parse_content(self, query: str) -> dict[str, Any]:
        content = self.client.complete_json(content_parser_system_prompt(), query)
        return validate_content_parse(content, query)
