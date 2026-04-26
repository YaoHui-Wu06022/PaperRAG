from __future__ import annotations

from typing import Any

from ....config import Settings
from ..common.parser_client import PlanParserClient
from .prompt import top_router_prompt
from .schema import validate_top_parse


class TopParserClient:
    def __init__(self, client: PlanParserClient) -> None:
        self.client = client

    @classmethod
    def from_settings(cls, settings: Settings) -> "TopParserClient":
        return cls(PlanParserClient.from_settings(settings))

    def parse_top(self, query: str) -> dict[str, Any]:
        content = self.client.complete_json(top_router_prompt(), query)
        return validate_top_parse(content, query)
