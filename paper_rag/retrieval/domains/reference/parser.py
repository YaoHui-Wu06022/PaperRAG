from __future__ import annotations

from typing import Any

from ....config import Settings
from ..common.parser_client import PlanParserClient
from .prompt import reference_parser_prompt
from .schema import validate_reference_parse


class ReferenceParserClient:
    def __init__(self, client: PlanParserClient) -> None:
        self.client = client

    @classmethod
    def from_settings(cls, settings: Settings) -> "ReferenceParserClient":
        return cls(PlanParserClient.from_settings(settings))

    def parse_reference(self, query: str) -> dict[str, Any]:
        content = self.client.complete_json(reference_parser_prompt(), query)
        return validate_reference_parse(content, query)
