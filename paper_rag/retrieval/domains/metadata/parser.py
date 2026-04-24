from __future__ import annotations

from typing import Any

from ....config import Settings
from ..common.parser_client import PlanParserClient
from .prompt import metadata_parser_system_prompt
from .schema import validate_metadata_parse


class MetadataParserClient:
    def __init__(self, client: PlanParserClient) -> None:
        self.client = client

    @classmethod
    def from_settings(cls, settings: Settings) -> "MetadataParserClient":
        return cls(PlanParserClient.from_settings(settings))

    def parse_metadata(self, query: str) -> dict[str, Any]:
        content = self.client.complete_json(metadata_parser_system_prompt(), query)
        return validate_metadata_parse(content, query)
