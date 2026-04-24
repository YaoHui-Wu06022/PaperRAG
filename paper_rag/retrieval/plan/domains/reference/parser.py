from __future__ import annotations

from typing import Any

from .....config import Settings
from ..metadata.parser import PlanParserClient, chat_completion_content
from ..metadata.schema import PlanParseError
from .prompt import reference_parser_prompt
from .schema import validate_reference_parse


class ReferenceParserClient(PlanParserClient):
    @classmethod
    def from_settings(cls, settings: Settings) -> "ReferenceParserClient":
        return cls(
            base_url=settings.plan_parser_base_url,
            api_key=settings.plan_parser_api_key,
            model=settings.plan_parser_model,
            timeout_seconds=settings.plan_parser_timeout_seconds,
        )

    def parse_reference(self, query: str) -> dict[str, Any]:
        if not self.base_url or not self.api_key or not self.model:
            raise PlanParseError("PLAN_PARSER_BASE_URL, PLAN_PARSER_API_KEY or PLAN_PARSER_MODEL is missing")
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": reference_parser_prompt()},
                {"role": "user", "content": query},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        try:
            data = self.chat_completion(payload)
        except PlanParseError as exc:
            if "response_format" not in str(exc):
                raise
            fallback_payload = dict(payload)
            fallback_payload.pop("response_format", None)
            data = self.chat_completion(fallback_payload)
        return validate_reference_parse(chat_completion_content(data), query)
