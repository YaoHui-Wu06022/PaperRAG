from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str:
        ...


@dataclass
class OpenAIClient:
    model: str
    temperature: float
    api_key: str
    base_url: str = ""
    api_mode: str = "chat"

    def generate(self, prompt: str) -> str:
        from openai import OpenAI

        client = OpenAI(
            api_key=self.api_key or None,
            base_url=self.base_url or None,
        )

        if self.api_mode.strip().lower() == "responses":
            kwargs = {
                "model": self.model,
                "input": prompt,
            }
            if self.temperature is not None:
                kwargs["temperature"] = self.temperature
            try:
                response = client.responses.create(**kwargs)
            except Exception as exc:
                message = str(exc).lower()
                if "temperature" in message and (
                    "not supported" in message
                    or "unsupported" in message
                    or "invalid" in message
                ):
                    kwargs.pop("temperature", None)
                    response = client.responses.create(**kwargs)
                else:
                    raise
            text = getattr(response, "output_text", None)
            if text:
                return str(text).strip()
            return str(response)

        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature

        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as exc:
            message = str(exc).lower()
            if "temperature" in message and (
                "not supported" in message
                or "unsupported" in message
                or "invalid parameter" in message
            ):
                kwargs.pop("temperature", None)
                response = client.chat.completions.create(**kwargs)
            else:
                raise
        content = response.choices[0].message.content
        return content.strip() if content else ""


def build_llm_client(
    model: str,
    temperature: float,
    api_key: str,
    base_url: str,
    api_mode: str,
) -> LLMClient:
    return OpenAIClient(
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
        api_mode=api_mode,
    )
