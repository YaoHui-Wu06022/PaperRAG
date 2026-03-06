from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import requests


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str:
        ...


def _extract_chat_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                texts.append(str(part.get("text", "")))
        return "\n".join(t for t in texts if t).strip()
    return ""


def _extract_responses_text(payload: dict[str, Any]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = payload.get("output")
    if isinstance(output, list):
        texts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict) and part.get("type") == "output_text":
                    texts.append(str(part.get("text", "")))
        if texts:
            return "\n".join(t for t in texts if t).strip()

    return str(payload)


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
            try:
                response = client.responses.create(**kwargs)
            except Exception:
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


@dataclass
class OllamaClient:
    model: str
    temperature: float
    base_url: str

    def generate(self, prompt: str) -> str:
        endpoint = f"{self.base_url.rstrip('/')}/api/generate"
        response = requests.post(
            endpoint,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": self.temperature},
            },
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        return str(payload.get("response", "")).strip()


@dataclass
class AIHubMixClient:
    model: str
    temperature: float
    api_key: str
    base_url: str = "https://aihubmix.com/v1"
    api_mode: str = "chat"

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise ValueError("AIHUBMIX_API_KEY is empty. Please set it in .env.")
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _post(self, endpoint: str, body: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}{endpoint}"
        response = requests.post(
            url,
            headers=self._headers(),
            json=body,
            timeout=120,
        )
        if response.status_code >= 400:
            detail = response.text
            raise RuntimeError(
                f"AIHubMix API error ({response.status_code}) at {endpoint}: {detail}"
            )
        return response.json()

    def generate(self, prompt: str) -> str:
        mode = self.api_mode.strip().lower()
        if mode == "responses":
            body: dict[str, Any] = {"model": self.model, "input": prompt}
            if self.temperature is not None:
                body["temperature"] = self.temperature
            try:
                payload = self._post("/responses", body)
            except RuntimeError as exc:
                message = str(exc).lower()
                if "temperature" in message and (
                    "not supported" in message
                    or "unsupported" in message
                    or "invalid" in message
                ):
                    body.pop("temperature", None)
                    payload = self._post("/responses", body)
                else:
                    raise
            return _extract_responses_text(payload)

        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature
        try:
            payload = self._post("/chat/completions", body)
        except RuntimeError as exc:
            message = str(exc).lower()
            if "temperature" in message and (
                "not supported" in message
                or "unsupported" in message
                or "invalid" in message
            ):
                body.pop("temperature", None)
                payload = self._post("/chat/completions", body)
            else:
                raise
        return _extract_chat_content(payload)


class MockClient:
    def generate(self, prompt: str) -> str:
        return (
            "Mock answer: LLM_PROVIDER=mock is active. "
            "Switch to aihubmix/openai/ollama in .env for real generation."
        )


def build_llm_client(
    provider: str,
    model: str,
    temperature: float,
    aihubmix_api_key: str,
    aihubmix_base_url: str,
    aihubmix_api_mode: str,
    openai_api_key: str,
    openai_base_url: str,
    openai_api_mode: str,
    ollama_base_url: str,
) -> LLMClient:
    provider = provider.strip().lower()
    if provider == "aihubmix":
        return AIHubMixClient(
            model=model,
            temperature=temperature,
            api_key=aihubmix_api_key,
            base_url=aihubmix_base_url,
            api_mode=aihubmix_api_mode,
        )
    if provider == "openai":
        return OpenAIClient(
            model=model,
            temperature=temperature,
            api_key=openai_api_key,
            base_url=openai_base_url,
            api_mode=openai_api_mode,
        )
    if provider == "ollama":
        return OllamaClient(
            model=model,
            temperature=temperature,
            base_url=ollama_base_url,
        )
    return MockClient()
