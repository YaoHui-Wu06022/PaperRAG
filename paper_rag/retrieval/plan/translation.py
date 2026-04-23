from __future__ import annotations

import hashlib
import json
import random
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass


CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")


class TranslationError(RuntimeError):
    pass


@dataclass(frozen=True)
class TranslationResult:
    text: str
    provider: str


class BaiduTranslator:
    def __init__(
        self,
        *,
        app_id: str | None,
        secret_key: str | None,
        endpoint: str,
        domain: str | None = None,
    ) -> None:
        self.app_id = app_id
        self.secret_key = secret_key
        self.endpoint = endpoint
        self.domain = domain

    def translate_to_english(self, text: str) -> TranslationResult:
        if not self.app_id or not self.secret_key:
            raise TranslationError("Baidu translate credentials are missing")
        salt = str(random.randint(32768, 65536))
        if self.domain:
            sign_source = f"{self.app_id}{text}{salt}{self.domain}{self.secret_key}"
        else:
            sign_source = f"{self.app_id}{text}{salt}{self.secret_key}"
        sign = hashlib.md5(sign_source.encode("utf-8")).hexdigest()
        params = {
            "q": text,
            "from": "zh",
            "to": "en",
            "appid": self.app_id,
            "salt": salt,
            "sign": sign,
        }
        if self.domain:
            params["domain"] = self.domain
        payload = urllib.parse.urlencode(params).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=payload,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8")
        data = json.loads(body)
        if "error_code" in data:
            message = data.get("error_msg") or data["error_code"]
            raise TranslationError(f"Baidu translate failed: {message}")
        results = data.get("trans_result") or []
        translated = " ".join(str(item.get("dst") or "") for item in results).strip()
        if not translated:
            raise TranslationError("Baidu translate returned an empty result")
        return TranslationResult(translated, "baidu")


def contains_chinese(text: str) -> bool:
    return bool(CHINESE_RE.search(text))
