from __future__ import annotations

import re


CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")


def contains_chinese(text: str) -> bool:
    return bool(CHINESE_RE.search(text))
