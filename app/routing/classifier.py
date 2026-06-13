"""
Deterministic keyword-based task classifier.

Routing logic (v1 — no AI):
  prompt contains a code keyword   → "qwen"
  prompt contains a reasoning keyword → "deepseek"
  otherwise                         → "llama"

The classify() function is the seam for a future AI-based classifier.
Replace only this function to upgrade routing without touching any
other layer.
"""

import re

from app.routing.rules import CODE_KEYWORDS, REASONING_KEYWORDS


def _matches(keyword: str, text: str) -> bool:
    """Word-boundary match to avoid false positives (e.g. 'api' inside 'capital')."""
    return bool(re.search(r"\b" + re.escape(keyword) + r"\b", text))


def classify(prompt: str) -> str:
    """Return a model key for the given prompt using deterministic rules."""
    normalised = prompt.lower()

    for keyword in CODE_KEYWORDS:
        if _matches(keyword, normalised):
            return "qwen"

    for keyword in REASONING_KEYWORDS:
        if _matches(keyword, normalised):
            return "deepseek"

    return "llama"
