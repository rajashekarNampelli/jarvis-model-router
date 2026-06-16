"""
LLM-based task classifier with safe default fallback.

Routing logic (v2 — AI-first):
  1. Ask a small LLM to classify the prompt -> CODE / REASONING / GENERAL
  2. If the LLM fails, times out, or returns unparseable output -> default to "llama"
  3. Cache results to avoid re-classifying identical prompts

The classify() function is async and returns a model key string.
"""

from collections import OrderedDict
from typing import Literal

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# ── Category mapping ────────────────────────────────────────────────────────
Category = Literal["CODE", "REASONING", "GENERAL"]

CATEGORY_TO_MODEL: dict[Category, str] = {
    "CODE": "qwen",
    "REASONING": "deepseek",
    "GENERAL": "llama",
}

# ── System prompt for the classifier LLM ───────────────────────────────────
_CLASSIFIER_SYSTEM = (
    "You are a router. Classify the user prompt as exactly one word: "
    "CODE, REASONING, or GENERAL. "
    "CODE = programming, debugging, software. "
    "REASONING = math, logic, proofs, algorithms. "
    "GENERAL = everything else. "
    "Respond with ONLY that one word. No explanation."
)

# ── LRU Cache ───────────────────────────────────────────────────────────────
_MAX_CACHE = 256


class _LRUCache:
    """Simple OrderedDict-based LRU cache for classification results."""

    def __init__(self, maxsize: int = _MAX_CACHE) -> None:
        self._maxsize = maxsize
        self._store: OrderedDict[str, str] = OrderedDict()

    def get(self, key: str) -> str | None:
        if key in self._store:
            self._store.move_to_end(key)
            return self._store[key]
        return None

    def put(self, key: str, value: str) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        else:
            self._store[key] = value
            if len(self._store) > self._maxsize:
                self._store.popitem(last=False)


_cache = _LRUCache()


# ── LLM classifier ─────────────────────────────────────────────────────────
def _parse_category(raw: str) -> Category | None:
    """Extract a valid category from the LLM's free-form response."""
    text = raw.strip().upper()
    for cat in CATEGORY_TO_MODEL:
        if cat in text:
            return cat
    return None


async def _llm_classify(prompt: str) -> str | None:
    """Ask the classifier LLM and return a model key, or None on failure."""
    from app.providers import _provider

    try:
        raw = await _provider.generate(
            model=settings.classifier_model,
            prompt=_CLASSIFIER_SYSTEM + "\n\nPrompt: " + prompt,
        )
        category = _parse_category(raw)
        if category is not None:
            model_key = CATEGORY_TO_MODEL[category]
            logger.info(
                "LLM classifier: prompt=%r -> category=%s -> model=%s",
                prompt[:60],
                category,
                model_key,
            )
            return model_key
        logger.warning("LLM classifier returned unparseable response: %r", raw[:80])
        return None
    except Exception:
        logger.warning("LLM classifier failed, defaulting to general", exc_info=True)
        return None


# ── Public API ──────────────────────────────────────────────────────────────
async def classify(prompt: str) -> str:
    """Return a model key for the given prompt. LLM classifier, default to general on failure."""
    if not prompt:
        return "llama"

    # Check cache first
    cached = _cache.get(prompt)
    if cached is not None:
        return cached

    # Try LLM classifier
    try:
        result = await _llm_classify(prompt)
        if result is not None:
            _cache.put(prompt, result)
            return result
    except Exception:
        logger.warning("LLM classifier raised unexpectedly, defaulting to general", exc_info=True)

    # Fallback: default to general
    logger.info("Classifier fallback: prompt=%r -> model=llama", prompt[:60])
    _cache.put(prompt, "llama")
    return "llama"
