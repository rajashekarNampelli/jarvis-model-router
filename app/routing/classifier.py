"""
Task classifier for model routing.

Strategy:
    Use a small LLM to classify the user prompt into one of three categories:
        CODE      → qwen      (programming, debugging, frameworks, infra-as-code)
        REASONING → deepseek  (math, logic, proofs, algorithm analysis)
        GENERAL   → llama     (everything else)

    Results are cached in-process (LRU) keyed by a normalized form of the prompt
    so repeated/equivalent prompts don't re-incur classifier latency.

    If the classifier call fails or returns something unparseable, we fall back
    to the safe default model (`llama`) rather than guessing.

Why LLM (not keywords)?
    Keyword sets are brittle: "JDK 21 virtual threads" has no keyword match but is
    obviously code; "optimize my marketing funnel" matches `optimize` but isn't
    math. An LLM understands the intent without us maintaining vocabulary lists.
"""

from collections import OrderedDict

from app.core.config import settings
from app.core.logging import get_logger
from app.routing.rules import DEFAULT_MODEL_KEY

logger = get_logger(__name__)

# ── Category → model mapping ─────────────────────────────────────────────────

CATEGORY_TO_MODEL: dict[str, str] = {
    "CODE": "qwen",
    "REASONING": "deepseek",
    "GENERAL": "llama",
}

# ── Classifier prompt (few-shot for accuracy + deterministic output) ────────

# Limit how much of the user prompt we ship to the classifier — first N chars
# is plenty of signal for routing intent and keeps classifier latency bounded.
_CLASSIFIER_PROMPT_CHAR_LIMIT = 500

_CLASSIFIER_SYSTEM = """You are a routing classifier. Read the user's prompt and respond with EXACTLY ONE WORD: CODE, REASONING, or GENERAL.

Definitions:
- CODE: anything about writing/reading/debugging software, programming languages, frameworks, libraries, APIs, databases, SQL, shells, configuration, build tools, CI/CD, containers, infrastructure-as-code, dev tooling.
- REASONING: math, logic, proofs, algorithmic complexity analysis, theoretical CS, formal problem solving, multi-step deduction puzzles.
- GENERAL: conversation, knowledge questions, creative writing, summaries, translation, opinions, advice — anything not clearly CODE or REASONING.

Examples:
Prompt: Write a Python function to reverse a string
Answer: CODE

Prompt: What are the key differences between JDK 8 and JDK 21?
Answer: CODE

Prompt: How do I set up a Spring Boot microservice?
Answer: CODE

Prompt: Write a Dockerfile for a Node.js app
Answer: CODE

Prompt: Prove that the square root of 2 is irrational
Answer: REASONING

Prompt: What is the time complexity of quicksort in the worst case?
Answer: REASONING

Prompt: Solve for x: 3x + 7 = 22
Answer: REASONING

Prompt: Tell me a short joke about programming
Answer: GENERAL

Prompt: What is the capital of France?
Answer: GENERAL

Prompt: Tell me a story about dragons
Answer: GENERAL

Now classify the next prompt. Reply with ONLY one word: CODE, REASONING, or GENERAL."""

# ── LRU cache ───────────────────────────────────────────────────────────────

_MAX_CACHE = 256


class _LRUCache:
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


def _normalize_cache_key(prompt: str) -> str:
    """Normalize so equivalent prompts hit the same cache entry."""
    return " ".join(prompt.strip().lower().split())


# ── LLM classifier ──────────────────────────────────────────────────────────


def _parse_category(raw: str) -> str | None:
    """Map the raw classifier response to a model key, or None if unparseable."""
    text = raw.strip().upper()
    for cat, model_key in CATEGORY_TO_MODEL.items():
        if cat in text:
            return model_key
    return None


async def _llm_classify(prompt: str) -> str | None:
    """Ask the classifier LLM; return a model key or None on any failure."""
    from app.providers import _provider

    truncated = prompt[:_CLASSIFIER_PROMPT_CHAR_LIMIT]
    classifier_input = f"{_CLASSIFIER_SYSTEM}\n\nPrompt: {truncated}\nAnswer:"

    try:
        raw = await _provider.classify(
            model=settings.classifier_model,
            prompt=classifier_input,
            timeout=settings.classifier_timeout,
        )
        result = _parse_category(raw)
        if result is not None:
            logger.info("LLM classifier: %r -> %s", prompt[:60], result)
            return result
        logger.warning("LLM classifier returned unparseable response: %r", raw[:80])
        return None
    except Exception:
        logger.warning(
            "LLM classifier failed; will fall back to default", exc_info=True
        )
        return None


# ── Public API ──────────────────────────────────────────────────────────────


async def classify(prompt: str) -> str:
    """Return a model key (`llama` | `qwen` | `deepseek`) for the given prompt.

    Uses the LLM classifier and caches the result. Falls back to the default
    model key on any failure so the request still routes successfully.
    """
    if not prompt or not prompt.strip():
        return DEFAULT_MODEL_KEY

    cache_key = _normalize_cache_key(prompt)
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        result = await _llm_classify(prompt)
    except Exception:
        # Defense in depth: _llm_classify already catches its own errors, but
        # if anything else goes wrong we must never break the request path.
        logger.warning("Classifier raised unexpectedly", exc_info=True)
        result = None

    if result is None:
        result = DEFAULT_MODEL_KEY
        logger.info("Classifier fallback to %s for: %r", result, prompt[:60])

    _cache.put(cache_key, result)
    return result
