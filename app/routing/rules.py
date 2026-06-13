from typing import TypedDict


class ModelEntry(TypedDict):
    provider: str
    name: str


MODEL_REGISTRY: dict[str, ModelEntry] = {
    "llama": {
        "provider": "ollama",
        "name": "llama3",
    },
    "qwen": {
        "provider": "ollama",
        "name": "qwen2.5-coder",
    },
    "deepseek": {
        "provider": "ollama",
        "name": "deepseek-r1:8b",
    },
}

# Keywords that indicate a coding-related task → Qwen
CODE_KEYWORDS: frozenset[str] = frozenset(
    [
        "java",
        "python",
        "javascript",
        "typescript",
        "golang",
        "rust",
        "kotlin",
        "swift",
        "c++",
        "c#",
        "sql",
        "code",
        "function",
        "class",
        "method",
        "unit test",
        "unittest",
        "pytest",
        "debug",
        "refactor",
        "implement",
        "api",
        "endpoint",
        "loop",
        "array",
        "list",
        "dict",
        "object",
        "variable",
        "compile",
        "runtime",
        "exception",
        "stack trace",
        "import",
        "module",
        "package",
        "library",
    ]
)

# Keywords that indicate a reasoning/math task → DeepSeek
REASONING_KEYWORDS: frozenset[str] = frozenset(
    [
        "algorithm",
        "prove",
        "proof",
        "theorem",
        "reasoning",
        "logic",
        "math",
        "mathematics",
        "calculus",
        "algebra",
        "geometry",
        "statistics",
        "probability",
        "equation",
        "formula",
        "solve",
        "deduce",
        "infer",
        "analyze",
        "complexity",
        "big o",
        "graph theory",
        "dynamic programming",
        "optimization",
        "derivative",
        "integral",
    ]
)
