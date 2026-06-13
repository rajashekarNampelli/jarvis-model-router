import pytest
from app.routing.classifier import classify


@pytest.mark.parametrize(
    "prompt,expected",
    [
        # Code keywords → qwen
        ("Write a Java unit test for my service", "qwen"),
        ("Implement a Python function to parse JSON", "qwen"),
        ("Debug this JavaScript code", "qwen"),
        ("Refactor this class to use dependency injection", "qwen"),
        ("Create a REST API endpoint", "qwen"),
        # Reasoning keywords → deepseek
        ("Prove that sqrt(2) is irrational", "deepseek"),
        ("Explain the algorithm for Dijkstra's shortest path", "deepseek"),
        ("Solve this differential equation", "deepseek"),
        ("What is the time complexity of quicksort?", "deepseek"),
        ("Apply dynamic programming to this problem", "deepseek"),
        # General → llama
        ("What is the capital of France?", "llama"),
        ("Tell me a story about dragons", "llama"),
        ("Summarise this article", "llama"),
        ("What is machine learning?", "llama"),
    ],
)
def test_classify(prompt: str, expected: str) -> None:
    assert classify(prompt) == expected


def test_classify_case_insensitive() -> None:
    assert classify("WRITE JAVA CODE") == "qwen"
    assert classify("PROVE A THEOREM") == "deepseek"


def test_classify_empty_defaults_to_llama() -> None:
    assert classify("") == "llama"
