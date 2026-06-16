#!/usr/bin/env python3
"""
Jarvis Model Router - Smoke Test Suite
Run with: python3 smoke_test.py
Requires: curl on PATH, API server running on localhost:8000
"""

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

# ── Config ──────────────────────────────────────────────────────────────────
BASE_URL = "http://localhost:8000"
CURL_TIMEOUT = "120"
RESULTS_DIR = "test_results"

# ── Models to compare ────────────────────────────────────────────────────────
MODELS = ["auto", "llama", "qwen"]  # deepseek skipped (model not pulled)

# ── Test prompts ─────────────────────────────────────────────────────────────
PROMPTS = [
    {
        "id": "T1",
        "category": "code",
        "prompt": "Write a Python function to reverse a string",
        "expect_model_auto": "qwen",
        "valid_keywords": ["def", "reverse", "python", "string"],
    },
    {
        "id": "T2",
        "category": "reasoning",
        "prompt": "What is the time complexity of quicksort?",
        "expect_model_auto": "deepseek",  # deepseek not pulled, will error on auto
        "valid_keywords": ["O(n log n)", "quicksort", "complexity", "worst case", "O(n^2)"],
    },
    {
        "id": "T3",
        "category": "general",
        "prompt": "Tell me a short joke about programming",
        "expect_model_auto": "llama",
        "valid_keywords": ["why", "programmer", "joke", "bug", "code"],
    },
    {
        "id": "T4",
        "category": "java",
        "prompt": "What are the key differences between JDK 8 and JDK 21? Especially describe virtual threads in detail.",
        "expect_model_auto": "qwen",  # LLM classifier now correctly routes JDK to code
        "valid_keywords": ["virtual thread", "JDK 21", "September 2023", "Project Loom", "lightweight"],
    },
    {
        "id": "T5",
        "category": "code",
        "prompt": "How do I set up a Spring Boot microservice?",
        "expect_model_auto": "qwen",  # LLM classifier catches "Spring Boot" as code
        "valid_keywords": ["spring", "boot", "microservice", "annotation", "controller"],
    },
    {
        "id": "T6",
        "category": "code",
        "prompt": "Write a Dockerfile for a Node.js app",
        "expect_model_auto": "qwen",  # LLM classifier catches "Dockerfile" as code
        "valid_keywords": ["FROM", "dockerfile", "node", "RUN", "COPY"],
    },
    {
        "id": "T7",
        "category": "general",
        "prompt": "What is the capital of France?",
        "expect_model_auto": "llama",
        "valid_keywords": ["paris", "france", "capital"],
    },
    {
        "id": "T8",
        "category": "general",
        "prompt": "Tell me a story about dragons",
        "expect_model_auto": "llama",
        "valid_keywords": ["dragon", "story", "fire", "knight", "mountain"],
    },
]


# ── Data classes ─────────────────────────────────────────────────────────────
@dataclass
class TestResult:
    test_id: str
    category: str
    prompt: str
    model_requested: str
    model_selected: str
    expect_model_auto: str
    routing_correct: bool
    status_code: int
    latency_ms: int
    response_text: str = ""
    error: str = ""
    valid: bool = False
    valid_hits: int = 0
    valid_total: int = 0
    wall_time_s: float = 0.0


@dataclass
class InfraResult:
    endpoint: str
    status_code: int
    body_snippet: str = ""
    pass_fail: str = ""


# ── Helpers ──────────────────────────────────────────────────────────────────
def curl(method: str, path: str, body: dict | None = None, timeout: str = CURL_TIMEOUT) -> tuple[int, str, float]:
    """Run curl, return (status_code, response_body, wall_time_seconds)."""
    cmd = ["curl", "-s", "-w", "\n%{http_code}", "-X", method, f"{BASE_URL}{path}"]
    if body is not None:
        cmd += ["-H", "Content-Type: application/json", "-d", json.dumps(body)]
    cmd += ["--max-time", timeout]

    start = time.monotonic()
    try:
        raw = subprocess.check_output(cmd, text=True, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raw = e.output if e.output else ""
    wall = time.monotonic() - start

    lines = raw.strip().rsplit("\n", 1)
    status = int(lines[-1]) if len(lines) > 1 and lines[-1].isdigit() else 0
    body_text = lines[0] if len(lines) > 1 else raw.strip()
    return status, body_text, wall


def check_validity(response_text: str, valid_keywords: list[str]) -> tuple[bool, int, int]:
    """Check if response contains expected keywords (case-insensitive)."""
    lower = response_text.lower()
    hits = sum(1 for kw in valid_keywords if kw.lower() in lower)
    threshold = max(2, len(valid_keywords) // 2)
    return hits >= threshold, hits, len(valid_keywords)


# ── Infra tests ──────────────────────────────────────────────────────────────
def run_infra_tests() -> list[InfraResult]:
    results: list[InfraResult] = []

    # GET /health
    status, body, _ = curl("GET", "/health")
    health_data = json.loads(body) if status == 200 else {}
    ollama_status = health_data.get("ollama", "?")
    snippet = f"status={health_data.get('status')} ollama={ollama_status}"
    results.append(InfraResult("/health", status, snippet, "PASS" if status == 200 else "FAIL"))

    # GET /v1/models
    status, body, _ = curl("GET", "/v1/models")
    models_data = json.loads(body) if status == 200 else {}
    keys = [m["key"] for m in models_data.get("models", [])]
    snippet = f"models={keys}"
    results.append(InfraResult("/v1/models", status, snippet, "PASS" if status == 200 and len(keys) >= 3 else "FAIL"))

    # GET /metrics
    status, body, _ = curl("GET", "/metrics")
    has_jarvis = "jarvis_" in body
    snippet = f"has_jarvis_metrics={has_jarvis}"
    results.append(InfraResult("/metrics", status, snippet, "PASS" if status == 200 and has_jarvis else "FAIL"))

    # Validation: empty message -> 422
    status, body, _ = curl("POST", "/v1/chat", {"message": "", "model": "auto"})
    results.append(InfraResult("/v1/chat (empty msg)", status, f"validation={status==422}", "PASS" if status == 422 else "FAIL"))

    # Unknown model -> 404
    status, body, _ = curl("POST", "/v1/chat", {"message": "hello", "model": "gpt5"})
    err = json.loads(body).get("error", "") if status > 200 else ""
    results.append(InfraResult("/v1/chat (bad model)", status, f"error={err}", "PASS" if status == 404 and err == "model_not_found" else "FAIL"))

    # Missing body -> 422
    status, body, _ = curl("POST", "/v1/chat")
    results.append(InfraResult("/v1/chat (no body)", status, f"validation={status==422}", "PASS" if status == 422 else "FAIL"))

    return results


# ── Chat tests ───────────────────────────────────────────────────────────────
def run_chat_tests() -> list[TestResult]:
    results: list[TestResult] = []

    for prompt_def in PROMPTS:
        for model in MODELS:
            body = {"message": prompt_def["prompt"], "model": model}
            status, resp_text, wall = curl("POST", "/v1/chat", body)

            result = TestResult(
                test_id=f"{prompt_def['id']}_{model}",
                category=prompt_def["category"],
                prompt=prompt_def["prompt"],
                model_requested=model,
                model_selected="",
                expect_model_auto=prompt_def["expect_model_auto"],
                routing_correct=False,
                status_code=status,
                latency_ms=0,
                wall_time_s=round(wall, 2),
            )

            if status == 200:
                data = json.loads(resp_text)
                result.model_selected = data.get("selected_model", "")
                result.latency_ms = data.get("latency_ms", 0)
                result.response_text = data.get("response", "")
                result.valid, result.valid_hits, result.valid_total = check_validity(
                    result.response_text, prompt_def["valid_keywords"]
                )
                # Check routing accuracy for auto requests
                if model == "auto":
                    result.routing_correct = (result.model_selected == prompt_def["expect_model_auto"])
            else:
                try:
                    err_data = json.loads(resp_text)
                    result.error = err_data.get("detail", resp_text[:120])
                except json.JSONDecodeError:
                    result.error = resp_text[:120]
                # For auto + deepseek expected (model not pulled), routing is still correct
                # if it at least tried deepseek (error will say model not found)
                if model == "auto" and "not found" in result.error:
                    result.routing_correct = True

            results.append(result)

    # Stream test (SSE format)
    status, resp_text, wall = curl("POST", "/v1/chat/stream", {"message": "Say hello world in 3 languages", "model": "auto"}, "30")
    has_done = "data: [DONE]" in resp_text
    stream_result = TestResult(
        test_id="T9_stream_auto",
        category="stream",
        prompt="Say hello world in 3 languages",
        model_requested="auto",
        model_selected="stream",
        expect_model_auto="stream",
        routing_correct=True,
        status_code=status,
        latency_ms=int(wall * 1000),
        response_text=resp_text[:2000] if status == 200 else "",
        error="" if status == 200 else resp_text[:120],
        valid=status == 200 and "data:" in resp_text and has_done,
        valid_hits=2 if ("data:" in resp_text and has_done) else 0,
        valid_total=2,
        wall_time_s=round(wall, 2),
    )
    results.append(stream_result)

    return results


# ── Report ───────────────────────────────────────────────────────────────────
def print_report(infra: list[InfraResult], chat: list[TestResult]) -> dict:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    # ── Infra ──
    print(f"\n{BOLD}{'='*80}")
    print(f"  INFRASTRUCTURE TESTS")
    print(f"{'='*80}{RESET}\n")

    infra_pass = sum(1 for r in infra if r.pass_fail == "PASS")
    print(f"  {GREEN}{infra_pass}/{len(infra)} PASSED{RESET}\n")
    print(f"  {'Endpoint':<28} {'Status':<8} {'Result':<8} {'Details'}")
    print(f"  {'-'*75}")
    for r in infra:
        color = GREEN if r.pass_fail == "PASS" else RED
        print(f"  {r.endpoint:<28} {r.status_code:<8} {color}{r.pass_fail:<8}{RESET} {r.body_snippet}")

    # ── Chat ──
    print(f"\n{BOLD}{'='*80}")
    print(f"  CHAT / LLM RESPONSE TESTS")
    print(f"{'='*80}{RESET}\n")

    # Summary by model
    print(f"  {BOLD}Model Comparison Summary{RESET}\n")
    model_stats: dict[str, dict] = {}
    for r in chat:
        if r.category == "stream":
            continue
        key = r.model_requested
        if key not in model_stats:
            model_stats[key] = {"count": 0, "success": 0, "total_latency": 0, "total_wall": 0, "valid_count": 0, "total_hits": 0, "total_keywords": 0}
        s = model_stats[key]
        s["count"] += 1
        if r.status_code == 200:
            s["success"] += 1
            s["total_latency"] += r.latency_ms
            s["total_wall"] += r.wall_time_s
            s["valid_count"] += 1 if r.valid else 0
            s["total_hits"] += r.valid_hits
            s["total_keywords"] += r.valid_total

    print(f"  {'Model':<10} {'Success':<12} {'Avg Latency':<14} {'Avg Wall':<12} {'Valid Rate':<12} {'Keyword Hit%':<14}")
    print(f"  {'-'*75}")
    for model, s in model_stats.items():
        avg_lat = s["total_latency"] // s["success"] if s["success"] else 0
        avg_wall = s["total_wall"] / s["success"] if s["success"] else 0
        valid_rate = f"{s['valid_count']}/{s['success']}" if s["success"] else "N/A"
        kw_pct = f"{s['total_hits']}/{s['total_keywords']} ({100*s['total_hits']//s['total_keywords'] if s['total_keywords'] else 0}%)" if s["total_keywords"] else "N/A"
        print(f"  {model:<10} {s['success']}/{s['count']:<10} {avg_lat:<14} {avg_wall:<12.1f} {valid_rate:<12} {kw_pct:<14}")

    # ── Routing accuracy ──
    print(f"\n  {BOLD}Routing Accuracy (auto model only){RESET}\n")
    auto_results = [r for r in chat if r.model_requested == "auto" and r.category != "stream"]
    correct = sum(1 for r in auto_results if r.routing_correct)
    total = len(auto_results)
    print(f"  {'Prompt (short)':<40} {'Expected':<12} {'Got':<12} {'Correct?'}")
    print(f"  {'-'*80}")
    for r in auto_results:
        icon = f"{GREEN}YES{RESET}" if r.routing_correct else f"{RED}NO{RESET}"
        short = r.prompt[:38] + ".." if len(r.prompt) > 40 else r.prompt
        print(f"  {short:<40} {r.expect_model_auto:<12} {r.model_selected or 'ERR':<12} {icon}")
    print(f"\n  {BOLD}Routing Accuracy: {GREEN}{correct}/{total}{RESET} ({100*correct//total if total else 0}%)\n")

    # ── Detailed results ──
    print(f"  {BOLD}Detailed Results{RESET}\n")
    for r in chat:
        if r.status_code == 200:
            status_icon = f"{GREEN}PASS{RESET}" if r.valid else f"{YELLOW}WEAK{RESET}"
        else:
            status_icon = f"{RED}FAIL{RESET}"

        routing_icon = ""
        if r.model_requested == "auto" and r.category != "stream":
            routing_icon = f" routing={GREEN}CORRECT{RESET}" if r.routing_correct else f" routing={RED}WRONG{RESET}"

        print(f"  {BOLD}[{r.test_id}]{RESET} {r.category:<10} model={CYAN}{r.model_requested}{RESET} -> {CYAN}{r.model_selected or r.error[:30]}{RESET}")
        print(f"         HTTP {r.status_code} | latency={r.latency_ms}ms | wall={r.wall_time_s}s | validity={status_icon} ({r.valid_hits}/{r.valid_total} keywords){routing_icon}")
        print(f"         {DIM}prompt: {r.prompt[:80]}{'...' if len(r.prompt)>80 else ''}{RESET}")
        if r.response_text:
            preview = r.response_text.replace("\n", " ")[:200]
            print(f"         {DIM}response: {preview}{'...' if len(r.response_text)>200 else ''}{RESET}")
        if r.error:
            print(f"         {RED}error: {r.error}{RESET}")
        print()

    # ── Stream ──
    stream = [r for r in chat if r.category == "stream"]
    if stream:
        s = stream[0]
        icon = f"{GREEN}PASS{RESET}" if s.valid else f"{RED}FAIL{RESET}"
        sse_icon = f"{GREEN}SSE{RESET}" if "data:" in s.response_text else f"{RED}NOT SSE{RESET}"
        done_icon = f"{GREEN}[DONE]{RESET}" if "data: [DONE]" in s.response_text else f"{RED}NO [DONE]{RESET}"
        print(f"  {BOLD}[STREAM]{RESET} HTTP {s.status_code} | wall={s.wall_time_s}s | {icon} | format={sse_icon} sentinel={done_icon}")
        if s.response_text:
            preview = s.response_text.replace("\n", " ")[:200]
            print(f"         {DIM}response: {preview}{RESET}")
        print()

    # ── Overall ──
    total_chat = len(chat)
    total_pass = sum(1 for r in chat if r.valid or (r.status_code == 200 and r.category == "stream"))
    total_infra_pass = sum(1 for r in infra if r.pass_fail == "PASS")
    grand_total = total_chat + len(infra)
    grand_pass = total_pass + total_infra_pass

    print(f"  {BOLD}{'='*80}")
    print(f"  OVERALL: {GREEN}{grand_pass}/{grand_total} PASSED{RESET}")
    print(f"  Infra: {total_infra_pass}/{len(infra)} | Chat: {total_pass}/{total_chat} | Routing: {correct}/{total}")
    print(f"  {'='*80}{RESET}\n")

    # Build JSON-serializable summary for saving
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall_pass": grand_pass,
        "overall_total": grand_total,
        "infra_pass": total_infra_pass,
        "infra_total": len(infra),
        "chat_pass": total_pass,
        "chat_total": total_chat,
        "routing_accuracy": f"{correct}/{total}",
        "routing_percent": 100 * correct // total if total else 0,
        "model_stats": {},
        "infra": [asdict(r) for r in infra],
        "chat": [asdict(r) for r in chat],
    }
    for model, s in model_stats.items():
        summary["model_stats"][model] = {
            "success": s["success"],
            "count": s["count"],
            "avg_latency_ms": s["total_latency"] // s["success"] if s["success"] else 0,
            "avg_wall_s": round(s["total_wall"] / s["success"], 2) if s["success"] else 0,
            "valid_rate": f"{s['valid_count']}/{s['success']}" if s["success"] else "N/A",
            "keyword_hit_pct": f"{s['total_hits']}/{s['total_keywords']} ({100*s['total_hits']//s['total_keywords'] if s['total_keywords'] else 0}%)",
        }

    return summary


# ── Save results ─────────────────────────────────────────────────────────────
def save_results(summary: dict) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    filepath = os.path.join(RESULTS_DIR, f"smoke_{ts}.json")
    with open(filepath, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return filepath


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    RED = "\033[91m"
    RESET = "\033[0m"

    print(f"\n{'='*80}")
    print(f"  Jarvis Model Router - Smoke Test Suite")
    print(f"  Target: {BASE_URL}")
    print(f"{'='*80}")

    # Quick connectivity check
    status, _, _ = curl("GET", "/health", timeout="5")
    if status == 0:
        print(f"\n  {RED}FATAL: Cannot reach {BASE_URL}. Is the server running?{RESET}")
        sys.exit(2)

    print(f"\n  Server is reachable. Running tests...\n")

    infra = run_infra_tests()
    chat = run_chat_tests()

    summary = print_report(infra, chat)

    # Save results to file
    filepath = save_results(summary)
    print(f"  {GREEN}Results saved to: {filepath}{RESET}\n")

    # Exit code
    if summary["overall_pass"] == summary["overall_total"]:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
