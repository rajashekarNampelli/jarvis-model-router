# Jarvis Model Router

A FastAPI service that routes chat requests to Ollama-backed models using an LLM-based classifier, with streaming, Prometheus metrics, health checks, and structured logging.

## Features

- Chat requests (`POST /v1/chat`)
- LLM-based model routing (a small classifier model picks `qwen` / `deepseek` / `llama`)
- In-process LRU cache for routing decisions (repeated prompts pay zero classifier cost)
- Streaming responses (`POST /v1/chat/stream`)
- Ollama integration with circuit breaker + error isolation
- Prometheus metrics (`GET /metrics`)
- Health checks (`GET /health`)

## Models

| Key | Ollama Model | Use Case |
|-----|-------------|----------|
| `llama` | llama3 | General purpose (default) |
| `qwen` | qwen2.5-coder | Code questions |
| `deepseek` | deepseek-r1:8b | Reasoning / math |

## Prerequisites

- Python 3.14+
- [Ollama](https://ollama.ai) installed locally

## Setup & Run

### 1. Install and start Ollama

```bash
# Install Ollama from https://ollama.ai, then:
ollama serve

# Pull required models (in a separate terminal)
ollama pull llama3
ollama pull qwen2.5-coder
ollama pull deepseek-r1:8b
```

### 2. Set up the Python environment

```bash
cd jarvis-model-router
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env if needed (defaults work for local Ollama)
```

### 4. Run the API server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API is now available at `http://localhost:8000`.

## API Reference

### POST /v1/chat

```bash
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Write a Java unit test", "model": "auto", "stream": false}'
```

Response:
```json
{
  "request_id": "abc123",
  "selected_model": "qwen",
  "response": "...",
  "latency_ms": 420
}
```

### POST /v1/chat/stream

```bash
curl -N -X POST http://localhost:8000/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain quicksort", "model": "auto", "stream": true}'
```

### GET /health

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "UP",
  "ollama": "UP",
  "memory": {"used_mb": 1024, "total_mb": 16384, "percent": 6.2},
  "gpu": "unavailable"
}
```

### GET /metrics

```bash
curl http://localhost:8000/metrics
```

Prometheus exposition format.

### GET /v1/models

```bash
curl http://localhost:8000/v1/models
```

### Model Selection

Pass `"model": "auto"` to let the router decide, or specify a key directly:

- `"model": "auto"` — LLM classifier routes to `qwen` / `deepseek` / `llama`
- `"model": "qwen"` — force Qwen (coding)
- `"model": "deepseek"` — force DeepSeek (reasoning)
- `"model": "llama"` — force Llama (general)

The classifier model and timeout are configurable via env vars
(`CLASSIFIER_MODEL`, `CLASSIFIER_TIMEOUT`). On any classifier failure the
router falls back to `llama` so the request still succeeds.

## Running Tests

```bash
pytest app/tests/ -v
```

## Project Structure

```
app/
├── api/            # FastAPI route handlers
├── core/           # Config, logging, exceptions
├── schemas/        # Pydantic request/response models
├── services/       # Router and inference orchestration
├── providers/      # LLM provider abstractions
├── routing/        # LLM classifier + model registry
├── metrics/        # Prometheus instrumentation
├── middleware/      # Request logging middleware
└── tests/          # pytest test suite
```
