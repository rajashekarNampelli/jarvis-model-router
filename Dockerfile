# ── Stage 1: dependency builder ──────────────────────────────────────────────
FROM python:3.13-slim AS builder

WORKDIR /build

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: runtime image ───────────────────────────────────────────────────
FROM python:3.13-slim

RUN groupadd -r jarvis && useradd -r -g jarvis -d /app -s /sbin/nologin jarvis

WORKDIR /app

COPY --from=builder /install /usr/local
COPY jarvis_model_router/ ./jarvis_model_router/

RUN chown -R jarvis:jarvis /app

USER jarvis

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c \
    "import httpx; \
     r = httpx.get('http://localhost:8000/health', timeout=8); \
     exit(0 if r.status_code == 200 else 1)"

CMD ["uvicorn", "jarvis_model_router.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
