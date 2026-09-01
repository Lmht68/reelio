FROM ghcr.io/astral-sh/uv:0.12.1 AS uv
FROM python:3.12.12-slim-bookworm AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY --from=uv /uv /uvx /bin/
COPY pyproject.toml uv.lock README.md ./

RUN uv sync --locked --no-dev --no-install-project

COPY src ./src

RUN uv sync --locked --no-dev --no-editable

FROM python:3.12.12-slim-bookworm AS runtime

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    REELIO_TEMP_MEDIA_DIR=/tmp/reelio \
    REELIO_WHISPER_DEVICE=cpu \
    REELIO_WHISPER_COMPUTE_TYPE=int8 \
    HF_HOME=/var/cache/huggingface

RUN groupadd --system --gid 10001 reelio \
    && useradd --system --uid 10001 --gid reelio --home-dir /app --shell /usr/sbin/nologin reelio \
    && mkdir -p /tmp/reelio /var/cache/huggingface \
    && chown -R reelio:reelio /tmp/reelio /var/cache/huggingface

WORKDIR /app

COPY --from=builder --chown=reelio:reelio /app/.venv /app/.venv

USER reelio

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5m --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["uvicorn", "reelio.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
