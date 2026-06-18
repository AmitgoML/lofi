FROM python:3.12-slim@sha256:39e4e1ccb01578e3c86f7a0cf7b7fd89b8dbe2c27a88de11cf726ba669469f49

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # App Runner will set PORT at runtime; default to 8000 for local runs
    PORT=8000 \
    # prevent uv from hardlinking when Docker layers are involved
    UV_LINK_MODE=copy

WORKDIR /app

# ---- install uv (pinned, no apt-get required) ----
COPY --from=ghcr.io/astral-sh/uv:0.7.12@sha256:4faec156e35a5f345d57804d8858c6ba1cf6352ce5f4bffc11b7fdebdef46a38 /uv /uvx /usr/local/bin/

# create a non-root user
RUN useradd -u 10001 -m appuser && chown appuser:appuser /app
USER appuser

# ---- dependency layer (best cache) ----
COPY --chown=appuser:appuser pyproject.toml uv.lock /app/
RUN uv sync --frozen --no-dev

# Add venv to PATH for all subsequent commands/layers
ENV PATH="/app/.venv/bin:${PATH}"

# ---- app code ----
COPY --chown=appuser:appuser . /app

# Install Zyte CA certificate into certifi bundle (inside venv)
RUN python scripts/install_zyte_cert.py

# (informational) container port
EXPOSE 8000

# start FastAPI with uvicorn; use exec form via entrypoint for signal handling
ENTRYPOINT ["/app/docker-entrypoint.sh"]
