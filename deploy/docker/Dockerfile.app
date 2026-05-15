FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY pyproject.toml .
RUN pip install --no-cache-dir \
    pydantic pydantic-settings structlog pyyaml numpy scikit-learn \
    jinja2 opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp \
    prometheus-client httpx sentence-transformers \
    "ray[default]>=2.9" redis pytest pytest-cov pytest-asyncio hypothesis \
    ruff

# Copy source
COPY src/ src/
COPY tests/ tests/
COPY scripts/ scripts/
COPY benchmarks/ benchmarks/
COPY evaluation/ evaluation/
COPY docs/ docs/
COPY Makefile .

ENV PYTHONPATH=/app/src
ENV RAY_ADDRESS=ray://ray-head:10001
ENV REDIS_URL=redis://redis:6379/0

# Default: drop into shell
CMD ["/bin/bash"]
