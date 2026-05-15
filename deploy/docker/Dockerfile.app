FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY pyproject.toml .

# Make pip resilient to slow/unstable PyPI connections
ENV PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=10

# Lightweight deps first (cached layer)
RUN pip install --no-cache-dir \
    pydantic pydantic-settings structlog pyyaml numpy scikit-learn \
    jinja2 opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp \
    prometheus-client httpx redis pytest pytest-cov pytest-asyncio hypothesis \
    ruff

# Heavy deps in their own layers so a retry doesn't redo the whole install.
# Ray version MUST match the cluster image (rayproject/ray:2.55.0-py311) —
# the Ray client refuses to connect across mismatched versions.
RUN pip install --no-cache-dir "ray[default]==2.55.0"

# Install CPU-only torch from PyTorch's CDN (smaller + faster than the default
# CUDA wheel from PyPI). sentence-transformers will then reuse this torch.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    torch
RUN pip install --no-cache-dir sentence-transformers

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
