# Reason-Reduce: Complete Setup & Running Guide

> **Platform:** MacBook Pro M4 (Apple Silicon)
> **LLM Inference:** Mock adapter for dev; MLX/llama.cpp for local; vLLM on HPC for benchmarks
> **Key constraint:** vLLM requires NVIDIA CUDA — use alternatives on Mac

This document covers everything needed to run the Reason-Reduce project from scratch — prerequisites, installation, dataset preparation, running the pipeline, benchmarks, and deployment.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Dataset Preparation](#dataset-preparation)
5. [Running the Pipeline](#running-the-pipeline)
6. [Testing](#testing)
7. [Benchmarking](#benchmarking)
8. [Local Cluster (Docker)](#local-cluster-docker)
9. [Kubernetes Deployment](#kubernetes-deployment)
10. [Monitoring & Observability](#monitoring--observability)
11. [Troubleshooting](#troubleshooting)
12. [Command Reference](#command-reference)

---

## Prerequisites

### Target Hardware

**MacBook Pro M4** — Apple Silicon (ARM64)

| Spec | Recommended |
|------|-------------|
| Chip | M4 / M4 Pro / M4 Max |
| RAM | 16 GB minimum (24–36 GB preferred for 7B models) |
| Storage | 20 GB free (models + datasets) |
| macOS | 15.0+ (Sequoia) |
| GPU | Unified memory — no discrete GPU needed |

> **Note:** Apple Silicon uses unified memory shared between CPU and GPU. A 7B quantized model needs ~5 GB. With 16 GB RAM, you can run inference + the pipeline. With 36+ GB, you can run 13B models.

### Software Requirements

```bash
# Required
python >= 3.11
git
docker desktop for mac    # for local cluster (optional)

# Recommended
uv                         # fast Python package manager
homebrew                   # macOS package manager
```

### Install System Dependencies (macOS Apple Silicon)

```bash
# Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Python 3.11+ via pyenv
brew install pyenv
pyenv install 3.11.7
pyenv local 3.11.7

# uv package manager (10-100x faster than pip)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Docker Desktop (for local Ray cluster — optional)
brew install --cask docker

# Redis (local, for embedding cache)
brew install redis

# Kind + kubectl (for K8s — optional)
brew install kind kubectl
```

---

## Installation

### Step 1: Clone the Repository

```bash
git clone <repo-url> reason-reduce
cd reason-reduce
```

### Step 2: Create Virtual Environment

```bash
# Using uv (recommended)
uv venv --python 3.11
source .venv/bin/activate

# OR using standard venv
python3.11 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Core + dev dependencies (no GPU needed)
uv pip install -e ".[dev]"

# With training dependencies (LoRA, quantization)
uv pip install -e ".[dev,train]"

# With evaluation dependencies (Spark baseline, plotting)
uv pip install -e ".[dev,eval]"

# Everything
uv pip install -e ".[dev,train,eval]"
```

### Step 4: Install Optional Heavy Dependencies

```bash
# Ray (required for distributed mode)
uv pip install "ray[default]>=2.9"

# Sentence-BERT (required for real semantic partitioning)
uv pip install sentence-transformers

# For LLM inference on Apple Silicon, use MLX (NOT vllm — vLLM requires CUDA)
uv pip install mlx mlx-lm

# UMAP + HDBSCAN (for HDBSCAN partitioning strategy)
uv pip install umap-learn hdbscan
```

> **Apple Silicon Note:** vLLM does NOT support macOS/MPS. For local M4 inference,
> use `mlx-lm` (Apple's native ML framework) or `llama-cpp-python` with Metal backend.
> The mock adapter is used for development/testing. Real vLLM runs on the IIT HPC cluster.

### Step 5: Verify Installation

```bash
# Quick check — should print version
python -c "import reason_reduce; print(reason_reduce.__version__)"

# Smoke test — full pipeline with mock LLM, no GPU
make smoke
```

Expected output:
```
smoke_test_passed  elapsed_seconds=4.3  n_reason_outputs=10  n_reduce_outputs=1
```

---

## Configuration

### Environment Variables

```bash
# Copy the example and fill in values
cp .env.example .env
```

Key variables in `.env`:

```bash
# Ray cluster
RAY_ADDRESS=auto              # "auto" for local, or "ray://host:10001" for remote
RAY_NUM_WORKERS=4

# Redis (for embedding cache + coordination)
REDIS_URL=redis://localhost:6379/0

# vLLM (for real inference)
VLLM_MODEL_ID=mistralai/Mistral-7B-Instruct-v0.3
VLLM_GPU_MEMORY_UTILIZATION=0.85
VLLM_MAX_MODEL_LEN=4096

# Weights & Biases (experiment tracking)
WANDB_PROJECT=reason-reduce
WANDB_ENTITY=your-username

# OpenTelemetry (tracing)
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_SERVICE_NAME=reason-reduce

# HuggingFace (model downloads)
HF_TOKEN=your-token-here
```

### Thresholds Configuration

Edit `src/reason_reduce/config/thresholds.yaml`:

```yaml
tau_confidence: 0.7      # Min confidence to accept result
tau_entropy: 0.5         # Max entropy before escalation
tau_consensus: 0.6       # Min consensus across workers
tau_conflict_low: 0.3    # DS conflict: normal below this
tau_conflict_high: 0.7   # DS conflict: escalate above this
max_workers: 4           # Default worker count
max_retries: 2           # Retries on parse failure
```

### Model Registry

Edit `src/reason_reduce/config/models.yaml` to add/modify models:

```yaml
models:
  - name: mistral-7b
    hf_id: mistralai/Mistral-7B-Instruct-v0.3
    vram_gb: 5.0
    ctx_len: 8192
    quantization: awq
    cost_per_1k: 0.02
```

---

## Dataset Preparation

### Quick Start (Sample Data)

```bash
# Creates small placeholder datasets for testing
bash scripts/download_datasets.sh
```

### PubMed Abstracts (Primary Dataset)

```bash
# Download 1M PubMed abstracts (baseline subset)
mkdir -p benchmarks/datasets/pubmed

# Option A: Use PubMed FTP (free, no API key)
wget -O benchmarks/datasets/pubmed/pubmed_abstracts.jsonl.gz \
  "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/pubmed24n0001.xml.gz"

# Option B: Use the Hugging Face dataset (easier)
python -c "
from datasets import load_dataset
ds = load_dataset('ccdv/pubmed-summarization', split='train[:10000]')
import json
with open('benchmarks/datasets/pubmed/pubmed_10k.jsonl', 'w') as f:
    for i, row in enumerate(ds):
        f.write(json.dumps({'id': str(i), 'text': row['article']}) + '\n')
print(f'Saved {len(ds)} abstracts')
"

# Verify
wc -l benchmarks/datasets/pubmed/pubmed_10k.jsonl
# Expected: 10000
```

### SEC 10-K Filings

```bash
mkdir -p benchmarks/datasets/sec

# Download SEC EDGAR filings (requires sec-edgar-downloader)
uv pip install sec-edgar-downloader

python -c "
from sec_edgar_downloader import Downloader
import json, os, glob

dl = Downloader('MyCompany', 'email@example.com', 'benchmarks/datasets/sec/raw')
tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']
for ticker in tickers:
    dl.get('10-K', ticker, limit=5)

# Convert to JSONL
docs = []
for f in glob.glob('benchmarks/datasets/sec/raw/**/*.txt', recursive=True):
    text = open(f).read()[:10000]
    docs.append({'id': os.path.basename(f), 'text': text})

with open('benchmarks/datasets/sec/sec_filings.jsonl', 'w') as f:
    for doc in docs:
        f.write(json.dumps(doc) + '\n')
print(f'Saved {len(docs)} filings')
"
```

### Common Crawl Subset

```bash
mkdir -p benchmarks/datasets/common_crawl

# Download a small CC subset via HuggingFace
python -c "
from datasets import load_dataset
import json

ds = load_dataset('allenai/c4', 'en', split='train', streaming=True)
docs = []
for i, row in enumerate(ds):
    if i >= 10000:
        break
    docs.append({'id': str(i), 'text': row['text'][:2000]})

with open('benchmarks/datasets/common_crawl/cc_10k.jsonl', 'w') as f:
    for doc in docs:
        f.write(json.dumps(doc) + '\n')
print(f'Saved {len(docs)} documents')
"
```

### Synthetic Stress Test Data

```bash
# Generate synthetic data for stress testing (no download needed)
python -c "
import json, random, string

random.seed(42)
with open('benchmarks/datasets/synthetic/stress_100k.jsonl', 'w') as f:
    for i in range(100000):
        text = ' '.join(random.choices(string.ascii_lowercase.split(), k=random.randint(50, 500)))
        f.write(json.dumps({'id': str(i), 'text': text}) + '\n')
print('Generated 100k synthetic documents')
"
```

### Dataset Versioning with DVC

```bash
# Initialize DVC (one-time)
dvc init
dvc remote add -d storage s3://your-bucket/reason-reduce-data  # or local path

# Track datasets
dvc add benchmarks/datasets/pubmed/pubmed_10k.jsonl
dvc add benchmarks/datasets/sec/sec_filings.jsonl
dvc add benchmarks/datasets/common_crawl/cc_10k.jsonl

# Push to remote storage
git add benchmarks/datasets/*.dvc .gitignore
git commit -m "[data] Track benchmark datasets via DVC"
dvc push
```

---

## Running the Pipeline

### Mode 1: Mock LLM (No GPU, Development)

```bash
# Smoke test — 10 docs, mock LLM, 2 partitions
make smoke

# Custom run
PYTHONPATH=src python -c "
from reason_reduce.ingestion.batch import load_documents, Doc
from reason_reduce.reason.api import reason
from reason_reduce.reason.worker import TaskSpec
from reason_reduce.reduce.api import reason_reduce

# Load documents
docs = [Doc(id=str(i), text=f'Sample document {i}') for i in range(100)]

# Run reason() with mock LLM
results = reason(docs, task=TaskSpec(task_type='ner'), model='mock', n_partitions=4, seed=42)
print(f'Reason outputs: {len(results)}')

# Run reason_reduce()
consensus = reason_reduce(results, strategy='ds', seed=42)
print(f'Consensus results: {len(consensus)}')
for r in consensus[:5]:
    print(f'  {r.key}: {r.value} (confidence={r.confidence:.3f})')
"
```

### Mode 2: Local LLM on Apple Silicon (MLX)

```bash
# Install MLX for Apple Silicon native inference
uv pip install mlx mlx-lm

# Download a quantized model (4-bit, fits in 5 GB)
python -c "from mlx_lm import load; load('mlx-community/Mistral-7B-Instruct-v0.3-4bit')"

# Run with MLX adapter (when implemented)
# For now, the mock adapter validates the full pipeline.
# Real inference happens on the IIT HPC cluster with vLLM + NVIDIA GPUs.

# Alternative: llama.cpp with Metal acceleration
uv pip install llama-cpp-python
# Download GGUF model
huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF \
  mistral-7b-instruct-v0.2.Q4_K_M.gguf --local-dir models/

# Quick test (llama.cpp on M4 — expect ~30 tokens/sec)
python -c "
from llama_cpp import Llama
llm = Llama(model_path='models/mistral-7b-instruct-v0.2.Q4_K_M.gguf', n_gpu_layers=-1)
output = llm('Extract entities from: Aspirin treats headaches.', max_tokens=100)
print(output['choices'][0]['text'])
"
```

> **Performance on M4:**
> - Mistral-7B 4-bit: ~30-40 tokens/sec (Metal GPU)
> - Embedding (SBERT): ~200 docs/sec
> - Full pipeline (100 docs, mock LLM): <1 second
> - Full pipeline (100 docs, MLX/llama.cpp): ~3-5 minutes

### Mode 3: Distributed (Ray Cluster)

```bash
# Start local Ray cluster
ray start --head --port=6379 --num-cpus=8

# Run distributed
PYTHONPATH=src python -c "
from reason_reduce.ingestion.batch import load_documents
from reason_reduce.reason.api import reason
from reason_reduce.reason.worker import TaskSpec
from reason_reduce.reduce.api import reason_reduce

docs = load_documents('benchmarks/datasets/pubmed/pubmed_10k.jsonl', max_docs=1000)
results = reason(
    docs,
    task=TaskSpec(task_type='ner'),
    model='mock',
    n_partitions=8,
    distributed=True,
    n_workers=4,
)
consensus = reason_reduce(results, strategy='ds')
print(f'{len(docs)} docs → {len(results)} reason outputs → {len(consensus)} consensus results')
"

# Stop Ray when done
ray stop
```

---

## Testing

### Unit Tests (Fast, No Dependencies)

```bash
make test
# Runs: pytest tests/unit with coverage report
# Expected: ~38 tests, <10s
```

### Integration Tests

```bash
make test-integration
# Runs: pytest tests/integration
# Includes: full pipeline, Ray distributed (skipped if Ray not installed)
```

### Full Test Suite

```bash
make test-all
# Runs: all tests including e2e
# Generates: HTML coverage report in htmlcov/
```

### Individual Test Files

```bash
PYTHONPATH=src python -m pytest tests/unit/test_dempster_shafer.py -v
PYTHONPATH=src python -m pytest tests/unit/test_partitioner.py -v
PYTHONPATH=src python -m pytest tests/unit/test_registry.py -v
PYTHONPATH=src python -m pytest tests/integration/test_reason_e2e.py -v
```

### Linting

```bash
make lint
# Runs: ruff check + ruff format --check
```

### Auto-format

```bash
make format
# Runs: ruff check --fix + ruff format
```

---

## Benchmarking

### Run the Main Benchmark

```bash
# Full benchmark: Reason-Reduce vs Spark+GPT-4
PYTHONPATH=src python benchmarks/run_benchmark.py \
  --dataset pubmed \
  --task ner \
  --n-runs 5 \
  --cluster-sizes 1,4,8,16 \
  --seed 42

# Results saved to: evaluation/results/<experiment_hash>.parquet
```

### Run Ablation Studies

```bash
# Partitioning strategy ablation
PYTHONPATH=src python -c "
from reason_reduce.ingestion.batch import load_documents
from reason_reduce.reason.partitioner import partition_documents

docs = load_documents('benchmarks/datasets/pubmed/pubmed_10k.jsonl', max_docs=1000)

for strategy in ['random', 'semantic', 'spectral']:
    partitions = partition_documents(docs, n_partitions=10, strategy=strategy, seed=42)
    coherences = [p.coherence_score for p in partitions]
    mean_c = sum(coherences) / len(coherences)
    print(f'{strategy:12s}: mean_coherence={mean_c:.4f}')
"

# Confidence threshold sweep
PYTHONPATH=src python evaluation/ablations/threshold_sweep.py \
  --thresholds 0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9
```

### Scalability Study

```bash
# Strong scaling: fixed 10k input, varying node count
PYTHONPATH=src python evaluation/scalability.py \
  --type strong \
  --input-size 10000 \
  --node-counts 1,4,8,16
```

---

## Local Cluster (Docker)

### Start All Services

```bash
make docker-up
# Starts: Ray head + 2 workers + Redis + Prometheus + Grafana
```

### Service Endpoints

| Service | URL | Credentials |
|---------|-----|-------------|
| Ray Dashboard | http://localhost:8265 | — |
| Redis | localhost:6380 | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | admin / admin |

### Run Pipeline Against Docker Cluster

```bash
# Set environment to point to Docker services
export RAY_ADDRESS=ray://localhost:10001
export REDIS_URL=redis://localhost:6380/0

# Run
PYTHONPATH=src python scripts/smoke.py
```

### Stop All Services

```bash
make docker-down
```

---

## Kubernetes Deployment

### Local (Kind)

```bash
# Create cluster
make k8s-local
# This runs: kind create cluster && kubectl apply -k deploy/k8s/

# Verify
kubectl get pods -w
# Wait for all pods to be Running

# Port-forward Ray dashboard
kubectl port-forward svc/ray-head 8265:8265

# Run benchmark against K8s cluster
export RAY_ADDRESS=ray://localhost:10001
PYTHONPATH=src python benchmarks/run_benchmark.py --dataset pubmed --task ner
```

### Production (GKE / IIT HPC)

```bash
# Apply the production overlay
kubectl apply -k deploy/k8s/overlays/production

# Scale workers
kubectl scale raycluster reason-reduce-cluster --replicas=16
```

### Teardown

```bash
make k8s-teardown
```

---

## Monitoring & Observability

### Structured Logs

All logs are JSON via structlog. View in real-time:

```bash
# During a run, logs appear as structured JSON
PYTHONPATH=src python scripts/smoke.py 2>&1 | python -m json.tool
```

### Prometheus Metrics

Available at `http://localhost:9090` when Docker cluster is running:

- `reason_reduce_reason_requests_total` — total reason() calls by model/task/status
- `reason_reduce_reason_latency_seconds` — latency distribution
- `reason_reduce_confidence` — confidence score distribution
- `reason_reduce_conflict_mass` — DS conflict mass during reduce
- `reason_reduce_active_workers` — current active worker count
- `reason_reduce_model_routing_total` — routing decisions

### Grafana Dashboard

Pre-built at `http://localhost:3000`:

1. Open Grafana → Dashboards → Reason-Reduce
2. Panels: throughput, latency P50/P95/P99, confidence distribution, conflict rate

### Weights & Biases

```bash
# Login (one-time)
wandb login

# Runs are auto-logged during benchmarks
# View at: https://wandb.ai/<entity>/reason-reduce
```

### OpenTelemetry Traces

```bash
# Start Jaeger for local trace viewing
docker run -d --name jaeger \
  -p 16686:16686 -p 4317:4317 \
  jaegertracing/all-in-one:latest

# Set OTEL endpoint
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317

# Run pipeline — traces appear at http://localhost:16686
PYTHONPATH=src python scripts/smoke.py
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'reason_reduce'"

```bash
# Ensure you're in the project root with PYTHONPATH set
export PYTHONPATH=src
# OR install in editable mode
uv pip install -e .
```

### "No module named 'ray'"

```bash
uv pip install "ray[default]>=2.9"
```

### "No module named 'sentence_transformers'"

```bash
# The partitioner will use TF-IDF fallback without this
# For real SBERT embeddings:
uv pip install sentence-transformers
```

### "CUDA out of memory" / GPU Issues on Mac

```bash
# You DON'T have CUDA on macOS. Apple Silicon uses Metal/MPS.
# vLLM will NOT work on Mac — it requires NVIDIA CUDA.
# Use one of these alternatives:

# Option A: Mock adapter (development — validates full pipeline)
model="mock"

# Option B: MLX (Apple-native, fast on M4)
uv pip install mlx mlx-lm

# Option C: llama.cpp with Metal backend
uv pip install llama-cpp-python
# Set n_gpu_layers=-1 to use Metal acceleration

# Option D: Run on IIT HPC cluster (real benchmarks)
export RAY_ADDRESS=ray://<hpc-cluster-ip>:10001
```

### Memory pressure on Mac (swap usage high)

```bash
# Check memory
vm_stat | head -5

# Reduce model size — use 4-bit quantization
# Mistral-7B-4bit needs ~5 GB
# If you have 16 GB total, close other apps during inference

# Reduce dataset size for local testing
# Use max_docs=50 instead of 1000

# Monitor during runs
top -l 1 | grep PhysMem
```

### "Ray cluster not connecting"

```bash
# Check Ray status
ray status

# Restart Ray
ray stop
ray start --head --port=6379 --num-cpus=8

# Verify
python -c "import ray; ray.init(); print(ray.cluster_resources())"
```

### "Tests taking too long"

```bash
# Skip integration tests
PYTHONPATH=src python -m pytest tests/unit -x -q

# Run a single test file
PYTHONPATH=src python -m pytest tests/unit/test_dempster_shafer.py -v
```

### Smoke test exceeds 5s

```bash
# This usually means sklearn is slow on first import
# Run once to warm up, second run will be faster
make smoke && make smoke
```

---

## Command Reference

| Command | Description |
|---------|-------------|
| `make smoke` | Run 10-doc pipeline with mock LLM (<5s, no GPU) |
| `make test` | Unit tests with coverage |
| `make test-all` | Full test suite (unit + integration + e2e) |
| `make test-integration` | Integration tests only |
| `make lint` | Check code with ruff |
| `make format` | Auto-format code |
| `make bench` | Run main benchmark |
| `make docker-up` | Start local Docker cluster |
| `make docker-down` | Stop Docker cluster |
| `make k8s-local` | Deploy to local Kind cluster |
| `make k8s-teardown` | Delete Kind cluster |
| `make reproduce` | Reproduce thesis headline numbers |
| `make clean` | Remove caches and build artifacts |
| `make install` | Install core + dev dependencies |
| `make dev` | Install all dependencies (dev + train + eval) |

---

## End-to-End Workflow (Day 1 → Results on MacBook Pro M4)

```bash
# 1. Setup
git clone <repo> && cd reason-reduce
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e ".[dev]"

# 2. Verify (no GPU needed — runs in <5 seconds)
make smoke   # Must pass

# 3. Install real ML deps for M4
uv pip install "ray[default]" sentence-transformers mlx mlx-lm

# 4. Download data
bash scripts/download_datasets.sh          # Placeholders first
# Then follow "Dataset Preparation" section for real PubMed/SEC data

# 5. Run pipeline with mock LLM (validates full plumbing)
PYTHONPATH=src python scripts/smoke.py

# 6. Run with real SBERT embeddings (partitioning ablation on M4)
PYTHONPATH=src python -c "
from reason_reduce.ingestion.batch import load_documents, Doc
from reason_reduce.reason.partitioner import partition_documents

docs = [Doc(id=str(i), text=f'Medical document about topic {i % 5}') for i in range(100)]
for strategy in ['random', 'semantic', 'spectral']:
    partitions = partition_documents(docs, n_partitions=5, strategy=strategy, seed=42)
    mean_c = sum(p.coherence_score for p in partitions) / len(partitions)
    print(f'{strategy:12s}: coherence={mean_c:.4f}')
"

# 7. Run distributed (Ray local, simulated 4 workers)
PYTHONPATH=src python -c "
from reason_reduce.ingestion.batch import Doc
from reason_reduce.reason.api import reason
from reason_reduce.reason.worker import TaskSpec
from reason_reduce.reduce.api import reason_reduce

docs = [Doc(id=str(i), text=f'Patient {i} received aspirin.') for i in range(50)]
results = reason(docs, task=TaskSpec(task_type='ner'), model='mock',
                 n_partitions=4, distributed=True, n_workers=4)
consensus = reason_reduce(results, strategy='ds')
print(f'{len(docs)} docs → {len(results)} outputs → {len(consensus)} consensus')
"

# 8. (Optional) Local LLM via llama.cpp on M4
uv pip install llama-cpp-python
huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF \
  mistral-7b-instruct-v0.2.Q4_K_M.gguf --local-dir models/
# ~5 GB download, runs at ~30 tok/sec on M4

# 9. For REAL benchmarks → deploy to IIT HPC cluster with NVIDIA GPUs
# See "Kubernetes Deployment" section

# 10. Reproduce results
bash scripts/reproduce_results.sh
```

### What Works on MacBook Pro M4 (No External Cluster)

| Feature | Status | Notes |
|---------|--------|-------|
| Full pipeline (mock LLM) | Works | <5 seconds |
| Semantic partitioning (SBERT) | Works | ~200 docs/sec on M4 |
| Dempster-Shafer aggregation | Works | Pure Python + NumPy |
| Ray distributed (local) | Works | Simulates multi-node |
| Unit + integration tests | Works | 45 tests in ~7s |
| LLM inference (mlx/llama.cpp) | Works | ~30-40 tok/s (7B-4bit) |
| LLM inference (vLLM) | Does NOT work | Requires NVIDIA CUDA |
| Full benchmark (multi-GPU) | Needs HPC | IIT Jodhpur cluster |

---

## Reproducibility Checklist

Every benchmark run automatically records:

- Git commit SHA
- Dataset SHA-256 hash
- Configuration hash (thresholds + model registry)
- Random seed
- Hardware fingerprint (CUDA device, RAM, node count)
- W&B run URL

To reproduce any result:

```bash
# Find the run in W&B or evaluation/results/*.parquet
# Check the commit SHA
git checkout <sha>

# Re-run with same seed
PYTHONPATH=src python benchmarks/run_benchmark.py --seed 42 --config <config_hash>
```
