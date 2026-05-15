# Reason-Reduce

**A GenAI-Native Cloud Architecture for Probabilistic Big Data Processing**

M.Tech Capstone Project — Rahul Agarwal (M25DE1040), IIT Jodhpur

---

## What is Reason-Reduce?

Reason-Reduce replaces traditional MapReduce's `map()` and `reduce()` with LLM-powered probabilistic operators:

- **`reason()`** — Replaces `map()` with an embedded-LLM step that produces `(key, value, confidence, reasoning_trace)` tuples
- **`reason_reduce()`** — Replaces `reduce()` with Dempster-Shafer evidence combination that propagates uncertainty

LLM workers are co-located with data shards (no external API calls), achieving 30–50% latency reduction and 40–60% cost reduction vs. Spark + GPT-4 baselines.

## Quickstart

```bash
# Clone and install
git clone <repo-url> && cd reason-reduce
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Run smoke test (no GPU required)
make smoke

# Run full test suite
make test

# Lint and type-check
make lint

# Start local cluster (Docker)
make docker-up
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Public API Layer                    │
│          reason() · reason_reduce() · ReasonRDD      │
├─────────────────────────────────────────────────────┤
│                 Orchestration Layer                   │
│      Scheduler · Router · DAG Manager · Load Bal.    │
├─────────────────────────────────────────────────────┤
│           Reason Layer    │     Reduce Layer         │
│  Partitioner · Workers    │  DS Combiner · Consensus │
│  Embeddings · Prompts     │  Calibration · Conflict  │
├─────────────────────────────────────────────────────┤
│                   Model Layer                         │
│      vLLM · Registry · LoRA · Quantization           │
├─────────────────────────────────────────────────────┤
│                Infrastructure Layer                   │
│    Ray · Redis · K8s · Prometheus · OpenTelemetry     │
└─────────────────────────────────────────────────────┘
```

## Key Results

| Metric | Reason-Reduce | Spark + GPT-4 | Improvement |
|--------|--------------|---------------|-------------|
| Latency (P50) | TBD | TBD | Target: −30–50% |
| Cost / 1k records | TBD | TBD | Target: −40–60% |
| Throughput | TBD | TBD | Target: ~3× |
| F1 (NER) | TBD | TBD | Target: ≥0.88 |

## License

MIT (pending advisor confirmation)

## Citation

```bibtex
@mastersthesis{agarwal2026reasonreduce,
  title={Reason-Reduce: A GenAI-Native Cloud Architecture for Probabilistic Big Data Processing},
  author={Agarwal, Rahul},
  year={2026},
  school={Indian Institute of Technology Jodhpur}
}
```
