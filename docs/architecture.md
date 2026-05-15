# Reason-Reduce Architecture

## 5-Layer Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Layer 5: API                       │
│          reason() · reason_reduce() · ReasonRDD      │
│                                                      │
│   Spark-RDD-compatible interface for users.          │
│   Lazy evaluation, type-safe, reproducible.          │
├─────────────────────────────────────────────────────┤
│                Layer 4: Orchestration                 │
│      Scheduler · Router · DAG Manager · Load Bal.    │
│                                                      │
│   Cost-aware routing, adaptive scheduling,           │
│   straggler mitigation, pipeline DAG execution.      │
├─────────────────────────────────────────────────────┤
│       Layer 3: Reason     │    Layer 3: Reduce       │
│  Partitioner · Workers    │  DS Combiner · Consensus │
│  Embeddings · Prompts     │  Calibration · Conflict  │
│                           │                          │
│  Semantic partitioning,   │  Dempster-Shafer,        │
│  CoT reasoning, conf.     │  Platt calibration,      │
│  scoring via logprobs     │  tree reduction           │
├─────────────────────────────────────────────────────┤
│                  Layer 2: Models                      │
│      vLLM · Registry · LoRA · Quantization           │
│                                                      │
│   Co-located inference, PagedAttention,              │
│   continuous batching, hot-swappable adapters.       │
├─────────────────────────────────────────────────────┤
│               Layer 1: Infrastructure                │
│    Ray · Redis · K8s · Prometheus · OpenTelemetry     │
│                                                      │
│   Distributed compute, state management,             │
│   container orchestration, observability.            │
└─────────────────────────────────────────────────────┘
```

## Data Flow

```
Input Documents
      │
      ▼
[Semantic Partitioner] ──── SBERT embeddings
      │
      ├──► Partition 1 ──► [Reason Worker 1] ──► ReasonOutput[]
      ├──► Partition 2 ──► [Reason Worker 2] ──► ReasonOutput[]
      └──► Partition N ──► [Reason Worker N] ──► ReasonOutput[]
                                                      │
                                                      ▼
                                            [Key Grouping]
                                                      │
                                                      ▼
                                        [Dempster-Shafer Combiner]
                                                      │
                                                      ▼
                                          ConsensusResult[]
                                          (with posterior confidence)
```

## Key Design Decisions

| Decision | Choice | Alternative | Rationale |
|----------|--------|-------------|-----------|
| Compute framework | Ray | Spark | Actor model suits LLM worker lifecycle |
| LLM serving | vLLM | TGI, Triton | PagedAttention, continuous batching |
| Aggregation theory | Dempster-Shafer | Bayesian | Explicit conflict handling |
| Embedding model | SBERT | OpenAI ada | No external API dependency |
| Package manager | uv | pip/poetry | 10-100x faster installs |
