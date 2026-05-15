# Related Work

## Positioning of Reason-Reduce

Reason-Reduce sits at the intersection of distributed data processing and LLM inference.
Unlike prior work that treats LLMs as external services called from a data pipeline,
Reason-Reduce embeds LLM reasoning directly in the data plane as a first-class operator.

## Adjacent Systems

### Lotus / SUQL (UC Berkeley)
- **What:** Semantic operators on relational tables using LLMs
- **Difference:** SQL-style declarative; single-node; confidence used as filter only.
  Reason-Reduce is MapReduce-style imperative, distributed-first, and propagates confidence
  through Dempster-Shafer aggregation.

### Palimpzest (MIT)
- **What:** LLM-powered declarative data system
- **Difference:** Declarative query model; no distributed execution story;
  no probabilistic aggregation with conflict handling.

### DocETL (UC Berkeley)
- **What:** Agentic data processing pipelines
- **Difference:** Agentic loop for iterative refinement; not distributed-first;
  no formal evidence combination theory.

### Snowflake Cortex / Databricks AI Functions
- **What:** In-database LLM operators
- **Difference:** Vendor-proprietary; SQL-focused; calls external LLM APIs (not co-located);
  no probabilistic semantics on outputs.

### LangChain / LlamaIndex
- **What:** LLM orchestration frameworks
- **Difference:** Orchestration layer (chains/agents), not data processing operators.
  No distributed execution, no probabilistic aggregation.

## Key Differentiators of Reason-Reduce
1. MapReduce-style imperative operators (not SQL declarative)
2. Distributed-first (Ray, multi-node) — not single-node
3. LLM co-located with data shards (not external API)
4. Probabilistic propagation via Dempster-Shafer (not confidence-as-filter)
5. Cost-aware routing across model sizes
