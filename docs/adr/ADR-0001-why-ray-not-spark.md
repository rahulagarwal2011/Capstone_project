# ADR-0001: Why Ray Instead of Apache Spark

## Status
Accepted

## Context
The Reason-Reduce system needs distributed compute for data-parallel processing.
Apache Spark is the industry standard for batch data processing (MapReduce paradigm).
However, our workload is different: each "map" step involves a stateful LLM worker
that loads model weights and maintains a KV-cache.

## Decision
Use Ray as the primary distributed compute framework instead of Apache Spark.

## Rationale

### Spark Limitations for LLM Workloads
1. **JVM-based**: Python UDFs incur serialization overhead via Py4J
2. **Stateless tasks**: Spark executors don't maintain state between tasks; LLM workers need persistent model weights
3. **No actor model**: Cannot represent a long-running LLM server as a first-class primitive
4. **GPU scheduling**: Spark's resource model has limited GPU-awareness

### Ray Advantages
1. **Native Python**: No JVM overhead, direct memory sharing
2. **Actor model**: `@ray.remote` classes maintain LLM state across invocations
3. **GPU-aware scheduling**: First-class `num_gpus` resource specification
4. **Ecosystem**: vLLM, Serve, Train all native Ray libraries
5. **Placement groups**: Control co-location of model weights and data shards

### What We Lose
1. Spark's mature SQL optimizer (Catalyst) — not needed for our workload
2. Spark's broad connector ecosystem — we only need filesystem/S3
3. Industry familiarity — mitigated by Spark-RDD-like API wrapper

## Consequences
- The public API (ReasonRDD) mirrors Spark semantics for familiarity
- A Spark plugin wrapper could be built later if needed (Phase 5)
- Deployment uses KubeRay operator instead of spark-on-k8s-operator

## References
- Moritz et al., "Ray: A Distributed Framework for Emerging AI Applications", OSDI 2018
- Zaharia et al., "Apache Spark: A Unified Engine for Big Data Processing", CACM 2016
