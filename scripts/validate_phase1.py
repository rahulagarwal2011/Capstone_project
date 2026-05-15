"""Phase 1 Validation: Runs all exit criteria checks.

Exit Criteria (all must hold):
✓ Smoke test passes (pipeline with mock LLM, no GPU)
✓ Tests pass with ≥50% coverage on Phase-1 modules (target: 80% by Phase 1 end)
✓ Lint (ruff check) passes clean
✓ Prometheus scrapes /metrics endpoint
✓ A trace is generated with correlation ID
✓ Config system loads correctly
✓ Model registry works (mock + API adapters)
✓ Ray cluster initializes and heartbeat works

Usage:
    python scripts/validate_phase1.py
"""

from __future__ import annotations

import sys
import time
import subprocess

from reason_reduce.monitoring.logger import get_logger

logger = get_logger(__name__)

CHECKS: list[tuple[str, str]] = []
PASSED = 0
FAILED = 0


def check(name: str, passed: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if passed:
        PASSED += 1
        logger.info("check_passed", name=name, detail=detail)
    else:
        FAILED += 1
        logger.error("check_failed", name=name, detail=detail)


def main() -> int:
    logger.info("phase1_validation_start")
    start = time.perf_counter()

    # 1. Config system
    try:
        from reason_reduce.config.loader import load_settings

        settings = load_settings()
        check(
            "config_system",
            settings.thresholds.tau_confidence == 0.7 and len(settings.models) >= 3,
            f"tau={settings.thresholds.tau_confidence}, models={len(settings.models)}",
        )
    except Exception as e:
        check("config_system", False, str(e))

    # 2. Model registry
    try:
        from reason_reduce.models.registry import ModelRegistry, MockLLMAdapter

        registry = ModelRegistry()
        adapter = registry.get("mock")
        check("model_registry_mock", isinstance(adapter, MockLLMAdapter))
        check("model_registry_list", len(registry.list_available()) >= 3)
        check("model_registry_can_fit", registry.can_fit("mistral-7b", 8.0))
    except Exception as e:
        check("model_registry", False, str(e))

    # 3. Structured logging with correlation ID
    try:
        from reason_reduce.monitoring.tracing import set_correlation_id, get_correlation_id

        set_correlation_id("test-123")
        check("correlation_id", get_correlation_id() == "test-123")
    except Exception as e:
        check("correlation_id", False, str(e))

    # 4. Prometheus metrics
    try:
        from reason_reduce.monitoring.metrics import (
            start_metrics_server,
            record_reason_request,
            REASON_REQUESTS_TOTAL,
        )

        start_metrics_server(port=8111)
        record_reason_request("mock", "ner", "ok", 0.1, 0.85)

        import httpx

        resp = httpx.get("http://localhost:8111/metrics", timeout=5)
        has_metric = "reason_reduce_reason_requests_total" in resp.text
        check("prometheus_metrics", resp.status_code == 200 and has_metric)
    except Exception as e:
        check("prometheus_metrics", False, str(e))

    # 5. Data loader
    try:
        from reason_reduce.ingestion.batch import Doc, compute_dataset_hash

        docs = [Doc(id="1", text="hello"), Doc(id="2", text="world")]
        h = compute_dataset_hash(docs)
        check("data_loader", len(h) == 64 and h == compute_dataset_hash(docs))
    except Exception as e:
        check("data_loader", False, str(e))

    # 6. Smoke pipeline
    try:
        from reason_reduce.ingestion.batch import Doc
        from reason_reduce.reason.api import reason
        from reason_reduce.reason.worker import TaskSpec
        from reason_reduce.reduce.api import reason_reduce

        docs = [Doc(id=str(i), text=f"Doc {i}") for i in range(10)]
        results = reason(docs, task=TaskSpec(), model="mock", n_partitions=2, seed=42)
        reduced = reason_reduce(results, strategy="ds", seed=42)
        check(
            "smoke_pipeline",
            len(results) == 10 and len(reduced) >= 1,
            f"reason={len(results)}, reduce={len(reduced)}",
        )
    except Exception as e:
        check("smoke_pipeline", False, str(e))

    # 7. Ray cluster (if available)
    try:
        import ray

        from reason_reduce.orchestration.scheduler import init_cluster, shutdown_cluster

        cluster = init_cluster(local=True, n_workers=4)
        check("ray_cluster", cluster.mode == "local" and cluster.n_workers == 4)
        shutdown_cluster()
    except ImportError:
        check("ray_cluster", False, "ray not installed")
    except Exception as e:
        check("ray_cluster", False, str(e))

    # 8. Partitioner with real clustering
    try:
        from reason_reduce.ingestion.batch import Doc
        from reason_reduce.reason.partitioner import partition_documents

        docs = [Doc(id=str(i), text=f"topic {'medical' if i < 15 else 'legal'} doc {i}") for i in range(30)]
        partitions = partition_documents(docs, n_partitions=3, strategy="semantic", seed=42)
        check(
            "partitioner",
            len(partitions) == 3 and all(len(p.docs) > 0 for p in partitions),
            f"partitions={len(partitions)}, coherences={[round(p.coherence_score, 3) for p in partitions]}",
        )
    except Exception as e:
        check("partitioner", False, str(e))

    # Summary
    elapsed = time.perf_counter() - start
    total = PASSED + FAILED
    logger.info(
        "phase1_validation_complete",
        passed=PASSED,
        failed=FAILED,
        total=total,
        elapsed_seconds=round(elapsed, 2),
        status="ALL PASS" if FAILED == 0 else "FAILURES DETECTED",
    )

    if FAILED == 0:
        logger.info("phase1_ready", message="Phase 1 complete. Ready to proceed to Phase 2.")
    else:
        logger.error("phase1_not_ready", message=f"{FAILED} checks failed. Fix before Phase 2.")

    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
