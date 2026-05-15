"""Ray-based cluster initialization, scheduling, and health monitoring.

Provides:
    - init_cluster(): Bootstrap Ray in local or distributed mode.
    - shutdown_cluster(): Graceful teardown.
    - HeartbeatActor: Per-node health monitoring actor.
    - ReasonWorkerActor: Ray-remote wrapper around ReasonWorker.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field

from reason_reduce.monitoring.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ClusterInfo:
    """Information about the active Ray cluster.

    Attributes:
        mode: "local" or "remote".
        n_workers: Number of worker CPUs available.
        address: Ray cluster address.
        node_ids: List of node IDs in the cluster.
    """

    mode: str
    n_workers: int
    address: str
    node_ids: list[str] = field(default_factory=list)


def init_cluster(local: bool = True, n_workers: int = 4) -> ClusterInfo:
    """Initialize the Ray cluster.

    Args:
        local: If True, start a local Ray instance with simulated workers.
               If False, connect to existing cluster via RAY_ADDRESS.
        n_workers: Number of CPUs for local mode.

    Returns:
        ClusterInfo describing the active cluster.
    """
    import ray

    if local:
        if not ray.is_initialized():
            # ray.init reads RAY_ADDRESS from env when address is unset,
            # which would silently flip us into remote mode and conflict
            # with num_cpus. Drop it for the duration of this call so
            # local=True actually means local.
            prev_addr = os.environ.pop("RAY_ADDRESS", None)
            try:
                ray.init(
                    num_cpus=n_workers,
                    ignore_reinit_error=True,
                    logging_level="WARNING",
                )
            finally:
                if prev_addr is not None:
                    os.environ["RAY_ADDRESS"] = prev_addr
        logger.info("ray_cluster_init", mode="local", n_workers=n_workers)
        nodes = ray.nodes()
        return ClusterInfo(
            mode="local",
            n_workers=n_workers,
            address="local",
            node_ids=[n["NodeID"] for n in nodes],
        )
    else:
        address = os.environ.get("RAY_ADDRESS", "auto")
        if not ray.is_initialized():
            ray.init(address=address, ignore_reinit_error=True)
        nodes = ray.nodes()
        total_cpus = sum(n.get("Resources", {}).get("CPU", 0) for n in nodes)
        logger.info(
            "ray_cluster_init",
            mode="remote",
            address=address,
            n_nodes=len(nodes),
            total_cpus=total_cpus,
        )
        return ClusterInfo(
            mode="remote",
            n_workers=int(total_cpus),
            address=address,
            node_ids=[n["NodeID"] for n in nodes],
        )


def shutdown_cluster() -> None:
    """Shut down the Ray cluster gracefully."""
    import ray

    if ray.is_initialized():
        ray.shutdown()
        logger.info("ray_cluster_shutdown")


def create_heartbeat_actors(cluster_info: ClusterInfo) -> list[object]:
    """Create a HeartbeatActor on each node for health monitoring.

    Args:
        cluster_info: Active cluster information.

    Returns:
        List of HeartbeatActor handles.
    """
    import ray

    @ray.remote(num_cpus=0.1)
    class HeartbeatActor:
        """Per-node actor that reports health status.

        Tracks uptime, processed count, and last heartbeat time.
        Used by the scheduler to detect stale/dead nodes.
        """

        def __init__(self, node_id: str) -> None:
            self._node_id = node_id
            self._start_time = time.time()
            self._heartbeat_count = 0
            self._last_heartbeat = time.time()

        def ping(self) -> dict[str, float | str | int]:
            """Respond to a health check.

            Returns:
                Dict with node_id, uptime, heartbeat_count, timestamp.
            """
            self._heartbeat_count += 1
            self._last_heartbeat = time.time()
            return {
                "node_id": self._node_id,
                "uptime_seconds": time.time() - self._start_time,
                "heartbeat_count": self._heartbeat_count,
                "timestamp": self._last_heartbeat,
            }

        def get_status(self) -> dict[str, float | str | int]:
            """Get current status without incrementing counter."""
            return {
                "node_id": self._node_id,
                "uptime_seconds": time.time() - self._start_time,
                "heartbeat_count": self._heartbeat_count,
                "last_heartbeat": self._last_heartbeat,
                "alive": True,
            }

    actors = []
    for node_id in cluster_info.node_ids:
        actor = HeartbeatActor.remote(node_id)  # type: ignore[attr-defined]
        actors.append(actor)

    logger.info("heartbeat_actors_created", n_actors=len(actors))
    return actors


def create_reason_worker_pool(
    n_workers: int,
    model_id: str = "mock",
    num_gpus_per_worker: float = 0.0,
) -> list[object]:
    """Create a pool of Ray-remote ReasonWorker actors.

    Args:
        n_workers: Number of worker actors to create.
        model_id: Model ID for worker initialization.
        num_gpus_per_worker: GPU fraction per worker (0 for CPU-only).

    Returns:
        List of Ray actor handles.
    """
    import ray

    from reason_reduce.models.registry import ModelRegistry

    @ray.remote
    class RemoteReasonWorker:
        """Ray-remote wrapper around ReasonWorker.

        Each actor holds a persistent LLM adapter instance.
        """

        def __init__(self, worker_id: int, model_id: str) -> None:
            from reason_reduce.reason.worker import ReasonWorker

            self._worker_id = worker_id
            registry = ModelRegistry()
            adapter = registry.get(model_id)
            self._worker = ReasonWorker(adapter=adapter, partition_id=worker_id)
            self._processed = 0

        def process(self, doc_data: dict[str, str], task_data: dict[str, str]) -> dict:  # type: ignore[type-arg]
            """Process a document (serialized as dict for Ray transport).

            Args:
                doc_data: Serialized Doc fields.
                task_data: Serialized TaskSpec fields.

            Returns:
                Serialized ReasonOutput as dict.
            """
            from reason_reduce.ingestion.batch import Doc
            from reason_reduce.reason.worker import TaskSpec

            doc = Doc(**doc_data)
            task = TaskSpec(**task_data)

            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(self._worker.process(doc, task))
            finally:
                loop.close()

            self._processed += 1
            return {
                "key": result.key,
                "value": result.value,
                "confidence": result.confidence,
                "trace": result.trace,
                "model_id": result.model_id,
                "latency_ms": result.latency_ms,
                "doc_id": result.doc_id,
            }

        def get_stats(self) -> dict[str, int]:
            """Get worker statistics."""
            return {
                "worker_id": self._worker_id,
                "processed": self._processed,
            }

    if num_gpus_per_worker > 0:
        RemoteReasonWorker = ray.remote(num_gpus=num_gpus_per_worker)(  # type: ignore[misc]  # noqa: N806
            RemoteReasonWorker.__wrapped__  # type: ignore[attr-defined]
        )

    actors = []
    for i in range(n_workers):
        actor = RemoteReasonWorker.remote(i, model_id)  # type: ignore[attr-defined]
        actors.append(actor)

    logger.info(
        "worker_pool_created",
        n_workers=n_workers,
        model_id=model_id,
        gpus_per_worker=num_gpus_per_worker,
    )
    return actors
