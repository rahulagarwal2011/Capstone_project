"""Weights & Biases integration for experiment tracking.

Provides lightweight wrappers for logging runs, metrics, and artifacts.
Gracefully no-ops if wandb is not installed or not logged in.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass

from reason_reduce.monitoring.logger import get_logger

logger = get_logger(__name__)

_wandb_available: bool | None = None


def _check_wandb() -> bool:
    """Check if wandb is importable and configured."""
    global _wandb_available
    if _wandb_available is not None:
        return _wandb_available
    try:
        import wandb  # noqa: F401

        _wandb_available = True
    except ImportError:
        _wandb_available = False
        logger.info("wandb_not_installed", hint="pip install wandb")
    return _wandb_available


@dataclass
class ExperimentConfig:
    """Configuration for a tracked experiment.

    Attributes:
        name: Experiment name (used as W&B run name).
        dataset: Dataset identifier.
        dataset_sha256: Hash of the dataset for reproducibility.
        model: Model used.
        task: Task type.
        seed: Random seed.
        git_sha: Current git commit SHA.
        config_hash: Hash of the configuration.
        n_docs: Number of documents processed.
        n_partitions: Number of partitions.
        strategy: Aggregation strategy.
    """

    name: str = ""
    dataset: str = ""
    dataset_sha256: str = ""
    model: str = "mock"
    task: str = "ner"
    seed: int = 42
    git_sha: str = ""
    config_hash: str = ""
    n_docs: int = 0
    n_partitions: int = 0
    strategy: str = "ds"


def get_git_sha() -> str:
    """Get the current git commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()[:8] if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def compute_config_hash(config: dict[str, object]) -> str:
    """Compute a deterministic hash of a configuration dict."""
    serialized = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:12]


def init_run(config: ExperimentConfig) -> object:
    """Initialize a W&B run for experiment tracking.

    Returns None if wandb is unavailable — all downstream calls no-op.

    Args:
        config: Experiment configuration.

    Returns:
        wandb.Run or None.
    """
    if not _check_wandb():
        return None

    import wandb

    config.git_sha = get_git_sha()
    config.config_hash = compute_config_hash(asdict(config))

    project = os.environ.get("WANDB_PROJECT", "reason-reduce")
    entity = os.environ.get("WANDB_ENTITY")

    try:
        run = wandb.init(
            project=project,
            entity=entity,
            name=config.name or f"{config.task}-{config.model}-{config.seed}",
            config=asdict(config),
            tags=[config.task, config.model, config.strategy],
        )
        logger.info("wandb_run_started", run_id=run.id, url=run.url)
        return run
    except Exception as e:
        logger.warning("wandb_init_failed", error=str(e))
        return None


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Log metrics to the active W&B run.

    No-ops if no run is active.
    """
    if not _check_wandb():
        return

    import wandb

    if wandb.run is not None:
        wandb.log(metrics, step=step)


def finish_run() -> None:
    """Finish the active W&B run."""
    if not _check_wandb():
        return

    import wandb

    if wandb.run is not None:
        wandb.finish()
        logger.info("wandb_run_finished")
