"""Pydantic-based configuration loading for Reason-Reduce.

Reads from config/*.yaml files and environment variables. Provides a single
Settings object that is the source of truth for all runtime configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

_CONFIG_DIR = Path(__file__).parent


def _load_yaml(filename: str) -> dict[str, Any]:
    """Load a YAML file from the config directory."""
    path = _CONFIG_DIR / filename
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


class ThresholdSettings(BaseSettings):
    """Confidence and routing thresholds.

    Confidence Semantics:
        All threshold values are in [0, 1] and represent calibrated probabilities.
        tau_confidence is the minimum confidence for a result to be accepted without escalation.
    """

    tau_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    tau_entropy: float = Field(default=0.5, ge=0.0, le=1.0)
    tau_consensus: float = Field(default=0.6, ge=0.0, le=1.0)
    tau_conflict_low: float = Field(default=0.3, ge=0.0, le=1.0)
    tau_conflict_high: float = Field(default=0.7, ge=0.0, le=1.0)
    max_workers: int = Field(default=4, ge=1)
    max_retries: int = Field(default=2, ge=0)

    @field_validator("tau_conflict_high")
    @classmethod
    def conflict_high_gt_low(cls, v: float, info: Any) -> float:
        """Ensure high conflict threshold exceeds low threshold."""
        low = info.data.get("tau_conflict_low", 0.3)
        if v <= low:
            msg = f"tau_conflict_high ({v}) must be > tau_conflict_low ({low})"
            raise ValueError(msg)
        return v


class ModelInfo(BaseSettings):
    """Registry entry for a single model."""

    name: str
    hf_id: str
    vram_gb: float
    ctx_len: int
    quantization: str = "none"
    cost_per_1k: float = 0.0


class Settings(BaseSettings):
    """Root settings for the Reason-Reduce system."""

    model_config = {"env_prefix": "RR_"}

    thresholds: ThresholdSettings = Field(default_factory=ThresholdSettings)
    models: list[ModelInfo] = Field(default_factory=list)

    ray_address: str = "auto"
    ray_num_workers: int = 4
    redis_url: str = "redis://localhost:6379/0"
    default_model: str = "mistral-7b"
    default_seed: int = 42


def load_settings() -> Settings:
    """Load settings from YAML config files and environment variables.

    Returns:
        Fully-resolved Settings object.
    """
    thresholds_data = _load_yaml("thresholds.yaml")
    models_data = _load_yaml("models.yaml")

    thresholds = ThresholdSettings(**thresholds_data) if thresholds_data else ThresholdSettings()
    models = [ModelInfo(**m) for m in models_data.get("models", [])]

    return Settings(thresholds=thresholds, models=models)
