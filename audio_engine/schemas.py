from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str = "ok"
    model_loaded: bool = False
    model_version: str | None = None
    service: str = "audio-engine"


class FeatureSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mfcc_mean: list[float] = Field(default_factory=list)
    zcr_mean: float | None = None
    rms_mean: float | None = None
    duration_s: float | None = None
    estimated_heart_rate_bpm: float | None = None


class PredictionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    probability_pathological: float
    threshold: float = 0.5
    n_segments: int = 1
    segment_probabilities: list[float] = Field(default_factory=list)
    duration_s: float | None = None
    model_version: str | None = None
    inference_ms: float | None = None
    features_summary: FeatureSummary | None = None


class DatasetMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recording_id: str
    source: str
    file_path: str
    original_label: str
    binary_label: int
    original_fs: int | None = None
    duration_s: float | None = None


class BenchmarkResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    algorithm: str
    accuracy: float
    recall_pathological: float | None = None
    specificity: float | None = None
    f1: float | None = None
    roc_auc: float | None = None
    pr_auc: float | None = None
    latency_ms: float | None = None
    model_size_mb: float | None = None
    params: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
