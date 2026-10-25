"""
backend/schemas/experiment.py
-----------------------------
Pydantic contracts for experiment configuration, benchmarking, and historical records.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ExperimentCreateRequest(BaseModel):
    dataset_id: str
    target_column: str
    task_type: Optional[str] = None  # Auto-detected if omitted
    mode: str = Field(default="guided", pattern="^(guided|expert)$")

    # Optional explicit list of input features (if None, all non-target columns used)
    feature_columns: Optional[List[str]] = None

    # Models to benchmark (if None, standard recommended suite is used)
    models: Optional[List[str]] = None

    # Validation strategy
    cv_folds: Optional[int] = Field(default=10, ge=2, le=20)
    test_size: Optional[float] = Field(default=0.20, ge=0.05, le=0.50)

    # Preprocessing controls
    scaler_type: Optional[str] = Field(default="standard", pattern="^(standard|minmax|robust|none)$")
    user_overrides: Optional[Dict[str, str]] = None

    # Specific algorithm hyperparameters (Expert Mode)
    hyperparameters: Optional[Dict[str, Dict[str, Any]]] = None


class ModelBenchmarkResult(BaseModel):
    model: str
    family: Optional[str] = "general"
    status: str
    training_time_seconds: float
    cv_mean: Optional[float] = None
    cv_std: Optional[float] = None
    cv_folds: Optional[int] = None
    cv_fold_scores: Optional[List[float]] = None
    holdout_sample_count: Optional[int] = None
    accuracy: Optional[float] = None
    f1: Optional[float] = None
    f1_macro: Optional[float] = None
    f1_weighted: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    sensitivity: Optional[float] = None
    specificity: Optional[float] = None
    roc_auc: Optional[float] = None
    r2: Optional[float] = None
    mae: Optional[float] = None
    rmse: Optional[float] = None
    error_message: Optional[str] = None


class ExperimentResponse(BaseModel):
    experiment_id: str
    user_id: str
    dataset_id: str
    dataset_name: str
    task_type: str
    target_column: str
    mode: str
    status: str
    created_at: str
    completed_at: Optional[str] = None
    best_model: Optional[str] = None
    winner_explanation: Optional[str] = None
    models: Optional[List[ModelBenchmarkResult]] = None
    results: Optional[Dict[str, Any]] = None
    cv_folds: Optional[int] = None
    test_size: Optional[float] = None
    configuration: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class ExperimentSummary(BaseModel):
    experiment_id: str
    dataset_name: str
    task_type: str
    target_column: str
    mode: str
    status: str
    best_model: Optional[str] = None
    created_at: str


class StorageQuotaResponse(BaseModel):
    user_id: str
    dataset_count: int
    dataset_bytes: int
    experiment_count: int
    artifact_bytes: int
    total_used_bytes: int
    quota_limit_bytes: int
    quota_exhausted: bool
