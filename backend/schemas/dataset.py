"""
backend/schemas/dataset.py
--------------------------
Pydantic contracts for dataset ingestion, profiling, and quality auditing.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class DatasetUploadResponse(BaseModel):
    dataset_id: str
    name: str
    row_count: int
    column_count: int
    file_type: str
    size_bytes: int
    created_at: str
    message: str


class ColumnSummary(BaseModel):
    name: str
    dtype: str
    missing_count: int
    missing_pct: float
    unique_count: int
    unique_pct: float
    is_numeric: bool
    is_categorical: bool
    is_constant: bool
    is_identifier_candidate: bool
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    skewness: Optional[float] = None
    top_categories: Optional[List[Dict[str, Any]]] = None


class DatasetProfileResponse(BaseModel):
    dataset_id: str
    name: str
    dimensions: Dict[str, int]
    missing_summary: Dict[str, Any]
    duplicate_rows: int
    numeric_column_count: int
    categorical_column_count: int
    columns: List[ColumnSummary]
    preview_rows: Optional[List[Dict[str, Any]]] = None


class QualityFinding(BaseModel):
    issue: str
    affected_column: str
    evidence: str
    severity: str
    why_it_matters: str
    recommended_action: str
    trade_offs: str
    user_review_required: bool


class TargetRecommendation(BaseModel):
    suggested_target: str
    detected_task: str
    confidence: str
    reasons: List[str]
    candidate_targets: List[str]


class PreprocessingRecommendation(BaseModel):
    feature: str
    detected_type: str
    issue: str
    recommendation: str
    reason: str
    evidence: str
    confidence: str
    why_it_matters: str
    trade_offs: str


class DatasetAuditResponse(BaseModel):
    dataset_id: str
    target_recommendation: TargetRecommendation
    quality_findings: List[QualityFinding]
    leakage_warnings: List[Dict[str, Any]]
    recommendations: List[PreprocessingRecommendation]
    health_score: float
    health_status: str
