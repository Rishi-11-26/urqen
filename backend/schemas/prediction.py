"""
backend/schemas/prediction.py
-----------------------------
Pydantic contracts for model inference and live predictions.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class PredictionRecord(BaseModel):
    prediction: Any
    probability: Optional[Dict[str, float]] = None


class PredictionResponse(BaseModel):
    experiment_id: str
    model_name: str
    task_type: str
    predictions_count: int
    predictions: List[PredictionRecord]
