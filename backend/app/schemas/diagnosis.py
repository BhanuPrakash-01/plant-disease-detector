"""Pydantic models for diagnosis endpoints."""

from __future__ import annotations

from pydantic import BaseModel

from app.schemas.rag import RagResponse


class RegionWeight(BaseModel):
    segment: int
    weight: float


class LimeResult(BaseModel):
    image: str  # base64-encoded PNG
    top_positive_regions: list[RegionWeight]
    top_negative_regions: list[RegionWeight]


class PredictionResult(BaseModel):
    predicted_class: str
    confidence: float


class PredictResponse(BaseModel):
    prediction: PredictionResult
    lime: LimeResult


class AnalyzeResponse(BaseModel):
    prediction: PredictionResult | None = None
    lime: LimeResult | None = None
    rag: RagResponse | None = None
