"""Pydantic models for RAG endpoints."""

from pydantic import BaseModel, Field


class RagRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The plant-health question.")


class SourceInfo(BaseModel):
    document_id: str
    title: str
    section: str
    score: float


class GenerationSection(BaseModel):
    heading: str
    content_markdown: str


class RagResponse(BaseModel):
    status: str
    summary: str | None = None
    sections: list[GenerationSection] | None = None
    sources: list[SourceInfo] | None = None
