"""
RAG endpoint — POST /api/rag/ask

Accepts a plant-health question and returns a Gemini-generated
answer grounded in CABI knowledge, with source citations.
"""

import logging

from fastapi import APIRouter, HTTPException, Request

from app.schemas.rag import RagRequest, RagResponse, SourceInfo

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/rag", tags=["RAG"])


@router.post("/ask", response_model=RagResponse)
async def ask(body: RagRequest, request: Request):
    """Answer a plant-health question using RAG."""

    retriever = request.app.state.retriever
    generator = request.app.state.generator

    # 1. Retrieve
    try:
        results = retriever.retrieve(body.question)
    except Exception as exc:
        logger.exception("RAG retrieval failed")
        raise HTTPException(status_code=502, detail="Knowledge retrieval failed.") from exc

    if not results:
        raise HTTPException(
            status_code=404,
            detail="No relevant information found in the knowledge base.",
        )

    # 2. Generate
    try:
        gen_output = generator.generate(body.question, results)
    except Exception:
        logger.exception("Gemini generation encountered a permanent failure")
        from app.services.generator import GeminiStructuredOutput
        gen_output = GeminiStructuredOutput(
            status="generation_unavailable",
            summary="The AI generation service is temporarily unavailable. Please try again later.",
            sections=[]
        )

    # 3. Build sources (backend-controlled — never invented by the LLM)
    sources = [
        SourceInfo(
            document_id=r["document_id"],
            title=r["title"],
            section=r["section"],
            score=r["score"],
        )
        for r in results
    ]

    from app.schemas.rag import GenerationSection
    sections = [
        GenerationSection(heading=s.heading, content_markdown=s.content_markdown)
        for s in (gen_output.sections or [])
    ]

    return RagResponse(
        status=gen_output.status,
        summary=gen_output.summary,
        sections=sections,
        sources=sources,
    )
