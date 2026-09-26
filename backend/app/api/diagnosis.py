"""
Diagnosis endpoints.

POST /api/diagnosis/predict   — image only → prediction + LIME
POST /api/diagnosis/analyze   — image and/or question → combined workflow
POST /api/diagnosis/followup  — follow-up question with prediction context
"""

import logging

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from app.schemas.diagnosis import (
    AnalyzeResponse,
    LimeResult,
    PredictResponse,
    PredictionResult,
    RegionWeight,
)
from app.schemas.rag import SourceInfo

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/diagnosis", tags=["Diagnosis"])

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}


# ── Request schema for follow-up ──────────────────────────────────────
class FollowUpRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The follow-up question.")
    predicted_class: str = Field(..., description="The predicted disease class from initial analysis.")
    confidence: float = Field(..., description="The confidence score from initial analysis.")


async def _read_image(image: UploadFile) -> bytes:
    """Validate and read uploaded image bytes."""
    if image.content_type and image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image format: {image.content_type}. "
                   f"Supported: JPEG, PNG, WebP, BMP.",
        )
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    return data


def _run_prediction(classifier, image_bytes: bytes) -> dict:
    """Run EfficientNet prediction."""
    try:
        return classifier.predict(image_bytes)
    except Exception as exc:
        logger.exception("Classifier prediction failed")
        raise HTTPException(status_code=500, detail="Image classification failed.") from exc


def _run_lime(lime_explainer, classifier, image_bytes: bytes) -> dict:
    """Run LIME explanation."""
    try:
        img_array = classifier.bytes_to_array(image_bytes)
        return lime_explainer.explain(img_array, classifier.predict_batch)
    except Exception as exc:
        logger.exception("LIME explanation failed")
        raise HTTPException(status_code=500, detail="LIME explanation generation failed.") from exc


def _build_rag_result(gen_output, results):
    """Build a RagResponse from generator output and retrieval results."""
    from app.schemas.rag import GenerationSection, RagResponse

    sources = [
        SourceInfo(
            document_id=r["document_id"],
            title=r["title"],
            section=r["section"],
            score=r["score"],
        )
        for r in results
    ]

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


# ──────────────────────────────────────────────────────────────────────
# POST /api/diagnosis/predict
# ──────────────────────────────────────────────────────────────────────

@router.post("/predict", response_model=PredictResponse)
async def predict(request: Request, image: UploadFile = File(...)):
    """Classify a plant-leaf image and return LIME explanation."""

    image_bytes = await _read_image(image)

    classifier = request.app.state.classifier
    lime_explainer = request.app.state.lime_explainer

    pred = _run_prediction(classifier, image_bytes)
    lime_result = _run_lime(lime_explainer, classifier, image_bytes)

    return PredictResponse(
        prediction=PredictionResult(
            predicted_class=pred["class"],
            confidence=pred["confidence"],
        ),
        lime=LimeResult(
            image=lime_result["image"],
            top_positive_regions=[RegionWeight(**r) for r in lime_result["top_positive_regions"]],
            top_negative_regions=[RegionWeight(**r) for r in lime_result["top_negative_regions"]],
        ),
    )


# ──────────────────────────────────────────────────────────────────────
# POST /api/diagnosis/analyze
# ──────────────────────────────────────────────────────────────────────

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    request: Request,
    image: UploadFile | None = File(None),
    question: str | None = Form(None),
):
    """
    Combined endpoint — supports:
    • image only → prediction + LIME + auto-generated disease summary
    • question only → RAG
    • image + question → prediction + LIME + RAG grounded in prediction
    """

    if image is None and not question:
        raise HTTPException(
            status_code=400,
            detail="Provide an image, a question, or both.",
        )

    classifier = request.app.state.classifier
    lime_explainer = request.app.state.lime_explainer
    retriever = request.app.state.retriever
    generator = request.app.state.generator

    prediction = None
    lime_data = None
    rag_result = None

    # ── Image processing ──────────────────────────────────────────
    if image is not None and image.filename:
        image_bytes = await _read_image(image)
        pred = _run_prediction(classifier, image_bytes)
        prediction = PredictionResult(
            predicted_class=pred["class"],
            confidence=pred["confidence"],
        )
        lime_raw = _run_lime(lime_explainer, classifier, image_bytes)
        lime_data = LimeResult(
            image=lime_raw["image"],
            top_positive_regions=[RegionWeight(**r) for r in lime_raw["top_positive_regions"]],
            top_negative_regions=[RegionWeight(**r) for r in lime_raw["top_negative_regions"]],
        )

    # ── Question / Auto-summary processing ────────────────────────
    if question and question.strip():
        # User provided a specific question — enrich retrieval with disease
        # context if a prediction is available, so we get relevant chunks.
        if prediction:
            disease_name = prediction.predicted_class.replace("___", " ").replace("_", " ")
            retrieval_query = f"{disease_name} {question}"
        else:
            retrieval_query = question

        try:
            results = retriever.retrieve(retrieval_query)
        except Exception as exc:
            logger.exception("RAG retrieval failed")
            raise HTTPException(status_code=502, detail="Knowledge retrieval failed.") from exc

        try:
            if prediction:
                gen_output = generator.generate_with_prediction(
                    question=question,
                    prediction=prediction.predicted_class,
                    confidence=prediction.confidence,
                    results=results,
                )
            else:
                gen_output = generator.generate(question, results)
        except Exception:
            logger.exception("Gemini generation encountered a permanent failure")
            from app.services.generator import GeminiStructuredOutput
            gen_output = GeminiStructuredOutput(
                status="generation_unavailable",
                summary="The AI generation service is temporarily unavailable. Please try again later.",
                sections=[]
            )

        rag_result = _build_rag_result(gen_output, results)

    elif prediction:
        # Image uploaded without a question → auto-generate disease summary
        disease_query = prediction.predicted_class.replace("___", " ").replace("_", " ")

        try:
            results = retriever.retrieve(disease_query)
        except Exception as exc:
            logger.exception("RAG retrieval failed for auto-summary")
            results = []

        if results:
            try:
                gen_output = generator.generate_disease_summary(
                    prediction=prediction.predicted_class,
                    confidence=prediction.confidence,
                    results=results,
                )
            except Exception:
                logger.exception("Gemini auto-summary generation failed")
                from app.services.generator import GeminiStructuredOutput
                gen_output = GeminiStructuredOutput(
                    status="generation_unavailable",
                    summary="The AI generation service is temporarily unavailable. Please try again later.",
                    sections=[]
                )

            rag_result = _build_rag_result(gen_output, results)

    return AnalyzeResponse(
        prediction=prediction,
        lime=lime_data,
        rag=rag_result,
    )


# ──────────────────────────────────────────────────────────────────────
# POST /api/diagnosis/followup
# ──────────────────────────────────────────────────────────────────────

@router.post("/followup", response_model=AnalyzeResponse)
async def followup(body: FollowUpRequest, request: Request):
    """
    Follow-up question endpoint — allows chatbot-style conversation
    after an initial image analysis without re-uploading the image.
    """

    retriever = request.app.state.retriever
    generator = request.app.state.generator

    # Build an enriched retrieval query that includes the disease context.
    # Without this, a generic question like "what treatment should I use"
    # retrieves random, unrelated CABI chunks.
    disease_name = body.predicted_class.replace("___", " ").replace("_", " ")
    enriched_query = f"{disease_name} {body.question}"

    # Retrieve knowledge relevant to the enriched query
    try:
        results = retriever.retrieve(enriched_query)
    except Exception as exc:
        logger.exception("RAG retrieval failed for follow-up")
        raise HTTPException(status_code=502, detail="Knowledge retrieval failed.") from exc

    # Generate answer with prediction context
    try:
        gen_output = generator.generate_with_prediction(
            question=body.question,
            prediction=body.predicted_class,
            confidence=body.confidence,
            results=results,
        )
    except Exception:
        logger.exception("Gemini generation encountered a permanent failure")
        from app.services.generator import GeminiStructuredOutput
        gen_output = GeminiStructuredOutput(
            status="generation_unavailable",
            summary="The AI generation service is temporarily unavailable. Please try again later.",
            sections=[]
        )

    rag_result = _build_rag_result(gen_output, results)

    return AnalyzeResponse(
        prediction=None,
        lime=None,
        rag=rag_result,
    )
