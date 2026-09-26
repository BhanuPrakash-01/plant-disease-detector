"""
Gemini LLM generation service.

Uses the official google-genai SDK to generate structured answers
grounded in CABI-retrieved context. Implements fallback and retry logic.
"""

import json
import logging
import time

from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.config import settings

logger = logging.getLogger(__name__)


class GeminiSection(BaseModel):
    heading: str
    content_markdown: str


class GeminiStructuredOutput(BaseModel):
    status: str
    summary: str | None = None
    sections: list[GeminiSection] | None = None


def should_retry_api_error(exc: Exception) -> bool:
    """Determine if the exception is a transient Gemini API error."""
    if isinstance(exc, APIError):
        # 429 = Too Many Requests
        # 500 = Internal Server Error
        # 502 = Bad Gateway
        # 503 = Service Unavailable
        # 504 = Gateway Timeout
        if exc.code in (429, 500, 502, 503, 504):
            return True
    return False


class GeminiGenerator:
    """Generates plant-health answers using Gemini + retrieved CABI context."""

    def __init__(self):
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured.")
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.primary_model = settings.gemini_primary_model
        self.fallback_model = settings.gemini_fallback_model
        logger.info(
            "Gemini client initialised — primary: %s | fallback: %s",
            self.primary_model,
            self.fallback_model,
        )

    def _call_model(self, model_name: str, prompt: str) -> GeminiStructuredOutput:
        """Call the model with bounded exponential backoff for transient errors."""

        @retry(
            retry=retry_if_exception(should_retry_api_error),
            wait=wait_exponential_jitter(initial=1, max=10),
            stop=stop_after_attempt(3),
            reraise=True,
        )
        def _attempt():
            start_time = time.time()
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=GeminiStructuredOutput,
                    ),
                )
                latency = time.time() - start_time
                logger.info(
                    "Structured generation successful [model=%s, latency=%.2fs]",
                    model_name,
                    latency,
                )

                if hasattr(response, "parsed") and response.parsed:
                    return response.parsed
                
                # Manual parsing if the SDK didn't automatically instantiate the Pydantic object
                data = json.loads(response.text)
                return GeminiStructuredOutput(**data)

            except Exception as e:
                latency = time.time() - start_time
                status_code = getattr(e, "code", "unknown")
                logger.warning(
                    "Generation attempt failed [model=%s, status=%s, latency=%.2fs, error=%s]",
                    model_name,
                    status_code,
                    latency,
                    type(e).__name__,
                )
                raise

        return _attempt()

    def _generate_with_fallback(self, prompt: str) -> GeminiStructuredOutput:
        """Execute the prompt against primary, then fallback if needed."""
        try:
            return self._call_model(self.primary_model, prompt)
        except Exception as primary_exc:
            logger.error(
                "Primary model %s failed (error: %s). Falling back to %s.",
                self.primary_model,
                type(primary_exc).__name__,
                self.fallback_model,
            )
            try:
                return self._call_model(self.fallback_model, prompt)
            except Exception as fallback_exc:
                logger.error(
                    "Fallback model %s also failed (error: %s). Returning unavailable status.",
                    self.fallback_model,
                    type(fallback_exc).__name__,
                )
                return GeminiStructuredOutput(
                    status="generation_unavailable",
                    summary="The AI generation service is temporarily overloaded or unavailable. Please try again later.",
                    sections=[],
                )

    # ── context building ──────────────────────────────────────────────

    @staticmethod
    def build_context(results: list[dict]) -> str:
        """Format retrieved CABI chunks into a prompt-friendly context block."""
        parts = []
        for i, r in enumerate(results, 1):
            parts.append(
                f"SOURCE {i}\n"
                f"Title: {r['title']}\n"
                f"Section: {r['section']}\n"
                f"Document ID: {r['document_id']}\n\n"
                f"Content:\n{r['text']}\n"
            )
        return "\n\n".join(parts)

    # ── generation ────────────────────────────────────────────────────

    def generate(self, question: str, results: list[dict]) -> GeminiStructuredOutput:
        """
        Generate a RAG answer for a standalone knowledge question.
        """
        context = self.build_context(results)

        prompt = f"""You are a plant-health information assistant with deep expertise in agriculture, plant pathology, and crop management.

Answer the user's question about plant health, diseases, pests, or crop management.

Instructions:
- FIRST, check if the retrieved CABI information below is relevant to the user's question.
- If the retrieved information IS relevant, use it as your PRIMARY source and cite specifics from it.
- If the retrieved information is NOT relevant or insufficient for the specific question, use your own expert knowledge to provide a helpful, accurate answer. In this case, set status to "ok" (NOT "insufficient_information") and mention that the answer is based on general agricultural knowledge.
- ALWAYS provide a useful answer. Never refuse to answer a plant health question.
- Give a clear and practical answer using the schema.
- Format all lists properly: each bullet point or numbered item MUST be on its own new line. Do not combine multiple list items into a single paragraph.
- Use useful headings (Symptoms, Causes, Management, Prevention, Identification, Treatment) where appropriate.
- Do not claim to diagnose an image unless a separate classifier has provided a prediction.
- Do not invent specific pesticide doses — instead recommend consulting a local agricultural extension officer for exact dosages.
- Never output raw internal chunk IDs or raw retrieval context in the answer.

Retrieved CABI information:

{context}

User question:

{question}
"""
        return self._generate_with_fallback(prompt)

    def generate_with_prediction(
        self,
        question: str,
        prediction: str,
        confidence: float,
        results: list[dict],
    ) -> GeminiStructuredOutput:
        """
        Generate a RAG answer that also incorporates the classifier prediction.
        """
        context = self.build_context(results)
        disease_name = prediction.replace("___", " — ").replace("_", " ")

        prompt = f"""You are a plant-health information assistant with deep expertise in agriculture, plant pathology, and crop management.

An image classifier has predicted the following disease:
  Prediction: {disease_name}
  Confidence: {confidence:.1%}

The user has a follow-up question about this prediction.

Instructions:
- FIRST, check if the retrieved CABI information below is relevant to the predicted disease and the user's question.
- If the retrieved information IS relevant, use it as your PRIMARY source and incorporate specific details from it.
- If the retrieved information is NOT relevant or insufficient (e.g., it covers a different crop or disease), use your own expert knowledge about "{disease_name}" to provide a helpful, accurate answer. In this case, set status to "ok" and mention the answer is based on general agricultural knowledge.
- ALWAYS provide a useful, detailed answer. Never refuse to answer.
- Clearly state the classifier's prediction at the beginning.
- Give practical, actionable advice about treatment, prevention, symptoms, or whatever the user is asking about.
- Format all lists properly: each bullet point or numbered item MUST be on its own new line. Do not combine multiple list items into a single paragraph.
- Use useful headings (Symptoms, Causes, Management, Prevention, Treatment) where appropriate.
- Do not invent specific pesticide doses — instead recommend consulting a local agricultural extension officer for exact dosages.
- Never output raw internal chunk IDs or raw retrieval context in the answer.

Retrieved CABI information:

{context}

User question:

{question}
"""
        return self._generate_with_fallback(prompt)

    def generate_disease_summary(
        self,
        prediction: str,
        confidence: float,
        results: list[dict],
    ) -> GeminiStructuredOutput:
        """
        Auto-generate a comprehensive disease summary when an image is
        uploaded without a specific question. Covers:
        - What the disease is
        - How it occurs / causes
        - Symptoms
        - Treatment and management measures
        """
        context = self.build_context(results)
        disease_name = prediction.replace("___", " — ").replace("_", " ")

        prompt = f"""You are a plant-health information assistant with deep expertise in agriculture, plant pathology, and crop management.

An image classifier has identified the following condition on a plant leaf:
  Prediction: {disease_name}
  Confidence: {confidence:.1%}

Provide a comprehensive summary about this condition.

Instructions:
- FIRST, check if the retrieved CABI information below is relevant to "{disease_name}".
- If the retrieved information IS relevant, use it as your PRIMARY source and incorporate specific details.
- If the retrieved information is NOT relevant or insufficient (e.g., it covers a different crop or disease), use your own expert knowledge about "{disease_name}" to provide a comprehensive answer. Set status to "ok" regardless.
- ALWAYS provide a complete, useful summary. Never say you don't have information.

Your response MUST include the following sections (use these exact headings):
1. "Overview" — A brief description of what this disease/condition is
2. "Symptoms" — How to identify this disease on the plant
3. "Causes" — What causes this disease and how it spreads
4. "Treatment & Management" — Practical steps to treat and manage the disease
5. "Prevention" — How to prevent future occurrences

Additional rules:
- Set the summary field to a concise 1-2 sentence overview of the diagnosis.
- Be practical and actionable in your recommendations.
- Format all lists properly: each bullet point or numbered item MUST be on its own new line. Do not combine multiple list items into a single paragraph.
- Do not invent specific pesticide doses — instead recommend consulting a local agricultural extension officer for exact dosages.
- If the prediction indicates a healthy plant, adjust the response accordingly — explain what a healthy specimen looks like and general care tips.
- Never output raw internal chunk IDs or raw retrieval context in the answer.

Retrieved CABI information:

{context}
"""
        return self._generate_with_fallback(prompt)
