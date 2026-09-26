"""
Plant Disease AI — FastAPI application.

All heavy models/artifacts are loaded once during startup
via the lifespan context manager and stored in app.state.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import diagnosis, rag
from app.services.classifier import PlantClassifier
from app.services.generator import GeminiGenerator
from app.services.lime_service import LimeExplainer
from app.services.retriever import CabiRetriever
from app.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan: load models once at startup ─────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all ML models and artifacts on startup, clean up on shutdown."""

    logger.info("═══ Starting model loading ═══")

    # 1. Classifier
    app.state.classifier = PlantClassifier(settings.model_path)

    # 2. LIME
    app.state.lime_explainer = LimeExplainer()

    # 3. RAG retriever (loads FAISS + BGE-M3 + CABI metadata)
    app.state.retriever = CabiRetriever()

    # 4. Gemini generator
    app.state.generator = GeminiGenerator()

    logger.info("═══ All models loaded — ready to serve ═══")
    yield
    logger.info("═══ Shutting down ═══")


# ── Application ───────────────────────────────────────────────────────

app = FastAPI(
    title="Plant Disease AI Assistant",
    description="Image classification + Explainable AI + RAG-powered plant health assistant",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(rag.router)
app.include_router(diagnosis.router)


@app.get("/health")
async def health():
    """Simple health-check endpoint."""
    return {"status": "ok"}
