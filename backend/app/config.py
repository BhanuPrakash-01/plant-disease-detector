"""
Centralised application configuration.

All settings are loaded from environment variables / .env file.
"""

from pathlib import Path

from pydantic_settings import BaseSettings


# ── paths ─────────────────────────────────────────────────────────────
BACKEND_DIR = Path(__file__).resolve().parents[1]          # backend/
MODEL_DIR = BACKEND_DIR / "models"
RAG_DIR = BACKEND_DIR / "rag"


class Settings(BaseSettings):
    """Application settings — sourced from .env"""

    # Gemini
    gemini_api_key: str
    gemini_primary_model: str = "gemini-3.1-flash-lite"
    gemini_fallback_model: str = "gemini-3.5-flash-lite"

    # Embedding
    embedding_model: str = "BAAI/bge-m3"

    # RAG
    rag_top_k: int = 5

    # Paths (can be overridden, but sensible defaults)
    model_path: str = str(MODEL_DIR / "efficientnetb0_stage2_best.keras")
    faiss_index_path: str = str(RAG_DIR / "cabi_faiss.index")
    chunks_csv_path: str = str(RAG_DIR / "cabi_chunks.csv")

    model_config = {"env_file": str(BACKEND_DIR / ".env"), "extra": "ignore"}


# Singleton
settings = Settings()
