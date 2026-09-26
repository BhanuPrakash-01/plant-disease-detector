"""
API integration tests.

Uses FastAPI's TestClient to test all three endpoints
without starting a real server.
"""

import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


@pytest.fixture(scope="module")
def client():
    """Create a test client. The lifespan loads all models."""
    with TestClient(app) as c:
        yield c


def _make_image_file(format="JPEG"):
    """Create an in-memory image file for uploads."""
    img = Image.new("RGB", (224, 224), color=(100, 180, 60))
    buf = io.BytesIO()
    img.save(buf, format=format)
    buf.seek(0)
    return buf


class TestHealthEndpoint:

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestRagEndpoint:

    def test_ask_valid_question(self, client):
        """POST /api/rag/ask with a valid question."""
        resp = client.post(
            "/api/rag/ask",
            json={"question": "What are symptoms of tomato early blight?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "sources" in data
        
        if data["status"] == "generation_unavailable":
            # Valid graceful fallback
            assert data["summary"] is not None
        else:
            assert data["status"] in ("success", "insufficient_information")
            assert len(data["sources"]) > 0

    def test_ask_empty_question(self, client):
        """Empty question should be rejected."""
        resp = client.post("/api/rag/ask", json={"question": ""})
        assert resp.status_code == 422  # validation error

    def test_ask_missing_question(self, client):
        """Missing question field should be rejected."""
        resp = client.post("/api/rag/ask", json={})
        assert resp.status_code == 422


class TestDiagnosisPredictEndpoint:

    def test_predict_valid_image(self, client):
        """POST /api/diagnosis/predict with a valid image."""
        img_buf = _make_image_file()
        resp = client.post(
            "/api/diagnosis/predict",
            files={"image": ("test.jpg", img_buf, "image/jpeg")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "prediction" in data
        assert "lime" in data
        assert data["prediction"]["predicted_class"]
        assert 0 <= data["prediction"]["confidence"] <= 1
        assert data["lime"]["image"]  # base64 string

    def test_predict_no_image(self, client):
        """Missing image should be rejected."""
        resp = client.post("/api/diagnosis/predict")
        assert resp.status_code == 422


class TestDiagnosisAnalyzeEndpoint:

    def test_analyze_image_only(self, client):
        """Image only → prediction + LIME, no answer."""
        img_buf = _make_image_file()
        resp = client.post(
            "/api/diagnosis/analyze",
            files={"image": ("test.jpg", img_buf, "image/jpeg")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["prediction"] is not None
        assert data["lime"] is not None
        assert data.get("rag") is None

    def test_analyze_question_only(self, client):
        """Question only → RAG answer, no prediction."""
        resp = client.post(
            "/api/diagnosis/analyze",
            data={"question": "What causes apple scab?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["prediction"] is None
        assert data["rag"] is not None
        assert "status" in data["rag"]

    def test_analyze_image_and_question(self, client):
        """Image + question → prediction + LIME + RAG answer."""
        img_buf = _make_image_file()
        resp = client.post(
            "/api/diagnosis/analyze",
            files={"image": ("test.jpg", img_buf, "image/jpeg")},
            data={"question": "What treatment should I use?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["prediction"] is not None
        assert data["lime"] is not None
        assert data["rag"] is not None
        assert "status" in data["rag"]

    def test_analyze_neither(self, client):
        """No image and no question should be rejected."""
        resp = client.post("/api/diagnosis/analyze")
        assert resp.status_code == 400
