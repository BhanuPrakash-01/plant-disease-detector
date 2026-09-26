"""
Tests for the RAG retriever service.
"""

import pytest

from app.services.retriever import CabiRetriever


@pytest.fixture(scope="module")
def retriever():
    """Load retriever once for all tests."""
    return CabiRetriever()


class TestRetriever:

    def test_retrieve_returns_results(self, retriever):
        """A valid query should return non-empty results."""
        results = retriever.retrieve("tomato early blight symptoms")
        assert len(results) > 0

    def test_retrieve_metadata_fields(self, retriever):
        """Each result should contain required metadata."""
        results = retriever.retrieve("potato late blight")
        for r in results:
            assert "document_id" in r
            assert "title" in r
            assert "section" in r
            assert "text" in r
            assert "score" in r

    def test_retrieve_top_k(self, retriever):
        """Should return at most top_k results."""
        results = retriever.retrieve("apple scab", top_k=3)
        assert len(results) <= 3

    def test_retrieve_scores_are_float(self, retriever):
        """Scores should be floats."""
        results = retriever.retrieve("grape black rot")
        for r in results:
            assert isinstance(r["score"], float)

    def test_retrieve_default_top_k(self, retriever):
        """Default top_k should return 5 results."""
        results = retriever.retrieve("corn leaf blight")
        assert len(results) == 5

    def test_retrieve_broader_topic(self, retriever):
        """RAG should handle topics beyond the 29 classifier classes."""
        results = retriever.retrieve("rice blast disease")
        assert len(results) > 0
