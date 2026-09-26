"""
Tests for the resilient Gemini generator service.
"""

from unittest.mock import MagicMock, patch

import pytest
from google.genai.errors import APIError

from app.services.generator import GeminiGenerator, GeminiStructuredOutput


@pytest.fixture
def generator():
    """Initialise the Gemini generator with dummy keys for testing fallback logic."""
    with patch("app.config.settings.gemini_api_key", "dummy"):
        return GeminiGenerator()


MOCK_RESULTS = [
    {
        "document_id": "test_doc_1",
        "title": "Tomato Early Blight Management",
        "section": "Symptoms",
        "text": "Early blight causes dark, concentric spots on older leaves.",
    }
]


def test_build_context(generator):
    context = generator.build_context(MOCK_RESULTS)
    assert "SOURCE 1" in context
    assert "Tomato Early Blight Management" in context


@patch("app.services.generator.genai.Client")
def test_successful_generation(mock_client_class, generator):
    """1. Successful Gemini request."""
    mock_client = mock_client_class.return_value
    mock_response = MagicMock()
    mock_response.parsed = GeminiStructuredOutput(
        status="success",
        summary="Test summary",
        sections=[],
    )
    mock_client.models.generate_content.return_value = mock_response
    generator.client = mock_client

    result = generator.generate("Test question", MOCK_RESULTS)
    assert result.status == "success"
    assert result.summary == "Test summary"
    # Should only be called once, with the primary model
    mock_client.models.generate_content.assert_called_once()
    assert mock_client.models.generate_content.call_args.kwargs["model"] == generator.primary_model


@patch("app.services.generator.genai.Client")
@patch("app.services.generator.time.sleep", return_value=None)  # skip wait times
def test_temporary_failure_and_retry(mock_sleep, mock_client_class, generator):
    """2. Temporary 502/503 failure and retry."""
    mock_client = mock_client_class.return_value
    
    # 2 failures (502), then 1 success
    error_502 = APIError(502, {})
    mock_success = MagicMock()
    mock_success.parsed = GeminiStructuredOutput(
        status="success", summary="Retry success", sections=[]
    )
    
    mock_client.models.generate_content.side_effect = [error_502, error_502, mock_success]
    generator.client = mock_client

    result = generator.generate("Retry question", MOCK_RESULTS)
    assert result.status == "success"
    assert result.summary == "Retry success"
    assert mock_client.models.generate_content.call_count == 3
    # All 3 calls should be to primary model
    for call in mock_client.models.generate_content.call_args_list:
        assert call.kwargs["model"] == generator.primary_model


@patch("app.services.generator.genai.Client")
@patch("app.services.generator.time.sleep", return_value=None)
def test_primary_failure_fallback_success(mock_sleep, mock_client_class, generator):
    """3. Primary-model failure followed by fallback model."""
    mock_client = mock_client_class.return_value
    
    # Primary fails permanently (e.g., 400 Bad Request, which we don't retry)
    error_400 = APIError(400, {})
    mock_success = MagicMock()
    mock_success.parsed = GeminiStructuredOutput(
        status="success", summary="Fallback success", sections=[]
    )
    
    # The first call (primary) raises 400.
    # The second call (fallback) succeeds.
    mock_client.models.generate_content.side_effect = [error_400, mock_success]
    generator.client = mock_client

    result = generator.generate("Fallback question", MOCK_RESULTS)
    assert result.status == "success"
    assert result.summary == "Fallback success"
    assert mock_client.models.generate_content.call_count == 2
    
    calls = mock_client.models.generate_content.call_args_list
    assert calls[0].kwargs["model"] == generator.primary_model
    assert calls[1].kwargs["model"] == generator.fallback_model


@patch("app.services.generator.genai.Client")
@patch("app.services.generator.time.sleep", return_value=None)
def test_both_models_unavailable(mock_sleep, mock_client_class, generator):
    """4. Both models unavailable."""
    mock_client = mock_client_class.return_value
    
    # Both fail permanently
    error_400 = APIError(400, {})
    mock_client.models.generate_content.side_effect = [error_400, error_400]
    generator.client = mock_client

    result = generator.generate("Impossible question", MOCK_RESULTS)
    
    # Should not crash, should return graceful application error
    assert result.status == "generation_unavailable"
    assert mock_client.models.generate_content.call_count == 2


@patch("app.services.generator.genai.Client")
def test_insufficient_source_response(mock_client_class, generator):
    """6. Insufficient-source response."""
    mock_client = mock_client_class.return_value
    mock_response = MagicMock()
    mock_response.parsed = GeminiStructuredOutput(
        status="insufficient_information",
        summary="No data found.",
        sections=[],
    )
    mock_client.models.generate_content.return_value = mock_response
    generator.client = mock_client

    result = generator.generate_with_prediction("Treat?", "Unknown", 0.1, [])
    assert result.status == "insufficient_information"
    assert mock_client.models.generate_content.call_count == 1
