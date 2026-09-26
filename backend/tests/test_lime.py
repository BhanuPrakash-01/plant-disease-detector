"""
Tests for LIME explainability service.
"""

import numpy as np
import pytest

from app.services.classifier import PlantClassifier
from app.services.lime_service import LimeExplainer
from app.config import settings


@pytest.fixture(scope="module")
def classifier():
    return PlantClassifier(settings.model_path)


@pytest.fixture(scope="module")
def lime_explainer():
    return LimeExplainer()


class TestLime:

    def test_explain_returns_dict(self, classifier, lime_explainer, sample_image_bytes):
        """LIME explain should return a dict with image, positive, and negative regions."""
        img_array = classifier.bytes_to_array(sample_image_bytes)
        result = lime_explainer.explain(
            img_array,
            classifier.predict_batch,
            num_samples=50,  # fewer samples for faster test
        )
        assert "image" in result
        assert "top_positive_regions" in result
        assert "top_negative_regions" in result

    def test_explain_image_is_base64(self, classifier, lime_explainer, sample_image_bytes):
        """The LIME overlay image should be a non-empty base64 string."""
        img_array = classifier.bytes_to_array(sample_image_bytes)
        result = lime_explainer.explain(
            img_array,
            classifier.predict_batch,
            num_samples=50,
        )
        assert isinstance(result["image"], str)
        assert len(result["image"]) > 100  # it's a PNG, should be substantial

    def test_explain_regions_format(self, classifier, lime_explainer, sample_image_bytes):
        """Positive/negative regions should be lists of dicts with segment/weight."""
        img_array = classifier.bytes_to_array(sample_image_bytes)
        result = lime_explainer.explain(
            img_array,
            classifier.predict_batch,
            num_samples=50,
        )
        for region in result["top_positive_regions"]:
            assert "segment" in region
            assert "weight" in region
