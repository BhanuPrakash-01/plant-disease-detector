"""
Tests for the classifier service.
"""

import numpy as np
import pytest

from app.services.classifier import CLASS_NAMES, PlantClassifier
from app.config import settings


@pytest.fixture(scope="module")
def classifier():
    """Load the classifier once for all tests in this module."""
    return PlantClassifier(settings.model_path)


class TestClassifier:

    def test_class_count(self):
        """Exactly 29 classes must be defined."""
        assert len(CLASS_NAMES) == 29

    def test_predict_valid_image(self, classifier, sample_image_bytes):
        """Prediction should return class, confidence, and probabilities."""
        result = classifier.predict(sample_image_bytes)
        assert "class" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert result["class"] in CLASS_NAMES
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_probability_shape(self, classifier, sample_image_bytes):
        """Probabilities array must have 29 elements."""
        result = classifier.predict(sample_image_bytes)
        assert result["probabilities"].shape == (29,)

    def test_predict_probabilities_sum(self, classifier, sample_image_bytes):
        """Probabilities should sum to approximately 1.0."""
        result = classifier.predict(sample_image_bytes)
        assert abs(result["probabilities"].sum() - 1.0) < 0.01

    def test_predict_small_image(self, classifier, small_image_bytes):
        """Smaller images should be resized and predicted correctly."""
        result = classifier.predict(small_image_bytes)
        assert result["class"] in CLASS_NAMES

    def test_predict_invalid_image(self, classifier, invalid_image_bytes):
        """Invalid image bytes should raise an error."""
        with pytest.raises(Exception):
            classifier.predict(invalid_image_bytes)

    def test_bytes_to_array_shape(self, classifier, sample_image_bytes):
        """bytes_to_array should return (224, 224, 3) uint8."""
        arr = classifier.bytes_to_array(sample_image_bytes)
        assert arr.shape == (224, 224, 3)
        assert arr.dtype == np.uint8

    def test_predict_batch(self, classifier, sample_image_bytes):
        """predict_batch should work with a batch of images."""
        arr = classifier.bytes_to_array(sample_image_bytes)
        batch = np.expand_dims(arr, 0).astype(np.float32) / 255.0
        probs = classifier.predict_batch(batch)
        assert probs.shape == (1, 29)
