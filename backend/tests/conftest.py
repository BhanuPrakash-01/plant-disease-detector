"""
Shared test fixtures and configuration.
"""

import io
import os
import sys

import pytest
from PIL import Image

# Ensure the backend directory is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


@pytest.fixture(scope="session")
def sample_image_bytes():
    """Create a simple 224×224 RGB test image as bytes."""
    img = Image.new("RGB", (224, 224), color=(100, 180, 60))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.getvalue()


@pytest.fixture(scope="session")
def small_image_bytes():
    """Create a tiny 32×32 image (to test resizing)."""
    img = Image.new("RGB", (32, 32), color=(200, 50, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


@pytest.fixture(scope="session")
def invalid_image_bytes():
    """Create garbage bytes that aren't a valid image."""
    return b"this is definitely not an image file"
