"""
LIME explainability service.

Generates visual explanations showing which image regions
contributed positively / negatively to the predicted class.
"""

import base64
import io
import logging

import numpy as np
from lime import lime_image
from PIL import Image
from skimage.segmentation import mark_boundaries

logger = logging.getLogger(__name__)


class LimeExplainer:
    """Wraps lime.lime_image for plant-disease predictions."""

    def __init__(self):
        self.explainer = lime_image.LimeImageExplainer()
        logger.info("LIME explainer initialised.")

    def explain(
        self,
        image_array: np.ndarray,
        predict_fn,
        num_samples: int = 300,
        top_labels: int = 1,
    ) -> dict:
        """
        Generate a LIME explanation for the given image.

        Parameters
        ----------
        image_array : np.ndarray
            (224, 224, 3) uint8 array.
        predict_fn : callable
            Accepts (N, 224, 224, 3) float32 [0-1] array, returns (N, 29) probs.
        num_samples : int
            Number of LIME perturbation samples.
        top_labels : int
            How many top labels to explain.

        Returns
        -------
        dict with keys: image (base64), top_positive_regions, top_negative_regions
        """
        explanation = self.explainer.explain_instance(
            image_array,
            classifier_fn=lambda imgs: predict_fn(imgs.astype(np.float32) / 255.0),
            top_labels=top_labels,
            hide_color=0,
            num_samples=num_samples,
        )

        top_label = explanation.top_labels[0]

        # Positive-only overlay
        temp_pos, mask_pos = explanation.get_image_and_mask(
            top_label,
            positive_only=True,
            num_features=5,
            hide_rest=False,
        )

        # Positive + negative overlay
        temp_full, mask_full = explanation.get_image_and_mask(
            top_label,
            positive_only=False,
            num_features=10,
            hide_rest=False,
        )

        overlay = mark_boundaries(temp_full / 255.0, mask_full)
        overlay_uint8 = (overlay * 255).astype(np.uint8)
        overlay_b64 = self._array_to_base64(overlay_uint8)

        # Extract region weights
        local_exp = explanation.local_exp[top_label]
        positive_regions = [
            {"segment": int(seg), "weight": round(float(w), 4)}
            for seg, w in local_exp if w > 0
        ][:5]
        negative_regions = [
            {"segment": int(seg), "weight": round(float(w), 4)}
            for seg, w in local_exp if w < 0
        ][:5]

        return {
            "image": overlay_b64,
            "top_positive_regions": positive_regions,
            "top_negative_regions": negative_regions,
        }

    @staticmethod
    def _array_to_base64(arr: np.ndarray) -> str:
        """Convert a uint8 numpy array to a base64-encoded PNG."""
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
