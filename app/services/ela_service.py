"""
SMARTZI - Error Level Analysis (ELA) Service
Detects image manipulation by analysing JPEG re-compression artifacts.

How ELA works:
1. Re-save the image at a known JPEG quality (e.g. 75)
2. Compute the absolute pixel difference between original and re-saved
3. Amplify differences for visibility
4. High-ELA regions indicate potential tampering (copy-paste, splicing)
"""

import io
import base64
import logging
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance

from app.core.config import settings
from app.schemas.analysis import ELAResult

logger = logging.getLogger("smartzi.ela")


class ELAService:
    """
    Error Level Analysis implementation using PIL and NumPy.
    Produces both numeric metrics and a base64-encoded heatmap PNG.
    """

    def __init__(self, quality: int = 75, scale: int = 15):
        """
        Args:
            quality: JPEG quality used for re-compression (lower → more artifacts)
            scale:   Amplification factor applied to the ELA difference image
        """
        self.quality = quality
        self.scale = scale

    def analyze(self, image_path: str) -> ELAResult:
        """
        Run ELA on the given image file and return structured metrics + heatmap.
        """
        try:
            with Image.open(image_path) as img:
                original = img.convert("RGB")
        except Exception as e:
            logger.error("Cannot open image for ELA: %s", e)
            raise ValueError(f"Cannot open image: {e}")

        # Step 1: Re-compress at target quality
        recompressed = self._recompress(original)

        # Step 2: Compute absolute difference
        diff = ImageChops.difference(original, recompressed)

        # Step 3: Amplify the difference
        diff_array = np.array(diff, dtype=np.float32)
        amplified = np.clip(diff_array * self.scale, 0, 255).astype(np.uint8)

        # Step 4: Compute metrics
        ela_mean = float(np.mean(amplified))
        ela_max = float(np.max(amplified))
        ela_std = float(np.std(amplified))

        # Step 5: Count suspicious regions (high-intensity blobs)
        suspicious_regions = self._count_suspicious_regions(amplified)

        # Step 6: Generate colourmap heatmap
        heatmap_b64 = self._generate_heatmap(amplified, original.size)

        logger.debug(
            "ELA complete | mean=%.2f max=%.2f std=%.2f regions=%d",
            ela_mean, ela_max, ela_std, suspicious_regions,
        )

        return ELAResult(
            ela_mean=round(ela_mean, 3),
            ela_max=round(ela_max, 3),
            ela_std=round(ela_std, 3),
            suspicious_regions=suspicious_regions,
            heatmap_base64=heatmap_b64,
        )

    # ── Private Helpers ───────────────────────────────────────────────────────
    def _recompress(self, img: Image.Image) -> Image.Image:
        """Save image to a BytesIO buffer at target quality and reload it."""
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=self.quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    def _count_suspicious_regions(self, amplified: np.ndarray) -> int:
        """
        Count connected high-intensity regions via contour detection.
        Regions with mean intensity above threshold are flagged as suspicious.
        """
        THRESHOLD = 30  # Pixel intensity threshold
        gray = cv2.cvtColor(amplified, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, THRESHOLD, 255, cv2.THRESH_BINARY)

        # Morphological closing to merge nearby blobs
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter out tiny noise contours (< 100 px²)
        significant = [c for c in contours if cv2.contourArea(c) > 100]
        return len(significant)

    def _generate_heatmap(self, amplified: np.ndarray, original_size: Tuple[int, int]) -> str:
        """
        Apply a JET colormap to the grayscale ELA diff and return as base64 PNG.
        The heatmap is resized to a reasonable display size.
        """
        DISPLAY_MAX = 512  # Max side length for heatmap
        gray = cv2.cvtColor(amplified, cv2.COLOR_RGB2GRAY)
        colored = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

        # Resize if image is very large
        h, w = gray.shape
        if max(h, w) > DISPLAY_MAX:
            scale = DISPLAY_MAX / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            colored = cv2.resize(colored, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Convert BGR→RGB, encode to PNG bytes, base64-encode
        rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG", optimize=True)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


# Module-level singleton using configured defaults
ela_service = ELAService(
    quality=settings.ELA_QUALITY,
    scale=settings.ELA_SCALE,
)
