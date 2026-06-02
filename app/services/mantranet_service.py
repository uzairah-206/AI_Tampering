"""
SMARTZI - ManTraNet Service
Wraps the ManTraNet model for pixel-level forgery detection & localization.

Produces:
  - Forgery probability score (0–1)
  - Tampered region count
  - Base64-encoded forgery heatmap
"""

import io
import base64
import logging
import os
from typing import Optional

import cv2
import numpy as np

# Safe PyTorch import sequence
TORCH_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError as e:
    logger = logging.getLogger("smartzi.mantranet")
    logger.error("Failed to load PyTorch: %s. ManTraNet will be disabled.", e)

from PIL import Image

from app.core.config import settings
from app.schemas.analysis import MantraNetResult

logger = logging.getLogger("smartzi.mantranet")

# Maximum dimension for input images (ManTraNet is memory-hungry)
MAX_INPUT_DIM = 512


class MantraNetService:
    """
    ManTraNet inference service.

    Loads the pre-trained MantraNetv4 weights at startup.
    Falls back gracefully if weights are not available — the analysis
    pipeline will still work using the other signals (ELA + CNN + metadata).
    """

    def __init__(self):
        if TORCH_AVAILABLE:
            self.device = torch.device(settings.DEVICE)
        else:
            self.device = None
        self.model = None
        self._available = False
        # Deprecated: ManTraNet disabled to prevent crashes

    def _load_model(self):
        """Load ManTraNet pre-trained weights."""
        weight_path = settings.MANTRANET_WEIGHTS_PATH
        WEIGHTS_URL = "https://github.com/RonyAbecidan/ManTraNet-pytorch/raw/main/MantraNet/MantraNetv4.pt"

        def _download_weights():
            import urllib.request
            os.makedirs(os.path.dirname(weight_path), exist_ok=True)
            logger.info("Downloading ManTraNet weights from %s...", WEIGHTS_URL)
            urllib.request.urlretrieve(WEIGHTS_URL, weight_path)
            logger.info("ManTraNet weights downloaded successfully.")

        if not os.path.exists(weight_path):
            try:
                _download_weights()
            except Exception as e:
                logger.error("Failed to download ManTraNet weights: %s", e)
                return
        else:
            # Check for corruption/truncation (~73MB expected)
            size_mb = os.path.getsize(weight_path) / (1024 * 1024)
            if size_mb < 50.0:
                logger.warning("ManTraNet weights are truncated/corrupted (%.1f MB). Re-downloading...", size_mb)
                try:
                    os.remove(weight_path)
                    _download_weights()
                except Exception as e:
                    logger.error("Failed to re-download ManTraNet weights: %s", e)
                    return

        try:
            # Dynamic import of mantranet module to prevent top-level PyTorch import errors
            from app.services.mantranet.mantranet import load_pretrained_mantranet
            self.model = load_pretrained_mantranet(weight_path, device=self.device)
            self._available = True
            logger.info("ManTraNet loaded successfully from %s", weight_path)
        except Exception as e:
            logger.error("Failed to load ManTraNet: %s", e)

    @property
    def is_available(self) -> bool:
        return TORCH_AVAILABLE and self._available

    def analyze(self, image_path: str) -> Optional[MantraNetResult]:
        """
        Run ManTraNet forgery detection on the given image.

        Returns MantraNetResult with forgery score, region count, and heatmap,
        or None if the model is not available.
        """
        if not TORCH_AVAILABLE or not self._available:
            return None

        try:
            with Image.open(image_path) as img:
                img_rgb = img.convert("RGB")
        except Exception as e:
            logger.error("Cannot open image for ManTraNet: %s", e)
            return None

        try:
            # Resize large images to avoid OOM
            img_array = np.array(img_rgb)
            h, w = img_array.shape[:2]
            scale = 1.0
            if max(h, w) > MAX_INPUT_DIM:
                scale = MAX_INPUT_DIM / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                img_array = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Convert to tensor: (1, 3, H, W), float32, pixel values [0, 255]
            tensor = torch.tensor(img_array, dtype=torch.float32)
            tensor = tensor.unsqueeze(0).permute(0, 3, 1, 2)  # (1, 3, H, W)
            tensor = tensor.to(self.device)

            # Run inference with torch.no_grad()
            with torch.no_grad():
                output = self.model(tensor)  # (1, 1, H, W) — forgery probability

            # Convert to numpy
            forgery_map = output[0, 0].cpu().numpy()  # (H, W) values in [0, 1]

            # Compute metrics
            forgery_score = float(np.mean(forgery_map))
            forgery_max = float(np.max(forgery_map))

            # Count tampered regions (threshold at 0.3)
            THRESHOLD = 0.3
            binary_mask = (forgery_map > THRESHOLD).astype(np.uint8) * 255
            kernel = np.ones((5, 5), np.uint8)
            closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            tampered_regions = len([c for c in contours if cv2.contourArea(c) > 50])

            # Compute tampered area percentage
            tampered_pixels = np.sum(forgery_map > THRESHOLD)
            total_pixels = forgery_map.shape[0] * forgery_map.shape[1]
            tampered_area_pct = round(float(tampered_pixels / total_pixels) * 100, 2)

            # Generate colored heatmap (JET colormap)
            heatmap_b64 = self._generate_heatmap(forgery_map)

            logger.debug(
                "ManTraNet | score=%.4f max=%.4f regions=%d area=%.1f%%",
                forgery_score, forgery_max, tampered_regions, tampered_area_pct,
            )

            return MantraNetResult(
                forgery_score=round(forgery_score, 4),
                forgery_max=round(forgery_max, 4),
                tampered_regions=tampered_regions,
                tampered_area_pct=tampered_area_pct,
                heatmap_base64=heatmap_b64,
            )

        except Exception as e:
            logger.error("ManTraNet inference failed: %s", e, exc_info=True)
            return None

    def _generate_heatmap(self, forgery_map: np.ndarray) -> str:
        """Generate a base64-encoded heatmap PNG from the forgery probability map."""
        DISPLAY_MAX = 512
        h, w = forgery_map.shape

        # Scale to 0-255
        heatmap_uint8 = (forgery_map * 255).astype(np.uint8)

        # Apply JET colormap
        colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # Resize if needed
        if max(h, w) > DISPLAY_MAX:
            sc = DISPLAY_MAX / max(h, w)
            colored = cv2.resize(colored, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)

        # Convert BGR → RGB, encode to PNG
        rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG", optimize=True)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


# Module-level singleton
mantranet_service = MantraNetService()
