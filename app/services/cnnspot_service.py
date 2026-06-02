"""
SMARTZI — CNNSpot / Wang2020 Global Feature Classifier
Loads a ResNet-50 fine-tuned on ProGAN with blur+JPEG augmentation
(blur_jpg_prob0.5.pth) and returns a probability that the input image
is AI-generated / synthetic.

Reference: Wang et al. "CNN-Generated Images Are Surprisingly Easy to
Spot… For Now" (CVPR 2020).
https://github.com/peterwang512/CNNDetection
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models

from app.core.config import settings

logger = logging.getLogger("smartzi.cnnspot")

def _select_device() -> "torch.device":
    """Pick the best available compute device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class CNNSpotService:
    """
    Thread-safe, lazily-initialised CNNSpot classifier.

    Weight file layout (Wang2020 checkpoint):
        state_dict keys are prefixed with ``model.`` and the final layer is
        ``model.fc`` with shape ``(1, 2048)`` (single sigmoid output).
    """

    # ImageNet normalization constants used during Wang2020 training
    _IMAGENET_MEAN = [0.485, 0.456, 0.406]
    _IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self) -> None:
        self._model: Optional[nn.Module] = None
        self._device: Optional[torch.device] = None
        self._transform = None
        self._lock = threading.Lock()
        self._init_attempted = False
        self._available = False
        self._reason = "uninitialized"

        # Resolve weight path from config (or env override)
        self._weights_path = Path(
            getattr(settings, "CNNSPOT_WEIGHTS_PATH", "")
            or os.path.join(
                os.path.dirname(__file__), "..", "..", "models", "cnnspot", "blur_jpg_prob0.5.pth"
            )
        )

    # ── Public API ────────────────────────────────────────────────────────

    @property
    def is_available(self) -> bool:
        return self._available

    def initialize(self) -> bool:
        """
        Lazy-load the ResNet-50 checkpoint.  Thread-safe; only runs once.

        Returns
        -------
        bool
            True if successfully loaded, False otherwise.
        """
        if self._init_attempted:
            return self._available

        with self._lock:
            if self._init_attempted:
                return self._available
            self._init_attempted = True

            self._device = _select_device()
            if self._device.type == "cpu":
                torch.set_num_threads(4)
                try:
                    torch.backends.mkldnn.enabled = True
                except Exception:
                    pass
            logger.info("CNNSpot device: %s", self._device)

            # ── Weight verification ───────────────────────────────────────
            if not self._weights_path.is_file():
                logger.warning("CNNSpot weights missing. Exiting initialization pipeline safely.")
                self._model = None
                self._available = False
                self._reason = "weights_missing"
                return False

            # ── Build model architecture ──────────────────────────────────
            try:
                model = models.resnet50(weights=None)
                # Wang2020 replaces the final fc with Linear(2048, 1) for
                # binary sigmoid classification.
                model.fc = nn.Linear(2048, 1)

                # ── Load checkpoint ───────────────────────────────────────
                checkpoint = torch.load(
                    str(self._weights_path),
                    map_location=self._device,
                    weights_only=False,
                )

                # Handle different checkpoint layouts:
                #   - Direct state_dict
                #   - Nested under "model" or "state_dict" key
                if isinstance(checkpoint, dict):
                    if "model" in checkpoint:
                        state_dict = checkpoint["model"]
                    elif "state_dict" in checkpoint:
                        state_dict = checkpoint["state_dict"]
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint

                # Strip "module." prefix from DataParallel checkpoints and
                # "model." prefix used by the Wang2020 training harness.
                cleaned: Dict[str, Any] = {}
                for key, value in state_dict.items():
                    clean_key = key
                    if clean_key.startswith("module."):
                        clean_key = clean_key[len("module."):]
                    if clean_key.startswith("model."):
                        clean_key = clean_key[len("model."):]
                    cleaned[clean_key] = value

                # Load with strict=False to tolerate minor mismatches
                # (e.g., extra running stats buffers)
                missing, unexpected = model.load_state_dict(cleaned, strict=False)
                if missing:
                    logger.warning("CNNSpot missing keys: %s", missing[:5])
                if unexpected:
                    logger.debug("CNNSpot unexpected keys: %s", unexpected[:5])

                model.to(self._device)
                model.eval()
                self._model = model
                self._available = True
                logger.info(
                    "CNNSpot ResNet-50 loaded | weights=%s | device=%s",
                    self._weights_path.name,
                    self._device,
                )

            except Exception as exc:
                logger.error("CNNSpot model initialization failed: %s", exc, exc_info=True)
                self._reason = "initialization_failed"
                self._available = False
                return False

            # ── Preprocessing transform (matches Wang2020 eval) ───────────
            self._transform = T.Compose([
                T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=self._IMAGENET_MEAN, std=self._IMAGENET_STD),
            ])

        return self._available

    def predict(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        Run CNNSpot inference on a single image.
        Returns a dict with inference results or unavailability reasons.
        """
        # Ensure model is loaded
        if not self.initialize():
            return {"available": False, "reason": self._reason}

        pil_img = Image.open(image_path).convert("RGB")
        tensor = self._transform(pil_img)
        if tensor.ndim != 3 or tensor.shape[0] != 3:
            raise ValueError("Unexpected tensor shape")
        tensor = tensor.unsqueeze(0).to(self._device)
        with torch.no_grad():
            logit = self._model(tensor)
            prob = float(torch.sigmoid(logit).cpu().item())
        prob = max(0.0, min(1.0, prob))
        prediction = "AI_GENERATED" if prob > 0.5 else "REAL"
        return {
            "available": True,
            "prediction": prediction,
            "confidence": round(prob, 6),
        }

    def dispose(self) -> None:
        """Release model weights and clear accelerator cache."""
        with self._lock:
            if self._model is not None:
                del self._model
                self._model = None
            self._available = False
            self._init_attempted = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("CNNSpot resources disposed.")


# ── Module-level singleton ────────────────────────────────────────────────────
cnnspot_service = CNNSpotService()