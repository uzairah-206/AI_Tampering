"""
SMARTZI - Statistical forensic signals (FFT, noise, metadata heuristics).
"""

import logging
from typing import Any, Dict, Optional

import cv2
import numpy as np
from PIL import Image

from app.services.metadata_service import metadata_extractor

logger = logging.getLogger("smartzi.forensic_stats")


def compute_statistical_signals(image_path: str) -> Dict[str, Any]:
    """Return normalized scores in [0,1] or None per signal."""
    out: Dict[str, Any] = {
        "fft": None,
        "noise": None,
        "metadata": None,
        "source": "statistical",
    }

    try:
        gray = np.array(Image.open(image_path).convert("L"))
        h, w = gray.shape

        f_shift = np.fft.fftshift(np.fft.fft2(gray))
        mag = 20 * np.log(np.abs(f_shift) + 1e-9)
        cy, cx = h // 2, w // 2
        r_low = min(h, w) // 10
        y_i, x_i = np.ogrid[:h, :w]
        dist = np.sqrt((y_i - cy) ** 2 + (x_i - cx) ** 2)
        low_m, high_m = dist <= r_low, (dist > r_low) & (dist <= min(h, w) // 3)
        lp = float(np.mean(mag[low_m])) if np.any(low_m) else 1.0
        hp = float(np.mean(mag[high_m])) if np.any(high_m) else 0.0
        out["fft"] = float(np.clip(hp / (lp + 1e-6) * 1.5, 0.0, 1.0))
    except Exception as e:
        logger.debug("FFT signal failed: %s", e)

    try:
        gray = np.array(Image.open(image_path).convert("L"))
        h, w = gray.shape
        residual = cv2.absdiff(gray, cv2.medianBlur(gray, 3))
        bs = 16
        vars_ = [
            float(np.var(residual[i * bs:(i + 1) * bs, j * bs:(j + 1) * bs]))
            for i in range(h // bs) for j in range(w // bs)
        ]
        if vars_:
            mv, sv = np.mean(vars_), np.std(vars_)
            out["noise"] = float(np.clip(sv / (mv + 1e-6) * 0.5, 0.0, 1.0))
    except Exception as e:
        logger.debug("Noise signal failed: %s", e)

    try:
        meta = metadata_extractor.extract(image_path)
        if not meta.has_exif:
            out["metadata"] = 0.25
        else:
            sw = (meta.software or "").lower()
            if any(k in sw for k in ("midjourney", "stable diffusion", "dall-e", "firefly")):
                out["metadata"] = 0.85
            elif any(k in sw for k in ("photoshop", "gimp", "lightroom")):
                out["metadata"] = 0.55
            else:
                out["metadata"] = 0.15
    except Exception as e:
        logger.debug("Metadata signal failed: %s", e)

    return out


def aggregate_statistical_score(signals: Dict[str, Any]) -> Optional[float]:
    pairs = [
        (signals.get("fft"), 0.4),
        (signals.get("noise"), 0.35),
        (signals.get("metadata"), 0.25),
    ]
    active = [(s, w) for s, w in pairs if s is not None]
    if not active:
        return None
    wsum = sum(w for _, w in active)
    return sum(s * w / wsum for s, w in active)
