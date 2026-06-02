"""
SMARTZI - Traditional Forensic Signals (ELA, FFT, Noise, Metadata)
Computes basic pixel, frequency, and metadata statistics.
All scores are normalized strictly between 0.0 and 1.0.
"""

import logging
from typing import Dict

import cv2
import numpy as np
from PIL import Image, ImageChops

from app.services.metadata_service import metadata_extractor

logger = logging.getLogger("smartzi.forensic_signals")


def compute_forensic_signals(image_path: str) -> Dict[str, float]:
    """
    Computes four independent forensic signals.
    If a signal fails, it falls back to 0.0 to prevent pipeline crashes.
    """
    scores = {
        "ela_score": 0.0,
        "fft_score": 0.0,
        "noise_score": 0.0,
        "metadata_score": 0.0,
    }

    # ── 1. ELA (Error Level Analysis) ─────────────────────────────────────────
    try:
        # Resave at 90% quality and diff
        original = Image.open(image_path).convert("RGB")
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            original.save(tmp_path, "JPEG", quality=90)
            resaved = Image.open(tmp_path)
            
            diff = ImageChops.difference(original, resaved)
            extrema = diff.getextrema()
            max_diff = max([ex[1] for ex in extrema]) if extrema else 1
            if max_diff == 0:
                max_diff = 1
                
            scale = 255.0 / max_diff
            diff_scaled = Image.eval(diff, lambda x: x * scale)
            
            # Simple average intensity of scaled diff
            arr = np.array(diff_scaled)
            ela_mean = np.mean(arr)
            # Normalize to 0-1 (heuristically, >30 is very high for 90% quality resave)
            scores["ela_score"] = float(np.clip(ela_mean / 30.0, 0.0, 1.0))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    except Exception as e:
        logger.debug("ELA signal failed: %s", e)

    # ── 2. FFT (Fast Fourier Transform) ───────────────────────────────────────
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
        
        # High frequency power ratio, normalized
        scores["fft_score"] = float(np.clip(hp / (lp + 1e-6) * 1.5, 0.0, 1.0))
    except Exception as e:
        logger.debug("FFT signal failed: %s", e)

    # ── 3. Noise Analysis ────────────────────────────────────────────────────
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
            scores["noise_score"] = float(np.clip(sv / (mv + 1e-6) * 0.5, 0.0, 1.0))
    except Exception as e:
        logger.debug("Noise signal failed: %s", e)

    # ── 4. Metadata ──────────────────────────────────────────────────────────
    try:
        meta = metadata_extractor.extract(image_path)
        if not meta.has_exif:
            scores["metadata_score"] = 0.25
        else:
            sw = (meta.software or "").lower()
            if any(k in sw for k in ("midjourney", "stable diffusion", "dall-e", "firefly")):
                scores["metadata_score"] = 0.85
            elif any(k in sw for k in ("photoshop", "gimp", "lightroom")):
                scores["metadata_score"] = 0.55
            else:
                scores["metadata_score"] = 0.15
    except Exception as e:
        logger.debug("Metadata signal failed: %s", e)

    return scores
