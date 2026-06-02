"""
SMARTZI - Dynamic Confidence Fusion Engine
Dynamically renormalizes active signals to a sum of 1.0.
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger("smartzi.confidence_fusion")

BASE_WEIGHTS = {
    "cnnspot": 0.35,
    "trufor": 0.25,
    "ela": 0.10,
    "fft": 0.10,
    "noise": 0.10,
    "metadata": 0.05,
}

def fuse_confidence_scores(
    cnnspot_conf: Optional[float] = None,
    trufor_conf: Optional[float] = None,
    ela_conf: Optional[float] = None,
    fft_conf: Optional[float] = None,
    noise_conf: Optional[float] = None,
    metadata_conf: Optional[float] = None,
    gemini_conf: Optional[float] = None,
) -> float:
    """
    Computes a dynamically renormalized confidence score.
    If a signal is None, its weight is dropped and the rest are scaled up.
    """
    signals = {
        "cnnspot": cnnspot_conf,
        "trufor": trufor_conf,
        "ela": ela_conf,
        "fft": fft_conf,
        "noise": noise_conf,
        "metadata": metadata_conf,
    }

    active: Dict[str, float] = {}
    for name, score in signals.items():
        if score is not None:
            active[name] = max(0.0, min(1.0, float(score)))

    if not active:
        logger.warning("No active confidence signals for fusion.")
        return 0.0

    active_weight_sum = sum(BASE_WEIGHTS[n] for n in active)
    if active_weight_sum <= 0.0:
        return 0.0

    # Apply penalty factor to avoid over‑confidence when many signals are active.
    # The factor 0.85 reduces the fused score slightly, encouraging downstream logic
    # to request additional verification (e.g., Gemini multimodal check) rather than
    # settling on an "Uncertain" gray‑area classification.
    fused = sum(
        active[n] * (BASE_WEIGHTS[n] / active_weight_sum) * 0.85
        for n in active
    )
    logger.info("Fusion | active=%s score=%.4f (weight_sum=%.2f)", list(active), fused, active_weight_sum)
    return float(max(0.0, min(1.0, fused)))
