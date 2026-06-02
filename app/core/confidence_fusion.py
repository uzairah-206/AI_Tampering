"""
SMARTZI - Confidence Fusion Engine
Weights sum to 1.0; unavailable sources are removed and weights renormalized.
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger("smartzi.confidence_fusion")

BASE_WEIGHTS = {
    "gemini": 0.25,
    "trufor": 0.25,
    "mantranet": 0.20,
    "ela": 0.15,
    "metadata": 0.05,
    "noise": 0.05,
    "fft": 0.05,
}


def fuse_confidence_scores(
    trufor_conf: Optional[float] = None,
    mantranet_conf: Optional[float] = None,
    ela_conf: Optional[float] = None,
    metadata_conf: Optional[float] = None,
    noise_conf: Optional[float] = None,
    fft_conf: Optional[float] = None,
    gemini_conf: Optional[float] = None,
    *,
    cnn_conf: Optional[float] = None,
    ai_conf: Optional[float] = None,
    ai_detector_conf: Optional[float] = None,
) -> float:
    """Σ(score × normalized_weight). Returns 0.0 when no active signals."""
    if trufor_conf is None and cnn_conf is not None:
        trufor_conf = cnn_conf
    if ai_conf is not None and trufor_conf is None:
        trufor_conf = ai_conf
    if fft_conf is None and ai_detector_conf is not None:
        fft_conf = ai_detector_conf

    signals = {
        "gemini": gemini_conf,
        "trufor": trufor_conf,
        "mantranet": mantranet_conf,
        "ela": ela_conf,
        "metadata": metadata_conf,
        "noise": noise_conf,
        "fft": fft_conf,
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

    fused = sum(
        active[n] * (BASE_WEIGHTS[n] / active_weight_sum)
        for n in active
    )
    logger.info("Fusion | active=%s score=%.4f", list(active), fused)
    return fused
