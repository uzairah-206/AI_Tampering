# backend/tests/test_fusion.py
import pytest
from app.services.confidence_fusion import fuse_confidence_scores, BASE_WEIGHTS

def test_fusion_all_active():
    """Verify fusion with all signals populated applying the 0.85 penalty."""
    score = fuse_confidence_scores(
        cnnspot_conf=0.8,
        trufor_conf=0.6,
        ela_conf=0.5,
        fft_conf=0.4,
        noise_conf=0.3,
        metadata_conf=0.2,
        gemini_conf=None # not in the active list of base weights dict in app.services.confidence_fusion
    )
    # Weights sum: cnnspot(0.35) + trufor(0.25) + ela(0.10) + fft(0.10) + noise(0.10) + metadata(0.05) = 0.95
    # Weighted sum: 0.8*0.35 + 0.6*0.25 + 0.5*0.10 + 0.4*0.10 + 0.3*0.10 + 0.2*0.05 = 0.28 + 0.15 + 0.05 + 0.04 + 0.03 + 0.01 = 0.56
    # Normalized: 0.56 / 0.95 = 0.58947
    # Dynamic penalty: 0.58947 * 0.85 = 0.501
    assert abs(score - 0.501) < 0.01

def test_fusion_partial_active():
    """Verify renormalization when some signals are missing."""
    score = fuse_confidence_scores(
        cnnspot_conf=1.0,
        trufor_conf=None,
        ela_conf=0.5,
        fft_conf=None,
        noise_conf=None,
        metadata_conf=None
    )
    # Active: cnnspot (0.35), ela (0.10) -> Sum of active weights = 0.45
    # Weighted: 1.0 * 0.35 + 0.5 * 0.10 = 0.40
    # Normalized: 0.40 / 0.45 = 0.8888
    # Dynamic penalty: 0.8888 * 0.85 = 0.7555
    assert abs(score - 0.756) < 0.01

def test_fusion_empty_active():
    """Verify it returns 0.0 when no signals are provided."""
    score = fuse_confidence_scores()
    assert score == 0.0

def test_fusion_single_active():
    """Verify that a single active signal is normalized to its original value times 0.85 penalty."""
    score = fuse_confidence_scores(trufor_conf=0.9)
    # Active: trufor (0.25). Active weight sum = 0.25.
    # Weighted: 0.9 * 0.25 = 0.225.
    # Normalized: 0.225 / 0.25 = 0.9.
    # Dynamic penalty: 0.9 * 0.85 = 0.765.
    assert abs(score - 0.765) < 0.01

def test_fusion_clipping():
    """Verify that inputs outside [0.0, 1.0] are clipped before weighting."""
    score1 = fuse_confidence_scores(trufor_conf=1.5)  # Should clip to 1.0
    score2 = fuse_confidence_scores(trufor_conf=1.0)
    assert score1 == score2

    score3 = fuse_confidence_scores(trufor_conf=-0.5)  # Should clip to 0.0
    assert score3 == 0.0

def test_fusion_penalty_capping():
    """Verify that the final score is capped between 0.0 and 1.0."""
    score = fuse_confidence_scores(trufor_conf=1.0)
    assert 0.0 <= score <= 1.0
