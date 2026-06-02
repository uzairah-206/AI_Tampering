"""
SMARTZI — Hybrid AI Image Detection Pipeline
=============================================

Production-ready, zero-training pipeline that runs five sequential stages:
    1. Validator (strict image input checking)
    2. Gemini (initial visual AI-generation analysis)
    3. AIDE (global feature binary classifier replacing CNNSpot)
    4. TruFor (local noise anomaly detector)
    5. Forensics (traditional signals: ELA, FFT, Noise, Metadata)
    6. Fusion (dynamic renormalization weighting)

Each step is cleanly wrapped in independent try/except blocks to ensure 
zero catastrophic crashes. The output maps rigidly to the unified response schema.
"""

from __future__ import annotations

import time
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from app.validators.image_validator import validate_input_image, ImageValidationError
from app.core.config import settings

# Global structural dependencies for ML pipeline components
from app.services.trufor_service import trufor_service
from app.services.aide_service import aide_service
from app.services.forensic_signals import compute_forensic_signals
from app.services.confidence_fusion import fuse_confidence_scores
from app.services.gemini_service import run_gemini_initial_analysis

logger = logging.getLogger("smartzi.hybrid_pipeline")

@dataclass(frozen=True)
class PipelineConfig:
    """Centralised pipeline knobs."""
    gemini_api_keys: str = os.environ.get("GEMINI_API_KEYS", settings.GEMINI_API_KEY)
    gemini_model: str = os.environ.get("GEMINI_MODEL", settings.GEMINI_MODEL)
    trufor_timeout: int = 120
    cnnspot_timeout: int = 30  
    gemini_timeout: int = 45

def execute_with_timeout(component_name: str, timeout_sec: int, func, *args, **kwargs) -> Any:
    """Runs a blocking synchronous ML function safely with a strict timeout limitation."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=float(timeout_sec))
        except TimeoutError:
            logger.error(f"{component_name} execution exceeded boundary allocation of {timeout_sec} seconds.")
            return {"available": False, "reason": "timeout"}
        except Exception as exc:
            logger.error(f"{component_name} execution failed: {exc}")
            return {"available": False, "reason": str(exc)}

def _run_trufor(image_path: str, timeout: int) -> Dict[str, Any]:
    res = execute_with_timeout("TruFor", timeout, trufor_service.analyze, image_path)
    if res is None:  
        return {"available": False, "reason": "unknown"}
    return res

def _run_forensics(image_path: str) -> Dict[str, float]:
    try:
        return compute_forensic_signals(image_path)
    except Exception as exc:
        logger.error("Forensic signals failed: %s", exc)
        return {"ela_score": 0.0, "fft_score": 0.0, "noise_score": 0.0, "metadata_score": 0.0}

def _run_fusion(aide_res: Dict[str, Any], trufor_res: Dict[str, Any], forensic_res: Dict[str, float], gemini_conf: Optional[float]) -> float:
    try:
        aide_available = False
        aide_conf = None
        
        if aide_res and "confidence" in aide_res:
            aide_conf = aide_res.get("confidence")
            aide_available = aide_res.get("available", True)

        tru_available = trufor_res.get("available", False) if trufor_res else False
        tru_conf = trufor_res.get("tampered_probability") if tru_available else None

        logger.info("Fusion processing. AIDE: conf=%s, Gemini: conf=%s", aide_conf, gemini_conf)

        return fuse_confidence_scores(
            cnnspot_conf=aide_conf if aide_available else None,
            trufor_conf=tru_conf,
            ela_conf=forensic_res.get("ela_score") if forensic_res else None,
            fft_conf=forensic_res.get("fft_score") if forensic_res else None,
            noise_conf=forensic_res.get("noise_score") if forensic_res else None,
            metadata_conf=forensic_res.get("metadata_score") if forensic_res else None,
            gemini_conf=gemini_conf
        )
    except Exception as exc:
        logger.error("Fusion execution error: %s", exc)
        return 0.50

def run_detection_pipeline(image_path: str, *, config: Optional[PipelineConfig] = None) -> Dict[str, Any]:
    cfg = config or PipelineConfig()
    t_start = time.monotonic()

    final_payload: Dict[str, Any] = {
        "prediction": "UNCERTAIN",
        "confidence": 0.0,
        "ai_probability": 0.0,
        "tampered_probability": 0.0,
        "ela_score": 0.0,
        "fft_score": 0.0,
        "noise_score": 0.0,
        "metadata_score": 0.0,
        "gemini_explanation": "",
        "heatmap_base64": "",
        "processing_time": 0.0,
        "is_gemini_dominant": False,
    }

    # 1. Validation Stage
    try:
        _, _ = validate_input_image(image_path)
    except Exception as exc:
        logger.error("Pipeline aborted — validation failed: %s", exc)
        final_payload["processing_time"] = round(time.monotonic() - t_start, 3)
        final_payload["gemini_explanation"] = f"Validation Error: {exc}"
        return final_payload

    # 2. Gemini Initial Visual Analysis (Rotates Keys Automatically)
    api_key_list = [k.strip() for k in cfg.gemini_api_keys.split(",") if k.strip()]
    gemini_res = run_gemini_initial_analysis(image_path, api_key_list, cfg.gemini_model)

    if gemini_res.get("available"):
        # ── GEMINI-DOMINANT PATH ─────────────────────────────────────────
        # Gemini succeeded — use its verdict as the authoritative result.
        # Skip all heavy local models to avoid contradictory signals.
        gemini_prob = gemini_res.get("ai_probability", 0.0)
        final_payload["is_gemini_dominant"] = True
        final_payload["ai_probability"] = gemini_prob
        final_payload["confidence"] = gemini_prob
        final_payload["gemini_explanation"] = gemini_res.get("explanation", "No explanation provided.")

        if gemini_prob > 0.65:
            final_payload["prediction"] = "AI_GENERATED"
        elif gemini_prob < 0.35:
            final_payload["prediction"] = "AUTHENTIC"
        else:
            final_payload["prediction"] = "UNCERTAIN"

        logger.info("Gemini-dominant path: prediction=%s confidence=%.4f",
                     final_payload["prediction"], gemini_prob)
        final_payload["processing_time"] = round(time.monotonic() - t_start, 3)
        return final_payload

    # ── FALLBACK PATH — Gemini unavailable ────────────────────────────────
    logger.warning("Gemini unavailable. Falling back to local model pipeline.")
    final_payload["gemini_explanation"] = "Gemini service unavailable. Analysis performed using local forensic models."

    # 3. AIDE Service
    aide_res = execute_with_timeout("AIDE", cfg.cnnspot_timeout, aide_service.predict, image_path)
    if aide_res and aide_res.get("available", True) and "confidence" in aide_res:
        final_payload["ai_probability"] = aide_res.get("confidence", 0.0)

    # 4. TruFor Local Feature Isolation Engine
    trufor_res = _run_trufor(image_path, cfg.trufor_timeout)
    if trufor_res.get("available"):
        final_payload["tampered_probability"] = trufor_res.get("tampered_probability", 0.0)
        final_payload["heatmap_base64"] = trufor_res.get("heatmap_base64", "")

    # 5. Traditional Quantization Signal Processing
    forensic_res = _run_forensics(image_path)
    final_payload["ela_score"] = forensic_res.get("ela_score", 0.0)
    final_payload["fft_score"] = forensic_res.get("fft_score", 0.0)
    final_payload["noise_score"] = forensic_res.get("noise_score", 0.0)
    final_payload["metadata_score"] = forensic_res.get("metadata_score", 0.0)

    # 6. Mathematical Layer Fusion Matrix Calculation
    fused_conf = _run_fusion(aide_res, trufor_res, forensic_res, None)
    final_payload["confidence"] = round(fused_conf, 4)
    
    aide_active = aide_res and aide_res.get("available", True) and "confidence" in aide_res
    if not aide_active and not trufor_res.get("available"):
        if fused_conf > 0.40:
            final_payload["prediction"] = "AI_GENERATED" if final_payload["metadata_score"] > 0.5 else "TAMPERED"
        else:
            final_payload["prediction"] = "AUTHENTIC"
    else:
        if fused_conf > 0.55:
            if final_payload["ai_probability"] >= final_payload["tampered_probability"]:
                final_payload["prediction"] = "AI_GENERATED"
            else:
                final_payload["prediction"] = "TAMPERED"
        elif fused_conf < 0.45:
            final_payload["prediction"] = "AUTHENTIC"
        else:
            final_payload["prediction"] = "UNCERTAIN"

    final_payload["processing_time"] = round(time.monotonic() - t_start, 3)
    return final_payload

if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Usage: python -m app.services.hybrid_pipeline <image_path>")
        sys.exit(1)

    result = run_detection_pipeline(sys.argv[1])
    
    def _redact(obj):
        if isinstance(obj, dict):
            return {k: ("<base64>" if k == "heatmap_base64" and v else _redact(v)) for k, v in obj.items()}
        return obj

    print(json.dumps(_redact(result), indent=2))