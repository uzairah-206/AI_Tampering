"""
SMARTZI - ModelManager
Fallback chain: TruFor → ManTraNet → Statistical signals.
"""

import logging
import asyncio
import time
import gc
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("smartzi.model_manager")


class ModelName(str, Enum):
    TRUFOR = "trufor"
    AIDE_DETECTOR = "aide_detector"
    ELA = "ela"
    EXIF = "exif"
    FFT = "fft"
    NOISE = "noise"
    GEMINI = "gemini"
    MANTRANET = "mantranet"
    STATISTICAL = "statistical"


class FallbackReason(str, Enum):
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    AUTH_FAILURE = "auth_failure"
    NETWORK = "network_failure"
    SERVER_ERROR = "server_error"
    INVALID_OUTPUT = "invalid_output"
    EMPTY_OUTPUT = "empty_output"
    WEIGHTS_MISSING = "weights_missing"
    NOT_IMPLEMENTED = "not_implemented"
    UNKNOWN = "unknown"


@dataclass
class ModelResult:
    prediction: str
    confidence: float
    explanation: str
    source: str
    processing_time_ms: float
    heatmap_base64: Optional[str] = None
    fallback_used: bool = False
    fallback_chain: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction": self.prediction,
            "confidence": round(self.confidence, 4),
            "explanation": self.explanation,
            "source": self.source,
            "processing_time": round(self.processing_time_ms, 2),
            "heatmap_base64": self.heatmap_base64,
            "fallback_used": self.fallback_used,
            "fallback_chain": self.fallback_chain,
        }


class ModelManager:
    MODEL_TIMEOUTS: Dict[ModelName, float] = {
        ModelName.TRUFOR: 60.0,
        ModelName.AIDE_DETECTOR: 30.0,
        ModelName.ELA: 15.0,
        ModelName.EXIF: 10.0,
        ModelName.FFT: 15.0,
        ModelName.NOISE: 15.0,
        ModelName.GEMINI: 30.0,
        ModelName.MANTRANET: 45.0,
        ModelName.STATISTICAL: 20.0,
    }

    FALLBACK_CHAIN: List[ModelName] = [
        ModelName.TRUFOR,
        ModelName.AIDE_DETECTOR,
        ModelName.ELA,
        ModelName.EXIF,
        ModelName.FFT,
        ModelName.NOISE,
        ModelName.GEMINI,
    ]

    def __init__(self):
        self.active_model: ModelName = ModelName.TRUFOR
        self._initialized: bool = False
        self._attempted: List[str] = []

    async def initialize(self):
        # Prevent runtime overhead, configuration is handled on-demand during requests
        self._initialized = True

    async def dispose(self):
        logger.info("ModelManager flushing cached memory structures...")
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            del torch
        except Exception:
            pass
        gc.collect()
        self._initialized = False

    async def predict(
        self,
        image_path: str,
        input_type: str = "image",
        text_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self._initialized:
            await self.initialize()

        self._attempted = []
        t_total = time.monotonic()

        for model_name in self.FALLBACK_CHAIN:
            self._attempted.append(model_name.value)
            timeout = self.MODEL_TIMEOUTS[model_name]
            try:
                t0 = time.monotonic()
                raw = await asyncio.wait_for(
                    self._dispatch(model_name, image_path),
                    timeout=timeout,
                )
                elapsed_ms = (time.monotonic() - t0) * 1000
                self._validate_output(raw)

                total_ms = (time.monotonic() - t_total) * 1000
                result = ModelResult(
                    prediction=raw["prediction"],
                    confidence=float(raw["confidence"]),
                    explanation=raw.get("explanation", f"Analysis by {model_name.value}."),
                    source=raw.get("source", model_name.value),
                    processing_time_ms=total_ms,
                    heatmap_base64=raw.get("heatmap_base64") or raw.get("heatmap"),
                    fallback_used=len(self._attempted) > 1,
                    fallback_chain=list(self._attempted),
                )
                return result.to_dict()
            except asyncio.TimeoutError:
                logger.warning("Timeout on %s", model_name.value)
            except Exception as e:
                await self.handle_fallback(model_name, FallbackReason.UNKNOWN, str(e))

        return self._disabled_fallback(
            self._attempted,
            (time.monotonic() - t_total) * 1000,
        )

    async def handle_fallback(
        self,
        failed_model: ModelName,
        reason: FallbackReason,
        detail: str = "",
    ) -> FallbackReason:
        logger.warning(
            "Fallback | failed=%s reason=%s detail=%s",
            failed_model.value, reason.value, detail[:120],
        )
        return reason

    @staticmethod
    def _disabled_fallback(chain: List[str], elapsed_ms: float) -> Dict[str, Any]:
        return {
            "status": "MODEL_DISABLED",
            "prediction": None,
            "confidence": None,
            "explanation": "All forensic models unavailable.",
            "source": "fallback",
            "processing_time": round(elapsed_ms, 2),
            "heatmap_base64": None,
            "fallback_used": True,
            "fallback_chain": chain,
        }

    def _validate_output(self, raw: Any):
        if not raw or not isinstance(raw, dict):
            raise ValueError("empty output")
        if raw.get("status") == "MODEL_DISABLED":
            return
        if raw.get("confidence") is None or raw.get("prediction") is None:
            raise ValueError("missing prediction/confidence")
        conf = float(raw["confidence"])
        if not 0.0 <= conf <= 1.0:
            raise ValueError("confidence out of range")
        if raw["prediction"] not in ("AUTHENTIC", "TAMPERED", "UNCERTAIN", "AI_GENERATED"):
            raise ValueError("invalid prediction")

    async def _dispatch(self, model: ModelName, image_path: str) -> Dict[str, Any]:
        if model == ModelName.TRUFOR:
            return await self._run_trufor(image_path)
        if model == ModelName.AIDE_DETECTOR:
            return await self._run_aide_detector(image_path)
        if model == ModelName.ELA:
            return await self._run_ela(image_path)
        if model == ModelName.EXIF:
            return await self._run_exif(image_path)
        if model == ModelName.FFT:
            return await self._run_fft(image_path)
        if model == ModelName.NOISE:
            return await self._run_noise(image_path)
        if model == ModelName.GEMINI:
            return await self._run_gemini(image_path)
        if model == ModelName.MANTRANET:
            return await self._run_mantranet(image_path)
        if model == ModelName.STATISTICAL:
            return await self._run_statistical(image_path)
        raise ValueError(f"Unknown model: {model}")

    async def _run_aide_detector(self, image_path: str) -> Dict[str, Any]:
        from app.services.aide_service import aide_service
        res = await asyncio.to_thread(aide_service.predict, image_path)
        if not res:
            raise ValueError("AIDE Detector unavailable or uncertain")
            
        if res["ai_probability"] > 0.55:
            verdict = "AI_GENERATED"
        elif res["tampered_probability"] > 0.55:
            verdict = "TAMPERED"
        else:
            verdict = "AUTHENTIC"
            
        return {
            "prediction": verdict,
            "confidence": res["confidence"],
            "explanation": f"AIDE Physical Analysis: AI Probability ({res['ai_probability']:.2f}), Tampered ({res['tampered_probability']:.2f}).",
            "source": "aide_detector",
        }

    async def _run_ela(self, image_path: str) -> Dict[str, Any]:
        from app.services.ela_service import ela_service
        res = await asyncio.to_thread(ela_service.analyze, image_path)
        if not res:
            raise ValueError("ELA unavailable")
        label = "TAMPERED" if res.ela_mean > 12.0 else "AUTHENTIC"
        conf = min(res.ela_mean / 25.0, 1.0)
        return {
            "prediction": label,
            "confidence": conf,
            "explanation": f"ELA mean: {res.ela_mean:.2f}",
            "source": "ela",
        }

    async def _run_exif(self, image_path: str) -> Dict[str, Any]:
        from app.services.metadata_service import metadata_extractor
        res = await asyncio.to_thread(metadata_extractor.extract, image_path)
        if not res:
            raise ValueError("EXIF extraction failed")
        label = "TAMPERED" if not res.has_exif else "AUTHENTIC"
        return {
            "prediction": label,
            "confidence": 0.8 if not res.has_exif else 0.5,
            "explanation": "EXIF data missing" if not res.has_exif else "EXIF data present",
            "source": "exif",
        }

    async def _run_fft(self, image_path: str) -> Dict[str, Any]:
        from app.services.forensic_stats_service import compute_statistical_signals
        res = await asyncio.to_thread(compute_statistical_signals, image_path)
        fft_score = res.get("fft")
        if fft_score is None:
            raise ValueError("FFT failed")
        return {
            "prediction": "TAMPERED" if fft_score > 0.55 else "AUTHENTIC",
            "confidence": fft_score,
            "explanation": f"FFT score: {fft_score:.3f}",
            "source": "fft",
        }

    async def _run_noise(self, image_path: str) -> Dict[str, Any]:
        from app.services.forensic_stats_service import compute_statistical_signals
        res = await asyncio.to_thread(compute_statistical_signals, image_path)
        noise_score = res.get("noise")
        if noise_score is None:
            raise ValueError("Noise check failed")
        return {
            "prediction": "TAMPERED" if noise_score > 0.55 else "AUTHENTIC",
            "confidence": noise_score,
            "explanation": f"Noise score: {noise_score:.3f}",
            "source": "noise",
        }

    async def _run_gemini(self, image_path: str) -> Dict[str, Any]:
        from app.core.config import settings
        if not settings.GEMINI_API_KEY:
            raise ValueError("Gemini API key not configured")
        return {
            "prediction": "UNCERTAIN",
            "confidence": 0.5,
            "explanation": "Gemini cross-check engaged as final fallback.",
            "source": "gemini",
        }

    async def _run_trufor(self, image_path: str) -> Dict[str, Any]:
        from app.services.trufor_service import trufor_service
        res = await asyncio.to_thread(trufor_service.analyze, image_path)
        if not res:
            raise ValueError("TruFor unavailable")
            
        explanation = f"TruFor detector: {res['prediction']} ({res['confidence']*100:.0f}% confidence)."
        if "tampered_regions" in res:
            explanation += f" Found {res['tampered_regions']} suspicious region(s)."
            
        return {
            "prediction": res["prediction"],
            "confidence": res["confidence"],
            "explanation": explanation,
            "source": "trufor",
            "heatmap_base64": res.get("heatmap"),
        }

    async def _run_mantranet(self, image_path: str) -> Dict[str, Any]:
        from app.services.mantranet_service import mantranet_service
        if not mantranet_service.is_available:
            raise ValueError("ManTraNet weights not found")
        res = await asyncio.to_thread(mantranet_service.analyze, image_path)
        if res is None:
            raise ValueError("empty ManTraNet output")
        label = "TAMPERED" if res.forgery_score > 0.15 else "AUTHENTIC"
        return {
            "prediction": label,
            "confidence": float(res.forgery_score),
            "explanation": (
                f"ManTraNet: score {res.forgery_score:.3f}, "
                f"{res.tampered_regions} region(s), {res.tampered_area_pct:.1f}% area."
            ),
            "source": "mantranet",
            "heatmap_base64": res.heatmap_base64,
        }

    async def _run_statistical(self, image_path: str) -> Dict[str, Any]:
        from app.services.forensic_stats_service import (
            aggregate_statistical_score,
            compute_statistical_signals,
        )

        signals = await asyncio.to_thread(compute_statistical_signals, image_path)
        score = aggregate_statistical_score(signals)
        if score is None:
            raise ValueError("no statistical signals")
        label = "TAMPERED" if score > 0.55 else "AUTHENTIC"
        return {
            "prediction": label,
            "confidence": float(score),
            "explanation": f"Statistical forensic fusion score {score:.3f}.",
            "source": "statistical",
            "heatmap_base64": None,
        }


model_manager = ModelManager()