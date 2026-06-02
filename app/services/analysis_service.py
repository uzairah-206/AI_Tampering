"""
SMARTZI - Analysis Orchestrator
Coordinates the full AI pipeline safely with isolated sequential memory execution.
"""

import logging
import os
import uuid
import gc
from datetime import datetime, timezone
from typing import List, Optional
from PIL import Image

from app.schemas.analysis import AnalysisResult, ImageMetadata, ELAResult, ClassifierResult, MantraNetResult
from app.services.metadata_service import metadata_extractor
from app.services.ela_service import ela_service
from app.services.trufor_service import trufor_service
from app.services.aide_service import aide_service
from app.services.forensic_stats_service import compute_statistical_signals
from app.core.firebase import firestore_service

logger = logging.getLogger("smartzi.analysis")


class AnalysisOrchestrator:
    ELA_SUSPICIOUS_THRESHOLD = 12.0
    ELA_REGION_THRESHOLD = 3
    TAMPERED_CONFIDENCE_HIGH = 0.75
    MANTRANET_SCORE_THRESHOLD = 0.15

    def _resize_image_if_large(self, image_path: str):
        """Downscale target image to fit into RAM memory boundaries during tensor operations."""
        try:
            with Image.open(image_path) as img:
                MAX_DIM = 720
                if img.size[0] > MAX_DIM or img.size[1] > MAX_DIM:
                    logger.info("Image exceeds safe resolution limits (%s). Compressing...", img.size)
                    img.thumbnail((MAX_DIM, MAX_DIM), Image.Resampling.LANCZOS)
                    # Overwrite file with memory-optimized footprint variation
                    img.save(image_path, quality=90, subsampling=0)
        except Exception as re:
            logger.error("Image pre-downscaling optimization skipped: %s", re)

    async def run(
        self,
        image_path: str,
        upload_id: str,
        user_id: str,
        filename: str,
    ) -> AnalysisResult:
        logger.info("Starting analysis | upload_id=%s filename=%s", upload_id, filename)

        # 1. Optimize input payload size boundaries
        self._resize_image_if_large(image_path)

        import asyncio
        from app.services.model_manager import model_manager

        # 2. RUN LIGHTWEIGHT RECURSIVE HEURISTICS IN PARALLEL (Metadata, ELA, Stats consume negligible RAM)
        logger.debug("Executing lightweight statistical and metadata extractions...")
        heuristic_tasks = [
            asyncio.to_thread(metadata_extractor.extract, image_path),
            asyncio.to_thread(ela_service.analyze, image_path),
            asyncio.to_thread(compute_statistical_signals, image_path),
        ]
        heuristic_results = await asyncio.gather(*heuristic_tasks, return_exceptions=True)

        metadata = heuristic_results[0] if not isinstance(heuristic_results[0], Exception) else ImageMetadata(filename=filename, file_size_kb=0, has_exif=False)
        ela = heuristic_results[1] if not isinstance(heuristic_results[1], Exception) else ELAResult(ela_mean=0, ela_max=0, ela_std=0, suspicious_regions=0)
        stat_signals = heuristic_results[2] if not isinstance(heuristic_results[2], Exception) else {"fft": None, "noise": None, "metadata": None}

        # 3. SEQUENTIAL ISOLATION FOR HEAVY MODELS (Prevents parallel footprint explosion)
        trufor_result = None
        try:
            logger.debug("Ingesting image layer data into isolated TruFor pipeline...")
            trufor_result = await asyncio.to_thread(trufor_service.analyze, image_path)
        except Exception as te:
            logger.error("TruFor execution failed: %s", te)
        finally:
            gc.collect() # Immediately drop structural memory references

        mm_result = None
        try:
            logger.debug("Ingesting image into model_manager fallback route...")
            mm_result = await model_manager.predict(image_path, "image", None)
        except Exception as mme:
            logger.error("ModelManager prediction failure: %s", mme)
        finally:
            gc.collect()

        aide_result = None
        try:
            logger.debug("Ingesting image into isolated AIDE detector pipeline...")
            aide_result = await asyncio.to_thread(aide_service.predict, image_path)
        except Exception as ae:
            logger.error("AIDE detector invocation failure: %s", ae)
        finally:
            gc.collect()

        # Handle fallback instantiation state defaults
        if mm_result is None or isinstance(mm_result, Exception):
            mm_result = {
                "status": "MODEL_DISABLED",
                "prediction": None,
                "confidence": None,
                "explanation": "Model manager pipeline error.",
                "source": "fallback",
                "processing_time": 0.0,
                "heatmap_base64": None,
                "fallback_used": True,
                "fallback_chain": [],
            }

        # 4. PARSE CLASSIFIER DATA STRUCTS
        if trufor_result:
            classifier = ClassifierResult(
                status="ACTIVE",
                prediction=trufor_result.get("prediction"),
                label=trufor_result.get("prediction", "UNCERTAIN"),
                confidence=trufor_result.get("confidence"),
                authentic_probability=(
                    1.0 - trufor_result["confidence"]
                    if trufor_result.get("prediction") == "AUTHENTIC" and trufor_result.get("confidence") is not None
                    else None
                ),
                tampered_probability=(
                    trufor_result.get("confidence")
                    if trufor_result.get("prediction") == "TAMPERED"
                    else None
                ),
                source="trufor",
            )
        elif mm_result.get("confidence") is not None:
            classifier = ClassifierResult(
                status="ACTIVE",
                prediction=mm_result.get("prediction"),
                label=mm_result.get("prediction", "UNCERTAIN"),
                confidence=mm_result.get("confidence"),
                source=mm_result.get("source", "fallback"),
            )
        else:
            classifier = ClassifierResult(
                status="MODEL_DISABLED",
                prediction=None,
                confidence=None,
                label="UNCERTAIN",
                source="fallback",
            )

        # 5. FUSION & VERDICT CONSOLIDATION
        _, _, flags = self._compute_verdict(metadata, ela, classifier, None)
        
        verdict = mm_result.get("prediction") or "UNCERTAIN"
        if verdict == "AUTHENTIC" and trufor_result and trufor_result.get("prediction") == "TAMPERED":
            verdict = "TAMPERED"
            
        ai_gen_prob = aide_result.get("ai_probability") if aide_result else None
        ai_tamp_prob = aide_result.get("tampered_probability") if aide_result else (mm_result.get("confidence") if mm_result.get("prediction") == "TAMPERED" else None)
        
        ai_meta_score = (stat_signals.get("metadata") or 0) * 100.0 if stat_signals.get("metadata") is not None else None
        ai_ela_score = ela.ela_mean

        metadata_summary = {
            "has_exif": metadata.has_exif,
            "camera_model": metadata.camera_model,
            "software": metadata.software,
            "file_size_kb": metadata.file_size_kb,
            "width": metadata.width,
            "height": metadata.height,
        }
        ela_summary = {"ela_mean": ela.ela_mean, "suspicious_regions": ela.suspicious_regions}
        classifier_summary = {
            "status": classifier.status,
            "label": classifier.label,
            "confidence": classifier.confidence,
            "authentic_probability": classifier.authentic_probability,
            "tampered_probability": classifier.tampered_probability,
        }
        ai_detector_summary = {
            "verdict": verdict,
            "statistical_signals": stat_signals,
            "trufor": trufor_result,
            "model_manager_source": mm_result.get("source"),
        }

        # 6. EXTERNAL REMOTE SYSTEM ASSESSMENTS (GEMINI API)
        from app.services.gemini_service import gemini_service
        from app.core.config import settings
        from app.core.confidence_fusion import fuse_confidence_scores as run_confidence_fusion

        gemini_cross = None
        if settings.GEMINI_API_KEY:
            try:
                gemini_cross = await gemini_service.cross_check(
                    image_path=image_path,
                    metadata_summary=metadata_summary,
                    ela_summary=ela_summary,
                    classifier_summary=classifier_summary,
                    ai_detector_summary=ai_detector_summary,
                    mantranet_summary=None,
                )
            except Exception as ge:
                logger.error("Gemini cross-check failed: %s", ge)

        if gemini_cross:
            ai_explanation = gemini_cross.explanation
        else:
            ai_explanation = mm_result.get("explanation", "")

        trufor_val = float(trufor_result["confidence"]) if trufor_result and trufor_result.get("confidence") is not None else None
        ela_val = float(min(ela.ela_mean / 25.0, 1.0))
        metadata_val = stat_signals.get("metadata")
        noise_val = stat_signals.get("noise")
        fft_val = stat_signals.get("fft")
        gemini_val = float(gemini_cross.agreement) if gemini_cross else None

        confidence = run_confidence_fusion(
            trufor_conf=trufor_val,
            mantranet_conf=None,
            ela_conf=ela_val,
            metadata_conf=metadata_val,
            noise_conf=noise_val,
            fft_conf=fft_val,
            gemini_conf=gemini_val,
        )
        
        if gemini_cross:
            adjustment = max(-0.10, min(0.10, gemini_cross.confidence_adjustment))
            confidence = max(0.0, min(1.0, confidence + adjustment))

        risk_score = confidence * 100.0
        summary = f"{ai_explanation}\n\n**Source:** {mm_result.get('source', 'forensic')}"
        if mm_result.get("fallback_used"):
            summary += f" (Fallback chain: {' → '.join(mm_result.get('fallback_chain', []))})"
            
        if flags:
            summary += "\n\n**Flags raised:**\n" + "\n".join([f"- {f}" for f in flags])

        # 7. WRITE OUTPUTS TO FIREBASE RECORD STORES
        scan_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc)

        firestore_payload = {
            "scan_id": scan_id,
            "upload_id": upload_id,
            "user_id": user_id,
            "filename": filename,
            "verdict": verdict,
            "risk_score": risk_score,
            "created_at": created_at,
            "flags": flags,
            "ela_mean": ela.ela_mean,
            "ela_max": ela.ela_max,
            "suspicious_regions": ela.suspicious_regions,
            "classifier_label": classifier.label,
            "classifier_confidence": classifier.confidence,
            "has_exif": metadata.has_exif,
            "camera_model": metadata.camera_model,
            "mantranet_score": None,
            "mantranet_regions": None,
            "mantranet_area_pct": None,
            "ai_generated_probability": ai_gen_prob,
            "tampered_probability": ai_tamp_prob,
            "metadata_score": ai_meta_score,
            "ela_score": ai_ela_score,
            "explanation": ai_explanation
        }
        await firestore_service.create_scan(firestore_payload)
        logger.info("Persisted scan | scan_id=%s verdict=%s risk=%.1f", scan_id, verdict, risk_score)

        return AnalysisResult(
            scan_id=scan_id,
            upload_id=upload_id,
            user_id=user_id,
            filename=filename,
            created_at=created_at,
            metadata=metadata,
            ela=ela,
            classifier=classifier,
            mantranet=None,
            verdict=verdict,
            risk_score=round(risk_score, 1),
            summary=summary,
            flags=flags,
            confidence=confidence,
            ai_generated_probability=ai_gen_prob,
            tampered_probability=ai_tamp_prob,
            metadata_score=ai_meta_score,
            ela_score=ai_ela_score,
            explanation=ai_explanation
        )

    def _compute_verdict(
        self,
        metadata: ImageMetadata,
        ela: ELAResult,
        classifier: ClassifierResult,
        mantranet: Optional[MantraNetResult],
    ):
        flags: List[str] = []

        ela_score = min(ela.ela_mean / 25.0, 1.0) * 100
        if ela.ela_mean > self.ELA_SUSPICIOUS_THRESHOLD:
            flags.append(f"High ELA mean ({ela.ela_mean:.1f}) — possible compression artifacts")
        if ela.suspicious_regions > self.ELA_REGION_THRESHOLD:
            flags.append(f"{ela.suspicious_regions} suspicious high-ELA regions detected")

        clf_score = (classifier.tampered_probability or 0.0) * 100
        if classifier.label == "TAMPERED" and classifier.confidence is not None and classifier.confidence > self.TAMPERED_CONFIDENCE_HIGH:
            flags.append(f"CNN classifier: TAMPERED (confidence {classifier.confidence*100:.0f}%)")

        meta_score = self._metadata_anomaly_score(metadata, flags)
        risk_score = (ela_score * 0.40) + (clf_score * 0.45) + (meta_score * 0.15)

        if risk_score >= 60:
            verdict = "TAMPERED"
        elif risk_score <= 35:
            verdict = "AUTHENTIC"
        else:
            verdict = "UNCERTAIN"

        return verdict, risk_score, flags

    def _metadata_anomaly_score(self, metadata: ImageMetadata, flags: List[str]) -> float:
        score = 0.0
        if not metadata.has_exif:
            score += 30
            flags.append("EXIF data missing — may have been stripped")
        if metadata.software and any(app in metadata.software.lower() for app in ["photoshop", "gimp", "canva", "midjourney"]):
            score += 40
            flags.append(f"Editing software signature found: {metadata.software}")
        return min(score, 100.0)