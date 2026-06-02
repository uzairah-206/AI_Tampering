"""
SMARTZI - Analysis Orchestrator
Coordinates the full AI pipeline: metadata → ELA → classifier → ManTraNet → verdict.
"""

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from app.schemas.analysis import AnalysisResult, ImageMetadata, ELAResult, ClassifierResult, MantraNetResult
from app.services.metadata_service import metadata_extractor
from app.services.ela_service import ela_service
from app.services.mantranet_service import mantranet_service
from app.services.trufor_service import trufor_service
from app.services.aide_service import aide_service
from app.services.forensic_stats_service import compute_statistical_signals
from app.core.firebase import firestore_service

logger = logging.getLogger("smartzi.analysis")


class AnalysisOrchestrator:
    """
    Combines all AI signals and applies a deterministic fusion rule to
    produce a final verdict and risk score.

    Fusion weights (when ManTraNet is available):
        - ELA signal:        25%
        - CNN classifier:    30%
        - ManTraNet:         35%
        - Metadata anomaly:  10%

    Fusion weights (fallback without ManTraNet):
        - ELA signal:        40%
        - CNN classifier:    45%
        - Metadata anomaly:  15%
    """

    # Thresholds tuned empirically
    ELA_SUSPICIOUS_THRESHOLD = 12.0   # mean ELA intensity above this → suspicious
    ELA_REGION_THRESHOLD = 3          # suspicious region count above this → flagged
    TAMPERED_CONFIDENCE_HIGH = 0.75   # classifier is "very confident" of tampering
    MANTRANET_SCORE_THRESHOLD = 0.15  # ManTraNet mean score above this → suspicious

    async def run(
        self,
        image_path: str,
        upload_id: str,
        user_id: str,
        filename: str,
    ) -> AnalysisResult:
        """
        Execute the full analysis pipeline and persist results to Firestore.
        """
        logger.info("Starting analysis | upload_id=%s filename=%s", upload_id, filename)

        import asyncio

        from app.services.model_manager import model_manager

        async def run_mantranet():
            # Deprecated: ManTraNet removed from active chain
            return None

        tasks = [
            asyncio.to_thread(metadata_extractor.extract, image_path),
            asyncio.to_thread(ela_service.analyze, image_path),
            asyncio.to_thread(compute_statistical_signals, image_path),
            asyncio.to_thread(trufor_service.analyze, image_path),
            run_mantranet(),
            model_manager.predict(image_path, "image", None),
            asyncio.to_thread(aide_service.predict, image_path),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions gracefully to prevent full request failure
        metadata = results[0] if not isinstance(results[0], Exception) else ImageMetadata(filename=filename, file_size_kb=0, has_exif=False)
        ela = results[1] if not isinstance(results[1], Exception) else ELAResult(ela_mean=0, ela_max=0, ela_std=0, suspicious_regions=0)
        stat_signals = results[2] if not isinstance(results[2], Exception) else {
            "fft": None, "noise": None, "metadata": None,
        }
        trufor_result = results[3] if not isinstance(results[3], Exception) else None
        mantranet_result = results[4] if not isinstance(results[4], Exception) else None
        mm_result = results[5]
        aide_result = results[6] if not isinstance(results[6], Exception) else None
        if isinstance(mm_result, Exception):
            logger.error("ModelManager failed entirely: %s", mm_result)
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

        logger.debug("Metadata extracted | has_exif=%s", metadata.has_exif)
        logger.debug("ELA complete | mean=%.2f regions=%d", ela.ela_mean, ela.suspicious_regions)
        logger.debug(
            "TruFor | %s",
            f"{trufor_result.get('prediction')} ({trufor_result.get('confidence')})" if trufor_result else "N/A",
        )

        if mantranet_result:
            logger.debug(
                "ManTraNet | score=%.4f regions=%d area=%.1f%%",
                mantranet_result.forgery_score,
                mantranet_result.tampered_regions,
                mantranet_result.tampered_area_pct,
            )
        else:
            logger.info("ManTraNet disabled — using fallback fusion")

        # ── Stage 5: Fusion & Verdict ─────────────────────────────────────
        # Retain local heuristic flags, but use ModelManager/AI Detector for primary verdict
        _, _, flags = self._compute_verdict(
            metadata, ela, classifier, mantranet_result
        )
        
        # Merge results: use AIGeneratedDetector's fine-grained forensic classifications
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
        mantranet_summary = (
            {
                "forgery_score": mantranet_result.forgery_score,
                "tampered_regions": mantranet_result.tampered_regions,
                "tampered_area_pct": mantranet_result.tampered_area_pct,
            }
            if mantranet_result
            else None
        )

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
                    mantranet_summary=mantranet_summary,
                )
            except Exception as ge:
                logger.error("Gemini cross-check failed: %s", ge)

        if gemini_cross:
            ai_explanation = gemini_cross.explanation
        else:
            ai_explanation = mm_result.get("explanation", "")

        trufor_val = float(trufor_result["confidence"]) if trufor_result and trufor_result.get("confidence") is not None else None
        mantra_val = float(mantranet_result.forgery_score) if mantranet_result else None
        ela_val = float(min(ela.ela_mean / 25.0, 1.0))
        metadata_val = stat_signals.get("metadata")
        noise_val = stat_signals.get("noise")
        fft_val = stat_signals.get("fft")
        gemini_val = float(gemini_cross.agreement) if gemini_cross else None

        confidence = run_confidence_fusion(
            trufor_conf=trufor_val,
            mantranet_conf=mantra_val,
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

        # ── Stage 6: Persist to Firestore ────────────────────────────────────
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
            # ManTraNet fields
            "mantranet_score": mantranet_result.forgery_score if mantranet_result else None,
            "mantranet_regions": mantranet_result.tampered_regions if mantranet_result else None,
            "mantranet_area_pct": mantranet_result.tampered_area_pct if mantranet_result else None,
            # AI detector fields
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
            mantranet=mantranet_result,
            verdict=verdict,
            risk_score=round(risk_score, 1),
            summary=summary,
            flags=flags,
            
            # AI-Generated Detector Response Schema Fields
            confidence=confidence,
            ai_generated_probability=ai_gen_prob,
            tampered_probability=ai_tamp_prob,
            metadata_score=ai_meta_score,
            ela_score=ai_ela_score,
            explanation=ai_explanation
        )

    # ── Fusion Logic ──────────────────────────────────────────────────────────
    def _compute_verdict(
        self,
        metadata: ImageMetadata,
        ela: ELAResult,
        classifier: ClassifierResult,
        mantranet: Optional[MantraNetResult],
    ):
        """
        Weighted fusion of signals into a final verdict + risk score.
        Uses 4-signal fusion when ManTraNet is available, 3-signal fallback otherwise.
        Returns (verdict, risk_score_0_to_100, flags)
        """
        flags: List[str] = []

        # --- ELA signal (0–100) ---
        ela_score = min(ela.ela_mean / 25.0, 1.0) * 100
        if ela.ela_mean > self.ELA_SUSPICIOUS_THRESHOLD:
            flags.append(f"High ELA mean ({ela.ela_mean:.1f}) — possible compression artifacts")
        if ela.suspicious_regions > self.ELA_REGION_THRESHOLD:
            flags.append(f"{ela.suspicious_regions} suspicious high-ELA regions detected")

        # --- Classifier signal (0–100) ---
        clf_score = (classifier.tampered_probability or 0.0) * 100
        if classifier.label == "TAMPERED" and classifier.confidence is not None and classifier.confidence > self.TAMPERED_CONFIDENCE_HIGH:
            flags.append(f"CNN classifier: TAMPERED (confidence {classifier.confidence*100:.0f}%)")

        # --- Metadata signal (0–100) ---
        meta_score = self._metadata_anomaly_score(metadata, flags)

        # --- ManTraNet signal (0–100) ---
        if mantranet is not None:
            mantra_score = min(mantranet.forgery_score / 0.5, 1.0) * 100  # normalise (0.5 is very high)

            if mantranet.forgery_score > self.MANTRANET_SCORE_THRESHOLD:
                flags.append(
                    f"ManTraNet: forgery detected (score {mantranet.forgery_score:.3f}, "
                    f"{mantranet.tampered_regions} region(s), {mantranet.tampered_area_pct:.1f}% area)"
                )
            if mantranet.tampered_area_pct > 5.0:
                flags.append(f"ManTraNet: {mantranet.tampered_area_pct:.1f}% of image area shows tampering signs")

            # ── 4-signal weighted fusion ──
            risk_score = (
                ela_score * 0.25
                + clf_score * 0.30
                + mantra_score * 0.35
                + meta_score * 0.10
            )
        else:
            # ── 3-signal fallback (original weights) ──
            risk_score = (ela_score * 0.40) + (clf_score * 0.45) + (meta_score * 0.15)

        # --- Verdict thresholds ---
        if risk_score >= 60:
            verdict = "TAMPERED"
        elif risk_score <= 35:
            verdict = "AUTHENTIC"
        else:
            verdict = "UNCERTAIN"

        return verdict, risk_score, flags

    def _metadata_anomaly_score(self, metadata: ImageMetadata, flags: List[str]) -> float:
        """
        Heuristic anomaly scoring based on metadata signals.
        Returns a 0–100 score where higher = more suspicious.
        """
        score = 0.0

        if not metadata.has_exif:
            score += 30  # EXIF stripped — common tampering step
            flags.append("EXIF data missing — may have been stripped")

        if metadata.software and any(
            kw in metadata.software.lower()
            for kw in ["photoshop", "gimp", "lightroom", "affinity", "pixelmator"]
        ):
            score += 50
            flags.append(f"Editing software detected in EXIF: {metadata.software}")

        # Mismatch between camera model presence and editing software
        if metadata.camera_model and metadata.software:
            score += 10
            flags.append("Both camera EXIF and editing software signature present")

        return min(score, 100)

    # ── Summary Generation ────────────────────────────────────────────────────
    def _generate_summary(
        self,
        verdict: str,
        risk_score: float,
        flags: List[str],
        metadata: ImageMetadata,
        ela: ELAResult,
        classifier: ClassifierResult,
        mantranet: Optional[MantraNetResult],
    ) -> str:
        """Generate a human-readable plain-English summary for the chat UI."""
        verdict_lines = {
            "AUTHENTIC": "✅ This image appears **authentic**.",
            "TAMPERED": "🚨 This image shows **strong signs of tampering**.",
            "UNCERTAIN": "⚠️ This image shows **some anomalies** — results are inconclusive.",
        }
        lines = [
            verdict_lines[verdict],
            f"Overall risk score: **{risk_score:.0f}/100**.",
            "",
            f"**ELA Analysis:** Mean intensity {ela.ela_mean:.1f}, {ela.suspicious_regions} suspicious region(s).",
            f"**CNN Classifier:** {classifier.label} ({f'{classifier.confidence*100:.0f}%' if classifier.confidence is not None else 'N/A'} confidence).",
        ]

        # ManTraNet section
        if mantranet is not None:
            lines.append(
                f"**ManTraNet Forensics:** Forgery score {mantranet.forgery_score:.3f}, "
                f"{mantranet.tampered_regions} tampered region(s), "
                f"{mantranet.tampered_area_pct:.1f}% area affected."
            )

        lines.append(
            f"**Metadata:** {'Present' if metadata.has_exif else 'Missing/stripped'}"
            + (f", shot with {metadata.camera_model}" if metadata.camera_model else "")
            + "."
        )

        if flags:
            lines += ["", "**Flags raised:**"] + [f"- {f}" for f in flags]

        return "\n".join(lines)


# Module-level singleton
analysis_orchestrator = AnalysisOrchestrator()
