"""
SMARTZI - AI forensic detector
Multi-signal analysis with dynamic renormalization. Unavailable signals return null — never fabricated.
"""

import os
import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from app.core.config import settings

logger = logging.getLogger("smartzi.ai_detector")

TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    import torchvision.models as models
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not available. DL signals disabled.")


def _renormalized_mean(pairs: List[Tuple[Optional[float], float]]) -> Optional[float]:
    active = [(s, w) for s, w in pairs if s is not None]
    if not active:
        return None
    wsum = sum(w for _, w in active)
    return sum(s * w / wsum for s, w in active)


class AIGeneratedDetector:
    def __init__(self):
        self.device = torch.device(settings.DEVICE) if TORCH_AVAILABLE else None
        self.clip_model = None
        self.clip_processor = None
        self.effnet_model = None
        self.mobilenet_model = None
        self._effnet_weights_loaded = False
        self._initialized = False

    def initialize(self) -> bool:
        if self._initialized:
            return True

        if not TORCH_AVAILABLE:
            self._initialized = True
            return True

        try:
            from transformers import CLIPModel, CLIPProcessor
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model.to(self.device)
            self.clip_model.eval()
            logger.info("CLIP loaded.")
        except Exception as e:
            logger.debug("CLIP unavailable: %s", e)

        ai_weights_path = os.path.join(os.path.dirname(__file__), "models", "efficientnet_ai.pth")
        self._effnet_weights_loaded = os.path.exists(ai_weights_path)

        if self._effnet_weights_loaded:
            try:
                weights = models.EfficientNet_B0_Weights.DEFAULT if hasattr(models, "EfficientNet_B0_Weights") else None
                self.effnet_model = models.efficientnet_b0(weights=weights)
                in_features = self.effnet_model.classifier[1].in_features
                self.effnet_model.classifier[1] = nn.Linear(in_features, 2)
                self.effnet_model.load_state_dict(
                    torch.load(ai_weights_path, map_location=self.device)
                )
                self.effnet_model.to(self.device)
                self.effnet_model.eval()
                logger.info("EfficientNet-B0 forensic weights loaded.")
            except Exception as e:
                logger.warning("EfficientNet load failed: %s", e)
                self.effnet_model = None
                self._effnet_weights_loaded = False

        self._initialized = True
        return True

    def preprocess(self, image_path: str) -> Tuple[Image.Image, np.ndarray, np.ndarray]:
        if not os.path.exists(image_path):
            raise ValueError(f"Image not found: {image_path}")
        pil_img = Image.open(image_path).convert("RGB")
        cv_color = cv2.imread(image_path)
        if cv_color is None:
            raise ValueError(f"OpenCV failed to read: {image_path}")
        return pil_img, cv2.cvtColor(cv_color, cv2.COLOR_BGR2GRAY), cv_color

    def predict(self, image_path: str) -> Dict[str, Any]:
        self.initialize()
        pil_img, gray, color = self.preprocess(image_path)
        h, w = gray.shape
        signals: List[Dict[str, Any]] = []

        magnitude_spectrum = None
        dist_from_center = None
        r_low = None

        # FFT
        fft_score: Optional[float] = None
        try:
            f_shift = np.fft.fftshift(np.fft.fft2(gray))
            magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-9)
            cy, cx = h // 2, w // 2
            r_low = min(h, w) // 10
            r_high = min(h, w) // 3
            y_i, x_i = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((y_i - cy) ** 2 + (x_i - cx) ** 2)
            low_m = dist_from_center <= r_low
            high_m = (dist_from_center > r_low) & (dist_from_center <= r_high)
            low_p = float(np.mean(magnitude_spectrum[low_m])) if np.any(low_m) else 1.0
            high_p = float(np.mean(magnitude_spectrum[high_m])) if np.any(high_m) else 0.0
            fft_score = float(np.clip(high_p / (low_p + 1e-6) * 1.5, 0.0, 1.0))
            signals.append({"name": "fft", "score": fft_score})
        except Exception as e:
            logger.error("FFT failed: %s", e)
            signals.append({"name": "fft", "score": None})

        # Checkerboard
        checkerboard_score: Optional[float] = None
        if magnitude_spectrum is not None and dist_from_center is not None and r_low is not None:
            try:
                high_freq = magnitude_spectrum.copy()
                high_freq[dist_from_center <= r_low] = 0
                thr = np.mean(high_freq) + 3.0 * np.std(high_freq)
                peaks = (high_freq > thr) & (high_freq > 0)
                checkerboard_score = float(np.clip(np.sum(peaks) / 120.0, 0.0, 1.0))
                signals.append({"name": "checkerboard", "score": checkerboard_score})
            except Exception as e:
                logger.error("Checkerboard failed: %s", e)
                signals.append({"name": "checkerboard", "score": None})
        else:
            signals.append({"name": "checkerboard", "score": None})

        # Noise
        noise_score: Optional[float] = None
        try:
            residual = cv2.absdiff(gray, cv2.medianBlur(gray, 3))
            bs = 16
            hb, wb = h // bs, w // bs
            vars_ = [
                float(np.var(residual[i * bs:(i + 1) * bs, j * bs:(j + 1) * bs]))
                for i in range(hb) for j in range(wb)
            ]
            if vars_:
                mv, sv = np.mean(vars_), np.std(vars_)
                noise_score = float(np.clip(sv / (mv + 1e-6) * 0.5, 0.0, 1.0))
            signals.append({"name": "noise", "score": noise_score})
        except Exception as e:
            logger.error("Noise failed: %s", e)
            signals.append({"name": "noise", "score": None})

        # JPEG residual
        jpeg_score: Optional[float] = None
        try:
            _, enc = cv2.imencode(".jpg", color, [cv2.IMWRITE_JPEG_QUALITY, 90])
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            jpeg_score = float(np.clip(np.mean(cv2.absdiff(color, dec)) / 20.0, 0.0, 1.0))
            signals.append({"name": "jpeg", "score": jpeg_score})
        except Exception as e:
            logger.error("JPEG residual failed: %s", e)
            signals.append({"name": "jpeg", "score": None})

        # Metadata
        metadata_score: Optional[float] = None
        try:
            from app.services.metadata_service import metadata_extractor
            meta = metadata_extractor.extract(image_path)
            if not meta.has_exif:
                metadata_score = 0.25
            else:
                sw = (meta.software or "").lower()
                ai_kw = ("midjourney", "stable diffusion", "dall-e", "craiyon", "firefly")
                if any(k in sw for k in ai_kw):
                    metadata_score = 0.85
                elif any(k in sw for k in ("photoshop", "gimp", "lightroom")):
                    metadata_score = 0.55
                elif meta.camera_model and not meta.aperture and not meta.shutter_speed:
                    metadata_score = 0.40
                else:
                    metadata_score = 0.15
            signals.append({"name": "metadata", "score": metadata_score})
        except Exception as e:
            logger.error("Metadata signal failed: %s", e)
            signals.append({"name": "metadata", "score": None})

        # CLIP
        clip_score: Optional[float] = None
        if TORCH_AVAILABLE and self.clip_model and self.clip_processor:
            try:
                inputs = self.clip_processor(
                    text=["a real organic photograph", "an AI-generated synthetic image"],
                    images=pil_img,
                    return_tensors="pt",
                    padding=True,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    logits = self.clip_model(**inputs).logits_per_image
                    clip_score = float(torch.softmax(logits, dim=1).cpu().numpy()[0][1])
                signals.append({"name": "clip", "score": clip_score})
            except Exception as e:
                logger.error("CLIP failed: %s", e)
                signals.append({"name": "clip", "score": None})
        else:
            signals.append({"name": "clip", "score": None})

        # EfficientNet (trained weights only)
        efficientnet_score: Optional[float] = None
        if TORCH_AVAILABLE and self.effnet_model is not None and self._effnet_weights_loaded:
            try:
                tfm = T.Compose([
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                t = tfm(pil_img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    efficientnet_score = float(
                        torch.softmax(self.effnet_model(t), dim=1).cpu().numpy()[0][1]
                    )
                signals.append({"name": "efficientnet", "score": efficientnet_score})
            except Exception as e:
                logger.error("EfficientNet failed: %s", e)
                signals.append({"name": "efficientnet", "score": None})
        else:
            signals.append({"name": "efficientnet", "score": None})

        signals.append({"name": "mobilenet", "score": None})

        score_map = {s["name"]: s["score"] for s in signals}

        ai_prob = _renormalized_mean([
            (score_map.get("fft"), 0.15),
            (score_map.get("checkerboard"), 0.20),
            (score_map.get("metadata"), 0.10),
            (score_map.get("clip"), 0.25),
            (score_map.get("efficientnet"), 0.15),
            (score_map.get("mobilenet"), 0.15),
        ])

        tamper_prob = _renormalized_mean([
            (score_map.get("noise"), 0.35),
            (score_map.get("jpeg"), 0.35),
            (score_map.get("metadata"), 0.10),
            (score_map.get("efficientnet"), 0.10),
            (score_map.get("mobilenet"), 0.10),
        ])

        from app.core.confidence_fusion import fuse_confidence_scores
        cnn_conf = score_map.get("efficientnet")
        confidence = fuse_confidence_scores(
            ai_conf=ai_prob,
            cnn_conf=cnn_conf,
            ela_conf=score_map.get("jpeg"),
            metadata_conf=score_map.get("metadata"),
            noise_conf=score_map.get("noise"),
            ai_detector_conf=score_map.get("checkerboard"),
        )

        if ai_prob is not None and ai_prob > 0.58:
            verdict = "AI_GENERATED"
            explanation = f"Forensic signals suggest AI synthesis (p={ai_prob:.2f})."
        elif tamper_prob is not None and tamper_prob > 0.55:
            verdict = "TAMPERED"
            explanation = f"Forensic signals suggest tampering (p={tamper_prob:.2f})."
        elif ai_prob is None and tamper_prob is None:
            verdict = "UNCERTAIN"
            explanation = "Insufficient forensic signals — no verdict fabricated."
            confidence = 0.0
        else:
            verdict = "AUTHENTIC"
            explanation = "Available forensic signals below tamper/AI thresholds."

        return {
            "ai_probability": round(ai_prob, 4) if ai_prob is not None else None,
            "tamper_probability": round(tamper_prob, 4) if tamper_prob is not None else None,
            "confidence": round(confidence, 4),
            "signals": signals,
            "verdict": verdict,
            "ai_generated_probability": round(ai_prob, 4) if ai_prob is not None else None,
            "tampered_probability": round(tamper_prob, 4) if tamper_prob is not None else None,
            "metadata_score": round(metadata_score * 100.0, 2) if metadata_score is not None else None,
            "ela_score": round(jpeg_score * 100.0, 2) if jpeg_score is not None else None,
            "explanation": explanation,
            "noise_inconsistency": score_map.get("noise"),
        }

    def dispose(self):
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.clip_model = None
            self.effnet_model = None
            torch.cuda.empty_cache()


ai_generated_detector = AIGeneratedDetector()
