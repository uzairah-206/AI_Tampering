"""
SMARTZI - AIDE (AI-Generated Image Detector Engine) Service
Detects GAN, Diffusion, frequency anomalies, checkerboard patterns, and noise inconsistencies.
"""

import logging
import os
import cv2
import numpy as np
import threading
from typing import Dict, Any, Optional
from PIL import Image

try:
    import torch
    import torch.nn as nn
    import torchvision.models as models
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger("smartzi.aide")

class AIDEService:
    def __init__(self):
        self._initialized = False
        self._lock = threading.Lock()
        self._available = True
        self.device = torch.device("cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu") if TORCH_AVAILABLE else None
        self.cnn_model = None
        self.cnn_transform = None
        self.weights_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "aide", "GenImage_train.pth")

    def _try_load_model(self, sd: Dict[str, Any]) -> Optional[nn.Module]:
        num_classes = 2
        for k, v in sd.items():
            if ('fc.weight' in k or 'classifier' in k or 'fc.bias' in k) and isinstance(v, torch.Tensor):
                if v.dim() in (1, 2):
                    num_classes = v.size(0)
                    break

        architectures = [
            ("efficientnet_b0", models.efficientnet_b0),
            ("resnet50", models.resnet50),
            ("mobilenet_v3_large", models.mobilenet_v3_large)
        ]

        for name, model_fn in architectures:
            try:
                # Initialize backbone
                model = model_fn(pretrained=True)
                
                # Replace head
                if name == "resnet50":
                    model.fc = nn.Linear(model.fc.in_features, num_classes)
                elif name == "efficientnet_b0":
                    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
                elif name == "mobilenet_v3_large":
                    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

                # Load matching layers
                model_sd = model.state_dict()
                # Clean up DDP or module. prefixes in checkpoint if any
                clean_sd = {k.replace('module.', ''): v for k, v in sd.items()}
                matched_sd = {k: v for k, v in clean_sd.items() if k in model_sd and v.size() == model_sd[k].size()}
                model.load_state_dict(matched_sd, strict=False)

                missing = [k for k in model_sd.keys() if k not in matched_sd]
                logger.info(f"Loaded AIDE checkpoint into {name}. Ignored/missing {len(missing)} keys.")
                return model
            except Exception as e:
                logger.debug(f"Failed to load AIDE into {name}: {e}")

        return None

    def initialize(self) -> bool:
        if self._initialized:
            return self._available
            
        with self._lock:
            if self._initialized:
                return self._available
            
            if TORCH_AVAILABLE and self.cnn_model is None:
                try:
                    if not os.path.exists(self.weights_path):
                        logger.warning(f"AIDE CNN weights missing at {self.weights_path} — falling back to physical signals.")
                    else:
                        sd = torch.load(self.weights_path, map_location=self.device)
                        if isinstance(sd, dict):
                            if "state_dict" in sd:
                                sd = sd["state_dict"]
                            elif "model" in sd:
                                sd = sd["model"]
                        
                        self.cnn_model = self._try_load_model(sd)
                        if self.cnn_model:
                            self.cnn_model.to(self.device)
                            self.cnn_model.eval()
                            self.cnn_transform = transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            ])
                            logger.info("AIDE CNN ACTIVE")
                        else:
                            logger.warning("AIDE physical mode active")
                except Exception as e:
                    logger.error(f"AIDE CNN initialization failed: {e}")
                    logger.warning("AIDE physical mode active")
                    
            else:
                logger.warning("AIDE physical mode active")
                
            self._initialized = True
            logger.info("AIDE service initialized.")
            return self._available

    def dispose(self):
        with self._lock:
            if self.cnn_model is not None:
                del self.cnn_model
                self.cnn_model = None
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._initialized = False
            logger.info("AIDE service disposed.")

    def predict(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        Extracts 5 key physical signals and mathematically fuses them into probabilities.
        """
        if not self.initialize():
            return None
            
        try:
            gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise ValueError("Failed to read image")
                
            h, w = gray.shape
            artifacts = {}
            
            # --- 1. Frequency anomalies & Checkerboard patterns (FFT) ---
            f_shift = np.fft.fftshift(np.fft.fft2(gray))
            magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-9)
            
            cy, cx = h // 2, w // 2
            r_low = min(h, w) // 10
            r_high = min(h, w) // 3
            y_i, x_i = np.ogrid[:h, :w]
            dist = np.sqrt((y_i - cy) ** 2 + (x_i - cx) ** 2)
            
            low_m = dist <= r_low
            high_m = (dist > r_low) & (dist <= r_high)
            
            low_p = float(np.mean(magnitude_spectrum[low_m])) if np.any(low_m) else 1.0
            high_p = float(np.mean(magnitude_spectrum[high_m])) if np.any(high_m) else 0.0
            
            # Frequency Anomalies: Ratio of high-freq to low-freq energy
            freq_anomaly_score = float(np.clip(high_p / (low_p + 1e-6) * 1.5, 0.0, 1.0))
            artifacts["frequency_anomalies"] = round(freq_anomaly_score, 4)
            
            # Checkerboard patterns: Periodic high frequency spikes typical in GAN upsampling
            high_freq = magnitude_spectrum.copy()
            high_freq[dist <= r_low] = 0
            thr = np.mean(high_freq) + 3.0 * np.std(high_freq)
            peaks = (high_freq > thr) & (high_freq > 0)
            checkerboard_score = float(np.clip(np.sum(peaks) / 120.0, 0.0, 1.0))
            artifacts["checkerboard_patterns"] = round(checkerboard_score, 4)
            
            # Diffusion artifacts: Unnatural high-frequency attenuation
            diffusion_score = float(np.clip(1.0 - (high_p / (low_p + 1e-6)), 0.0, 1.0))
            artifacts["diffusion_artifacts"] = round(diffusion_score, 4)
            
            # GAN artifacts: Combined representation of checkerboard + frequency bursts
            gan_score = float(np.clip((freq_anomaly_score + checkerboard_score) / 2.0, 0.0, 1.0))
            artifacts["gan_artifacts"] = round(gan_score, 4)
            
            # --- 2. Local noise inconsistencies ---
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
            else:
                noise_score = 0.0
            artifacts["local_noise_inconsistencies"] = round(noise_score, 4)
            
            # --- Fusion ---
            phys_ai_prob = float(np.clip((gan_score * 0.4) + (diffusion_score * 0.4) + (freq_anomaly_score * 0.2), 0.0, 1.0))
            phys_tamp_prob = noise_score
            
            cnn_score = None
            if self.cnn_model and self.cnn_transform:
                try:
                    pil_img = Image.open(image_path).convert("RGB")
                    tensor = self.cnn_transform(pil_img).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        out = self.cnn_model(tensor)
                        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
                        # Assume class 1 is AI/Tampered
                        cnn_score = float(probs[1]) if len(probs) > 1 else float(torch.sigmoid(out).item())
                        artifacts["cnn_score"] = round(cnn_score, 4)
                except Exception as e:
                    logger.error(f"AIDE CNN inference failed: {e}")
            
            if cnn_score is not None:
                ai_prob = (cnn_score * 0.70) + (phys_ai_prob * 0.30)
                tamp_prob = (cnn_score * 0.70) + (phys_tamp_prob * 0.30)
            else:
                ai_prob = phys_ai_prob
                tamp_prob = phys_tamp_prob
            
            # Overall confidence is the max of specific findings
            confidence = max(ai_prob, tamp_prob)
            
            return {
                "ai_probability": round(ai_prob, 4),
                "tampered_probability": round(tamp_prob, 4),
                "confidence": round(confidence, 4),
                "artifacts": artifacts,
                "source": "aide_detector"
            }
            
        except Exception as e:
            logger.error("AIDE prediction failed: %s", e)
            return None

aide_service = AIDEService()
