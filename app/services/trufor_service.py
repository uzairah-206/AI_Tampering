"""
SMARTZI - TruFor forensic detector service.
Loads official pretrained TruFor weights (auto-download) and runs inference when code assets are present.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import shutil
import threading
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.request import urlretrieve
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# ── Global imports that sub-threads will need ────────────────────────────────
from PIL import Image
import cv2
import numpy as np
import torch
import torch.nn.functional as F

try:
    import matplotlib
    matplotlib.use('Agg')
except Exception:
    pass  # logger isn't defined yet; matplotlib is optional for heatmap rendering

from app.core.config import settings

logger = logging.getLogger("smartzi.trufor")

# ── Path constants ───────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models" / "trufor"
WEIGHTS_ARCHIVE_URL = "https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip"
CODE_ARCHIVE_URL = "https://github.com/grip-unina/TruFor/archive/refs/heads/main.zip"
WEIGHTS_FILE = MODELS_DIR / "trufor.pth.tar"
CODE_ROOT = MODELS_DIR / "code"
TRUFOR_TRAIN_TEST = CODE_ROOT / "TruFor-main" / "TruFor_train_test"

# Maximum input dimension (pixels) before downscaling kicks in.
# TruFor is a dense pixel-mapping model; inference cost scales quadratically with resolution.
# Capping at 512px reduces matrix computation by ~75% on CPU.
MAX_INPUT_DIM = 1024


class TruForService:
    """Singleton TruFor detector with lazy weight/code bootstrap."""

    def __init__(self):
        self.device = None
        self._model = None
        self._available = False
        self._init_attempted = False
        self._lock = threading.Lock()

    @property
    def is_available(self) -> bool:
        return self._available

    def ensure_assets(self) -> bool:
        """Download weights (and TruFor code tree) once into backend/models/trufor/."""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        if not WEIGHTS_FILE.exists():
            try:
                self._download_weights()
            except Exception as e:
                logger.error("TruFor weights download failed: %s", e)
                return False

        if not (TRUFOR_TRAIN_TEST / "lib").exists():
            try:
                self._download_code()
            except Exception as e:
                logger.warning("TruFor code download failed: %s", e)

        return WEIGHTS_FILE.exists()

    def _download_weights(self) -> None:
        archive = MODELS_DIR / "TruFor_weights.zip"
        logger.info("Downloading TruFor weights...")
        urlretrieve(WEIGHTS_ARCHIVE_URL, archive)
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(MODELS_DIR)
        archive.unlink(missing_ok=True)
        # Common layouts after extract
        if not WEIGHTS_FILE.exists():
            candidates = list(MODELS_DIR.rglob("trufor.pth.tar"))
            if candidates:
                shutil.copy2(candidates[0], WEIGHTS_FILE)

    def _download_code(self) -> None:
        archive = MODELS_DIR / "trufor_code.zip"
        logger.info("Downloading TruFor inference code...")
        urlretrieve(CODE_ARCHIVE_URL, archive)
        CODE_ROOT.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive, "r") as zf:
            for member in zf.namelist():
                if "TruFor_train_test/" in member:
                    zf.extract(member, CODE_ROOT)
        archive.unlink(missing_ok=True)

    def initialize(self) -> bool:
        if self._init_attempted:
            return self._available
            
        with self._lock:
            if self._init_attempted:
                return self._available
            self._init_attempted = True
    
            # ── CPU Performance Tuning ───────────────────────────────────────────────────
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if self.device.type == "cpu":
                torch.set_num_threads(4)
                try:
                    torch.backends.mkldnn.enabled = True
                except Exception:
                    pass
                logger.info("CPU mode: threads=4, mkldnn=%s", getattr(torch.backends.mkldnn, 'enabled', False))

            if not self.ensure_assets():
                return False
            if not (TRUFOR_TRAIN_TEST / "lib").exists():
                logger.warning("TruFor code tree missing — cannot load architecture.")
                return False

        try:
            parent = str(TRUFOR_TRAIN_TEST.parent)
            if parent not in sys.path:
                sys.path.insert(0, parent)

            # Add TRUFOR_TRAIN_TEST itself to sys.path so 'lib' can be resolved internally
            trufor_path = str(TRUFOR_TRAIN_TEST)
            if trufor_path not in sys.path:
                sys.path.insert(0, trufor_path)

            from TruFor_train_test.lib.config import config, update_config
            from TruFor_train_test.lib.utils import get_model

            class _Args:
                experiment = "trufor_ph3"
                opts = [
                    "TEST.MODEL_FILE",
                    str(WEIGHTS_FILE),
                    "WORKERS",
                    "0",
                ]

            # Temporarily change directory to TRUFOR_TRAIN_TEST so yacs merge_from_file can find relative lib/config path
            old_cwd = os.getcwd()
            try:
                os.chdir(str(TRUFOR_TRAIN_TEST))
                update_config(config, _Args())
            finally:
                os.chdir(old_cwd)

            checkpoint = torch.load(
                str(WEIGHTS_FILE),
                map_location=self.device,
                weights_only=False,
            )
            model = get_model(config)
            model.load_state_dict(checkpoint["state_dict"])
            model.to(self.device)
            model.eval()
            self._model = model
            self._config = config
            self._available = True
            logger.info("TruFor model loaded on %s", self.device)
        except Exception as e:
            logger.error("TruFor initialization failed: %s", e)
            self._model = None
            self._available = False

        return self._available

    # ── Public entry point (called directly by pipeline timeout sandbox) ──
    def analyze(self, image_path: str) -> Dict[str, Any]:
        """
        Run TruFor inference on a single image.
        
        NOTE: This method is intentionally NOT internally timeout-wrapped.
        The pipeline orchestrator (hybrid_pipeline.execute_with_timeout) is
        responsible for the timeout boundary.  Having two nested
        ThreadPoolExecutors would waste threads and complicate cancellation.
        """
        if not self.initialize():
            if not WEIGHTS_FILE.exists():
                return {"available": False, "reason": "weights_missing"}
            return {"available": False, "reason": "initialization_failed"}

        try:
            # ── Defensive image downscaling for CPU performance ───────────
            rgb = Image.open(image_path).convert("RGB")
            w, h = rgb.size
            max_dim = max(w, h)
            if max_dim > MAX_INPUT_DIM:
                scale = MAX_INPUT_DIM / max_dim
                new_w, new_h = int(w * scale), int(h * scale)
                rgb = rgb.resize((new_w, new_h), Image.BILINEAR)
                logger.info(
                    "TruFor downscaled %dx%d → %dx%d (%.0f%% reduction)",
                    w, h, new_w, new_h, (1 - scale) * 100,
                )

            arr = np.array(rgb)
            tensor = torch.tensor(arr, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            tensor = tensor.to(self.device)

            with torch.no_grad():
                pred, conf, det, _npp = self._model(tensor, save_np=False)

            score = float(torch.sigmoid(det).item()) if det is not None else None
            pred_map = torch.squeeze(pred, 0)
            pred_map = F.softmax(pred_map, dim=0)[1].cpu().numpy()
            map_mean = float(np.mean(pred_map))

            tamper_conf = score if score is not None else map_mean
            prediction = "TAMPERED" if tamper_conf > 0.5 else "AUTHENTIC"
            confidence = float(max(0.0, min(1.0, tamper_conf)))

            # Calculate tampered regions
            THRESHOLD = 0.5
            binary_mask = (pred_map > THRESHOLD).astype(np.uint8) * 255
            kernel = np.ones((5, 5), np.uint8)
            closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            tampered_regions = len([c for c in contours if cv2.contourArea(c) > 50])

            heatmap_b64 = self._heatmap_b64(pred_map)

            return {
                "available": True,
                "prediction": prediction,
                "confidence": confidence,
                "tampered_regions": tampered_regions,
                "heatmap": heatmap_b64,
                "tampered_probability": round(confidence, 4),
                "heatmap_base64": heatmap_b64,
            }
        except Exception as e:
            logger.error("TruFor inference failed: %s", e)
            return {"available": False, "reason": "inference_failed"}

    def dispose(self):
        """Clean up model and release GPU memory."""
        with self._lock:
            if self._model is not None:
                del self._model
                self._model = None
            self._available = False
            self._init_attempted = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("TruFor resources disposed cleanly.")

    @staticmethod
    def _heatmap_b64(forgery_map: np.ndarray) -> str:
        heat = (forgery_map * 255).astype(np.uint8)
        colored = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="PNG", optimize=True)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


trufor_service = TruForService()
