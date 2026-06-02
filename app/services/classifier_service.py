"""
SMARTZI - CNN Classifier Service
MobileNetV3 binary classifier — inference only when fine-tuned weights are present.
"""

import logging
import os
from PIL import Image

TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    import torchvision.models as models
    TORCH_AVAILABLE = True
except ImportError as e:
    logging.getLogger("smartzi.classifier").error(
        "PyTorch unavailable: %s. Classifier disabled.", e
    )

from app.core.config import settings
from app.schemas.analysis import ClassifierResult

logger = logging.getLogger("smartzi.classifier")


def _disabled_result() -> ClassifierResult:
    return ClassifierResult(
        status="MODEL_DISABLED",
        prediction=None,
        confidence=None,
        authentic_probability=None,
        tampered_probability=None,
        label="UNCERTAIN",
        source="fallback",
    )


if TORCH_AVAILABLE:
    class SmartziClassifier(nn.Module):
        def __init__(self, pretrained: bool = True):
            super().__init__()
            backbone = models.mobilenet_v3_small(
                weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            )
            self.features = backbone.features
            self.avgpool = backbone.avgpool
            self.classifier = nn.Sequential(
                nn.Linear(576, 256),
                nn.Hardswish(),
                nn.Dropout(p=0.3),
                nn.Linear(256, 2),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            return self.classifier(x)

    INFERENCE_TRANSFORM = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
else:
    SmartziClassifier = None  # type: ignore
    INFERENCE_TRANSFORM = None


class ClassifierService:
    def __init__(self):
        self.device = None
        self.model = None
        self._weights_loaded = False
        if TORCH_AVAILABLE:
            self.device = torch.device(settings.DEVICE)
            self._load_model()

    def _load_model(self):
        if not os.path.exists(settings.MODEL_PATH):
            logger.warning(
                "No checkpoint at %s — classifier disabled (no untrained inference).",
                settings.MODEL_PATH,
            )
            return

        try:
            self.model = SmartziClassifier(pretrained=True)
            state_dict = torch.load(settings.MODEL_PATH, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            self._weights_loaded = True
            logger.info("Loaded fine-tuned weights from %s", settings.MODEL_PATH)
        except Exception as e:
            logger.error("Classifier load failed: %s", e)
            self.model = None
            self._weights_loaded = False

    def classify(self, image_path: str) -> ClassifierResult:
        if not os.path.exists(image_path):
            raise ValueError(f"Cannot open image for classification: File not found: {image_path}")

        if not TORCH_AVAILABLE or not self._weights_loaded or self.model is None:
            logger.info("Classifier MODEL_DISABLED — missing PyTorch or trained weights.")
            return _disabled_result()

        try:
            with Image.open(image_path) as img:
                img_rgb = img.convert("RGB")
        except Exception as e:
            raise ValueError(f"Cannot open image for classification: {e}")

        with torch.no_grad():
            tensor = INFERENCE_TRANSFORM(img_rgb).unsqueeze(0).to(self.device)
            probs = torch.softmax(self.model(tensor), dim=1).squeeze(0)

        authentic_prob = float(probs[0])
        tampered_prob = float(probs[1])
        label = "TAMPERED" if tampered_prob > authentic_prob else "AUTHENTIC"
        confidence = max(authentic_prob, tampered_prob)

        return ClassifierResult(
            status="ACTIVE",
            prediction=label,
            label=label,
            confidence=round(confidence, 4),
            authentic_probability=round(authentic_prob, 4),
            tampered_probability=round(tampered_prob, 4),
            source="mobilenetv3",
        )


classifier_service = ClassifierService()
