"""
SMARTZI - Application Configuration
Reads from environment variables with sensible defaults.
"""

import os
import tempfile
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    # ── App ─────────────────────────────────────────────────────────────────
    APP_NAME: str = "SMARTZI"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "smartzi-secret-change-in-production")
    
    # ── External API Keys ───────────────────────────────────────────────────
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


    # ── CORS ────────────────────────────────────────────────────────────────
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8080",
        "https://smartzi.app",
        "*",  # Restrict in production
    ]
    ALLOWED_HOSTS: List[str] = ["*"]

    # ── Firebase ────────────────────────────────────────────────────────────
    FIREBASE_PROJECT_ID: str = os.getenv("FIREBASE_PROJECT_ID", "")
    FIREBASE_CREDENTIALS_PATH: str = os.getenv(
        "FIREBASE_CREDENTIALS_PATH", "firebase-credentials.json"
    )

    # ── File Upload ──────────────────────────────────────────────────────────
    MAX_FILE_SIZE_MB: int = 10
    ALLOWED_EXTENSIONS: List[str] = ["jpg", "jpeg", "png", "webp", "bmp", "tiff"]
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", os.path.join(tempfile.gettempdir(), "smartzi_uploads"))

    # ── AI Model ────────────────────────────────────────────────────────────
    MODEL_PATH: str = os.getenv("MODEL_PATH", str(BASE_DIR / "app" / "services" / "models" / "ela_classifier.pth"))
    DEVICE: str = "cpu"  # Use CPU for free-tier deployment
    ELA_QUALITY: int = 75  # JPEG quality for ELA computation
    ELA_SCALE: int = 15   # ELA amplification scale

    # ── ManTraNet ────────────────────────────────────────────────────────
    MANTRANET_WEIGHTS_PATH: str = os.getenv(
        "MANTRANET_WEIGHTS_PATH", str(BASE_DIR / "models" / "mantranet" / "MantraNetv4.pt")
    )

    # ── CNNSpot / Wang2020 ──────────────────────────────────────────────
    CNNSPOT_WEIGHTS_PATH: str = os.getenv(
        "CNNSPOT_WEIGHTS_PATH", str(BASE_DIR / "models" / "cnnspot" / "blur_jpg_prob0.5.pth")
    )

    # ── Hybrid Pipeline Thresholds ──────────────────────────────────────
    TRUFOR_TAMPER_THRESHOLD: float = 0.5
    CNNSPOT_AI_THRESHOLD: float = 0.5

    # ── Rate Limiting ────────────────────────────────────────────────────────
    RATE_LIMIT_PER_MINUTE: int = 20

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
