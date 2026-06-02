"""
SMARTZI - Pydantic Schemas
Request/response models for API validation and serialization.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# ── Upload ───────────────────────────────────────────────────────────────────
class UploadResponse(BaseModel):
    """Response returned after a successful image upload."""
    upload_id: str = Field(..., description="Unique identifier for the uploaded image")
    filename: str
    file_size_kb: float
    message: str = "Image uploaded successfully"


# ── Metadata ─────────────────────────────────────────────────────────────────
class ImageMetadata(BaseModel):
    """Extracted EXIF and file metadata."""
    filename: str
    file_size_kb: float
    format: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    mode: Optional[str] = None
    # EXIF fields
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    date_taken: Optional[str] = None
    gps_latitude: Optional[float] = None
    gps_longitude: Optional[float] = None
    software: Optional[str] = None
    orientation: Optional[int] = None
    iso: Optional[int] = None
    shutter_speed: Optional[str] = None
    aperture: Optional[str] = None
    focal_length: Optional[str] = None
    flash: Optional[str] = None
    color_space: Optional[str] = None
    has_exif: bool = False
    raw_exif: Dict[str, Any] = Field(default_factory=dict)


# ── AI Analysis ───────────────────────────────────────────────────────────────
class ELAResult(BaseModel):
    """Error Level Analysis output."""
    ela_mean: float = Field(..., description="Mean ELA pixel intensity")
    ela_max: float = Field(..., description="Max ELA pixel intensity")
    ela_std: float = Field(..., description="Standard deviation of ELA")
    suspicious_regions: int = Field(..., description="Number of high-ELA regions")
    heatmap_base64: Optional[str] = Field(None, description="Base64 PNG heatmap")


class ClassifierResult(BaseModel):
    """CNN classifier output."""
    status: str = Field(default="ACTIVE", description="Model status: ACTIVE or MODEL_DISABLED")
    prediction: Optional[str] = Field(default=None, description="'AUTHENTIC' or 'TAMPERED'")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    authentic_probability: Optional[float] = None
    tampered_probability: Optional[float] = None
    label: str = "UNCERTAIN"
    source: Optional[str] = Field(default=None, description="Inference source or 'fallback'")



class MantraNetResult(BaseModel):
    """ManTraNet pixel-level forgery detection output."""
    forgery_score: float = Field(..., ge=0.0, le=1.0, description="Mean forgery probability across all pixels")
    forgery_max: float = Field(..., ge=0.0, le=1.0, description="Peak forgery probability")
    tampered_regions: int = Field(..., description="Number of detected tampered regions")
    tampered_area_pct: float = Field(..., ge=0.0, le=100.0, description="Percentage of image area flagged as tampered")
    heatmap_base64: Optional[str] = Field(None, description="Base64 PNG forgery heatmap")


class AnalysisResult(BaseModel):
    """Flat analysis result matching the hybrid pipeline output."""
    scan_id: str
    upload_id: str
    user_id: str
    filename: str
    created_at: str = Field(..., description="ISO 8601 UTC timestamp string")

    # Pipeline verdict
    prediction: str = Field("UNCERTAIN", description="'AUTHENTIC' or 'TAMPERED' or 'AI_GENERATED' or 'UNCERTAIN'")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Fused confidence score [0.0 - 1.0]")
    ai_probability: float = Field(0.0, ge=0.0, le=1.0, description="Probability of image being AI-generated [0.0 - 1.0]")
    tampered_probability: float = Field(0.0, ge=0.0, le=1.0, description="Probability of image being digitally tampered [0.0 - 1.0]")

    # Forensic signal scores
    ela_score: float = Field(0.0, description="Error Level Analysis score [0.0 - 1.0]")
    fft_score: float = Field(0.0, description="FFT frequency analysis score [0.0 - 1.0]")
    noise_score: float = Field(0.0, description="Noise consistency score [0.0 - 1.0]")
    metadata_score: float = Field(0.0, description="Metadata anomaly score [0.0 - 1.0]")

    # Gemini analysis
    gemini_explanation: str = Field("", description="Gemini forensic explanation text")
    is_gemini_dominant: bool = Field(False, description="True if Gemini was the primary analysis source")

    # Optional artifacts
    heatmap_base64: str = Field("", description="Base64 TruFor heatmap image")
    processing_time: float = Field(0.0, description="Pipeline execution time in seconds")


# ── History ───────────────────────────────────────────────────────────────────
class ScanHistoryItem(BaseModel):
    """Lightweight scan item for history listing."""
    scan_id: str
    filename: str
    verdict: str
    risk_score: float
    created_at: datetime
    thumbnail_url: Optional[str] = None


class HistoryResponse(BaseModel):
    """Paginated history response."""
    items: List[ScanHistoryItem]
    total: int
    user_id: str


# ── Error ─────────────────────────────────────────────────────────────────────
class ErrorResponse(BaseModel):
    """Standard error response envelope."""
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None


# ── Chat ──────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    prompt: str = Field(..., description="The user's query prompt.")


class ChatSuccessResponse(BaseModel):
    success: bool = True
    response: str = Field(..., description="The response from the assistant.")


class ChatFailureResponse(BaseModel):
    success: bool = False
    error_code: str = Field(..., description="Error classification code.")

