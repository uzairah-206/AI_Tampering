"""
SMARTZI - API Route Handlers
All public API endpoints with input validation and auth middleware.

Routes:
    POST /upload        — Upload an image; returns upload_id
    POST /analyze       — Run AI analysis on an uploaded image
    GET  /history       — Fetch scan history for authenticated user
    GET  /scan/{id}     — Fetch a single scan result
"""

import logging
import os
import shutil
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import settings
from app.core.firebase import firestore_service, verify_firebase_token
from app.schemas.analysis import AnalysisResult, HistoryResponse, ScanHistoryItem, UploadResponse, ChatRequest, ChatSuccessResponse, ChatFailureResponse
from app.validators.request_validator import RequestValidator

logger = logging.getLogger("smartzi.api")
router = APIRouter()
security = HTTPBearer()


# ── Auth Dependency ───────────────────────────────────────────────────────────
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """
    FastAPI dependency: verify Firebase Bearer token and return decoded claims.
    Raises HTTP 401 if token is missing or invalid.
    """
    token = credentials.credentials
    claims = await verify_firebase_token(token)
    if claims is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token",
        )
    return claims


# ── Helpers ───────────────────────────────────────────────────────────────────
def _validate_file(file: UploadFile) -> None:
    """Validate file extension and size before processing."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="File must be an image")

    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: .{ext}. Allowed: {settings.ALLOWED_EXTENSIONS}",
        )


def _save_upload(file: UploadFile, upload_id: str) -> str:
    """Persist an uploaded file to the local temp directory."""
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    ext = file.filename.rsplit(".", 1)[-1].lower()
    dest = os.path.join(settings.UPLOAD_DIR, f"{upload_id}.{ext}")

    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    written = 0
    with open(dest, "wb") as f:
        while chunk := file.file.read(8192):
            written += len(chunk)
            if written > max_bytes:
                f.close()
                os.remove(dest)
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum size is {settings.MAX_FILE_SIZE_MB} MB",
                )
            f.write(chunk)

    try:
        RequestValidator.validate(dest, file.filename)
    except ValueError as e:
        os.remove(dest)
        raise HTTPException(status_code=422, detail=str(e))

    return dest


def _find_uploaded_file(upload_id: str) -> str:
    """Locate an uploaded file by its upload_id (any allowed extension)."""
    for ext in settings.ALLOWED_EXTENSIONS:
        path = os.path.join(settings.UPLOAD_DIR, f"{upload_id}.{ext}")
        if os.path.exists(path):
            return path
    raise HTTPException(status_code=404, detail=f"Upload {upload_id!r} not found")


def _parse_created_at(val) -> datetime:
    """Safely parse created_at into a timezone-aware datetime object."""
    if isinstance(val, datetime):
        if val.tzinfo is None:
            return val.replace(tzinfo=timezone.utc)
        return val
    if isinstance(val, str):
        try:
            return datetime.fromisoformat(val.replace("Z", "+00:00"))
        except Exception:
            pass
    return datetime.now(timezone.utc)


# ── POST /upload ──────────────────────────────────────────────────────────────
@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload an image for analysis",
)
async def upload_image(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, WEBP, etc.)"),
    user: dict = Depends(get_current_user),
):
    """
    Upload an image and receive an upload_id.
    The upload_id is used as input to /analyze.
    """
    _validate_file(file)
    upload_id = str(uuid.uuid4())
    saved_path = _save_upload(file, upload_id)
    size_kb = round(os.path.getsize(saved_path) / 1024, 2)

    logger.info(
        "Image uploaded | user=%s upload_id=%s filename=%s size=%.1fKB",
        user["uid"], upload_id, file.filename, size_kb,
    )

    return UploadResponse(
        upload_id=upload_id,
        filename=file.filename or "unknown",
        file_size_kb=size_kb,
    )


# ── POST /analyze ─────────────────────────────────────────────────────────────
@router.post(
    "/analyze",
    response_model=AnalysisResult,
    summary="Run AI analysis on an uploaded image",
)
async def analyze_image(
    upload_id: str = Form(..., description="upload_id returned from /upload"),
    filename: Optional[str] = Form(None),
    user: dict = Depends(get_current_user),
):
    """
    Trigger the full AI pipeline on a previously uploaded image.
    """
    image_path = _find_uploaded_file(upload_id)
    resolved_filename = filename or os.path.basename(image_path)

    try:
        from app.services.hybrid_pipeline import run_detection_pipeline
        import asyncio
        
        # Run new unified pipeline
        result = await asyncio.to_thread(run_detection_pipeline, image_path)
        
        scan_id = str(uuid.uuid4())
        created_at_dt = datetime.now(timezone.utc)
        
        # Direct Firestore injection of the exact 11-key flat schema
        firestore_payload = {
            "scan_id": scan_id,
            "upload_id": upload_id,
            "user_id": user["uid"],
            "filename": resolved_filename,
            "created_at": created_at_dt,
            **result
        }
        
        await firestore_service.create_scan(firestore_payload)
        
        # Convert created_at to ISO string for Pydantic response serialization
        response_payload = {**firestore_payload}
        response_payload["created_at"] = created_at_dt.isoformat()
        return response_payload
        
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Analysis failed for upload_id=%s: %s", upload_id, e)
        raise HTTPException(status_code=500, detail="Analysis pipeline failed")


# ── GET /history ──────────────────────────────────────────────────────────────
@router.get(
    "/history",
    response_model=HistoryResponse,
    summary="Fetch recent scan history for the authenticated user",
)
async def get_history(
    limit: int = 20,
    user: dict = Depends(get_current_user),
):
    """Return the user's most recent scans, newest first."""
    scans = await firestore_service.get_user_scans(user["uid"], limit=min(limit, 50))

    items = [
        ScanHistoryItem(
            scan_id=s["scan_id"],
            filename=s.get("filename", "unknown"),
            verdict=s.get("prediction", s.get("verdict", "UNKNOWN")),
            risk_score=(s.get("confidence") * 100) if "confidence" in s else s.get("risk_score", 0.0),
            created_at=_parse_created_at(s.get("created_at")),
        )
        for s in scans
    ]

    return HistoryResponse(items=items, total=len(items), user_id=user["uid"])


# ── GET /scan/{scan_id} ────────────────────────────────────────────────────────
@router.get(
    "/scan/{scan_id}",
    response_model=AnalysisResult,
    summary="Fetch a single scan result by ID",
)
async def get_scan(
    scan_id: str,
    user: dict = Depends(get_current_user),
):
    """Retrieve a previously completed scan. Only the owner can access it."""
    doc = await firestore_service.get_scan(scan_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Scan not found")
    if doc.get("user_id") != user["uid"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    response_payload = {**doc}
    if "created_at" in response_payload:
        dt = response_payload["created_at"]
        if isinstance(dt, datetime):
            response_payload["created_at"] = dt.isoformat()
        elif isinstance(dt, str):
            # Already a string
            pass
    return response_payload


# ── POST /chat ────────────────────────────────────────────────────────────────
@router.post(
    "/chat",
    response_model=ChatSuccessResponse,
    responses={
        200: {"model": ChatSuccessResponse},
        400: {"model": ChatFailureResponse},
        500: {"model": ChatFailureResponse},
    },
    summary="Generic text chat endpoint with AI forensics copilot",
)
async def chat_endpoint(
    req: ChatRequest,
    user: dict = Depends(get_current_user),
):
    """
    Accepts text queries from user regarding image forensics, ELA, EXIF metadata, etc.
    Queries the Gemini service and returns the formatted response.
    """
    from app.services.gemini_service import gemini_service

    if not req.prompt or not req.prompt.strip():
        return JSONResponse(
            status_code=400,
            content=ChatFailureResponse(
                success=False,
                error_code="EMPTY_PROMPT"
            ).model_dump()
        )

    try:
        response_text = await gemini_service.chat(prompt=req.prompt)
        return ChatSuccessResponse(
            success=True,
            response=response_text
        )
    except ValueError as e:
        logger.error("Chat validation failure: %s", e)
        return JSONResponse(
            status_code=400,
            content=ChatFailureResponse(
                success=False,
                error_code="VALIDATION_ERROR"
            ).model_dump()
        )
    except ConnectionError as e:
        logger.error("Chat network/service failure: %s", e)
        return JSONResponse(
            status_code=503,
            content=ChatFailureResponse(
                success=False,
                error_code="GEMINI_UNAVAILABLE"
            ).model_dump()
        )
    except Exception as e:
        logger.exception("Unexpected error in chat endpoint: %s", e)
        return JSONResponse(
            status_code=500,
            content=ChatFailureResponse(
                success=False,
                error_code="INTERNAL_SERVER_ERROR"
            ).model_dump()
        )

