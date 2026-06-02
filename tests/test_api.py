# backend/tests/test_api.py
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi import status
import io

# 1. Test health check endpoint (Public)
def test_health(client):
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "smartzi-api"

# 2. Test upload unauthorized
def test_upload_unauthorized(client):
    # Temporarily remove dependency override to simulate unauthorized request
    from main import app
    from app.api.routes import get_current_user
    app.dependency_overrides.pop(get_current_user, None)
    
    response = client.post("/api/v1/upload", files={"file": ("test.png", b"data", "image/png")})
    assert response.status_code == status.HTTP_403_FORBIDDEN

# 3. Test upload success
@patch("app.api.routes._save_upload")
@patch("os.path.getsize")
def test_upload_success(mock_getsize, mock_save, client):
    mock_save.return_value = "/tmp/smartzi_uploads/dummy.png"
    mock_getsize.return_value = 15360 # 15 KB
    
    file_payload = {"file": ("test.png", b"fake_png_data", "image/png")}
    response = client.post("/api/v1/upload", files=file_payload)
    
    assert response.status_code == status.HTTP_201_CREATED
    data = response.json()
    assert "upload_id" in data
    assert data["filename"] == "test.png"
    assert data["file_size_kb"] == 15.0

# 4. Test upload invalid mime type
def test_upload_invalid_mime(client):
    file_payload = {"file": ("test.txt", b"plain text data", "text/plain")}
    response = client.post("/api/v1/upload", files=file_payload)
    assert response.status_code == status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    assert "must be an image" in response.json()["detail"]

# 5. Test analyze missing upload_id
def test_analyze_missing_id(client):
    response = client.post("/api/v1/analyze", data={})
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

# 6. Test analyze upload not found
@patch("app.api.routes._find_uploaded_file")
def test_analyze_not_found(mock_find, client):
    from fastapi import HTTPException
    mock_find.side_effect = HTTPException(status_code=404, detail="Upload dummy not found")
    
    response = client.post("/api/v1/analyze", data={"upload_id": "missing_id"})
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert "not found" in response.json()["detail"]

# 7. Test analyze under Gemini-dominant success path
@patch("app.api.routes._find_uploaded_file")
@patch("app.services.hybrid_pipeline.run_detection_pipeline")
def test_analyze_gemini_dominant_success(mock_pipeline, mock_find, client, mock_firestore):
    mock_find.return_value = "/tmp/smartzi_uploads/dummy.png"
    mock_pipeline.return_value = {
        "prediction": "AI_GENERATED",
        "confidence": 0.95,
        "ai_probability": 0.95,
        "tampered_probability": 0.0,
        "ela_score": 0.0,
        "fft_score": 0.0,
        "noise_score": 0.0,
        "metadata_score": 0.0,
        "gemini_explanation": "Exquisite synthetic details.",
        "is_gemini_dominant": True,
        "heatmap_base64": "",
        "processing_time": 0.85
    }
    
    response = client.post("/api/v1/analyze", data={"upload_id": "mock_id_123", "filename": "test.png"})
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["is_gemini_dominant"] is True
    assert data["prediction"] == "AI_GENERATED"
    assert data["confidence"] == 0.95
    assert data["gemini_explanation"] == "Exquisite synthetic details."
    mock_firestore.create_scan.assert_called_once()

# 8. Test analyze under local models fallback success path
@patch("app.api.routes._find_uploaded_file")
@patch("app.services.hybrid_pipeline.run_detection_pipeline")
def test_analyze_fallback_local_success(mock_pipeline, mock_find, client, mock_firestore):
    mock_find.return_value = "/tmp/smartzi_uploads/dummy.png"
    mock_pipeline.return_value = {
        "prediction": "TAMPERED",
        "confidence": 0.72,
        "ai_probability": 0.15,
        "tampered_probability": 0.78,
        "ela_score": 0.45,
        "fft_score": 0.32,
        "noise_score": 0.65,
        "metadata_score": 0.15,
        "gemini_explanation": "Gemini service unavailable. Analysis performed using local forensic models.",
        "is_gemini_dominant": False,
        "heatmap_base64": "base64_heatmap_string",
        "processing_time": 4.5
    }
    
    response = client.post("/api/v1/analyze", data={"upload_id": "mock_id_fallback", "filename": "test.png"})
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["is_gemini_dominant"] is False
    assert data["prediction"] == "TAMPERED"
    assert data["confidence"] == 0.72
    assert data["tampered_probability"] == 0.78
    assert data["heatmap_base64"] == "base64_heatmap_string"
    mock_firestore.create_scan.assert_called_once()

# 9. Test scan history retrieving successfully when empty
def test_history_empty(client, mock_firestore):
    mock_firestore.get_user_scans.return_value = []
    
    response = client.get("/api/v1/history")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["total"] == 0
    assert len(data["items"]) == 0
    assert data["user_id"] == "test_user_123"

# 10. Test scan history retrieving successfully when populated
def test_history_populated(client, mock_firestore):
    from datetime import datetime, timezone
    mock_firestore.get_user_scans.return_value = [
        {
            "scan_id": "scan_123",
            "filename": "spliced.png",
            "prediction": "TAMPERED",
            "confidence": 0.85,
            "created_at": datetime.now(timezone.utc)
        }
    ]
    
    response = client.get("/api/v1/history")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["total"] == 1
    assert data["items"][0]["scan_id"] == "scan_123"
    assert data["items"][0]["verdict"] == "TAMPERED"
    assert data["items"][0]["risk_score"] == 85.0

# 11. Test fetch single scan not found
def test_get_scan_not_found(client, mock_firestore):
    mock_firestore.get_scan.return_value = None
    response = client.get("/api/v1/scan/non_existent_scan_id")
    assert response.status_code == status.HTTP_404_NOT_FOUND

# 12. Test fetch single scan access denied (different owner)
def test_get_scan_forbidden(client, mock_firestore):
    mock_firestore.get_scan.return_value = {
        "scan_id": "scan_other_user",
        "user_id": "another_uid_456",
        "filename": "secret.png"
    }
    response = client.get("/api/v1/scan/scan_other_user")
    assert response.status_code == status.HTTP_403_FORBIDDEN

# 13. Test fetch single scan successfully
def test_get_scan_success(client, mock_firestore):
    from datetime import datetime, timezone
    mock_firestore.get_scan.return_value = {
        "scan_id": "scan_ok",
        "upload_id": "upload_ok",
        "user_id": "test_user_123",
        "filename": "validated.png",
        "created_at": datetime.now(timezone.utc),
        "prediction": "AUTHENTIC",
        "confidence": 0.12,
        "ai_probability": 0.05,
        "tampered_probability": 0.08,
        "ela_score": 0.0,
        "fft_score": 0.0,
        "noise_score": 0.0,
        "metadata_score": 0.0,
        "gemini_explanation": "Clean pixels.",
        "is_gemini_dominant": True,
        "heatmap_base64": "",
        "processing_time": 0.32
    }
    
    response = client.get("/api/v1/scan/scan_ok")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["scan_id"] == "scan_ok"
    assert data["prediction"] == "AUTHENTIC"
    assert data["is_gemini_dominant"] is True

# 14. Test chat prompt cannot be empty
def test_chat_empty_prompt(client):
    response = client.post("/api/v1/chat", json={"prompt": "  "})
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json()["error_code"] == "EMPTY_PROMPT"

# 15. Test chat success
@patch("app.services.gemini_service.gemini_service.chat", new_callable=AsyncMock)
def test_chat_success(mock_chat, client):
    mock_chat.return_value = "Forensics answer explanation."
    
    response = client.post("/api/v1/chat", json={"prompt": "How does FFT work?"})
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["success"] is True
    assert data["response"] == "Forensics answer explanation."
    mock_chat.assert_called_once_with(prompt="How does FFT work?")

# 16. Test chat gemini service unavailable error handling
@patch("app.services.gemini_service.gemini_service.chat", new_callable=AsyncMock)
def test_chat_gemini_failure(mock_chat, client):
    mock_chat.side_effect = ConnectionError("All Gemini API keys exhausted for chat.")
    
    response = client.post("/api/v1/chat", json={"prompt": "Tell me about EXIF."})
    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    data = response.json()
    assert data["success"] is False
    assert data["error_code"] == "GEMINI_UNAVAILABLE"

# 17. Test chat unexpected internal server error
@patch("app.services.gemini_service.gemini_service.chat", new_callable=AsyncMock)
def test_chat_internal_server_error(mock_chat, client):
    mock_chat.side_effect = RuntimeError("Unexpected segfault in model inference")
    
    response = client.post("/api/v1/chat", json={"prompt": "What is noise analysis?"})
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    data = response.json()
    assert data["success"] is False
    assert data["error_code"] == "INTERNAL_SERVER_ERROR"

# 18. Test upload rejects oversized file via chunked write guard
@patch("app.api.routes._save_upload")
def test_upload_file_too_large(mock_save, client):
    from fastapi import HTTPException
    mock_save.side_effect = HTTPException(
        status_code=413,
        detail="File too large. Maximum size is 10 MB"
    )
    
    file_payload = {"file": ("huge.png", b"x" * 1024, "image/png")}
    response = client.post("/api/v1/upload", files=file_payload)
    assert response.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
    assert "too large" in response.json()["detail"].lower()

# 19. Test analyze catches pipeline crash and returns 500
@patch("app.api.routes._find_uploaded_file")
@patch("app.services.hybrid_pipeline.run_detection_pipeline")
def test_analyze_pipeline_crash(mock_pipeline, mock_find, client, mock_firestore):
    mock_find.return_value = "/tmp/smartzi_uploads/dummy.png"
    mock_pipeline.side_effect = Exception("CUDA out of memory")
    
    response = client.post("/api/v1/analyze", data={"upload_id": "crash_id", "filename": "bomb.png"})
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "pipeline failed" in response.json()["detail"].lower()

# 20. Test history limit is capped at 50
def test_history_limit_capped(client, mock_firestore):
    mock_firestore.get_user_scans.return_value = []
    
    response = client.get("/api/v1/history?limit=999")
    assert response.status_code == status.HTTP_200_OK
    # Verify that get_user_scans was called with min(999, 50) = 50
    call_args = mock_firestore.get_user_scans.call_args
    assert call_args.kwargs.get("limit") == 50 or call_args[1].get("limit") == 50

# 21. Test health check response contains all expected fields
def test_health_response_structure(client):
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "status" in data
    assert "service" in data
    assert "version" in data
    assert data["version"] == "1.0.0"

# 22. Test upload with missing file field returns 422
def test_upload_missing_file_field(client):
    response = client.post("/api/v1/upload")
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

# 23. Test history returns multiple items in correct schema
def test_history_multiple_items(client, mock_firestore):
    from datetime import datetime, timezone
    mock_firestore.get_user_scans.return_value = [
        {
            "scan_id": "scan_a",
            "filename": "photo1.jpg",
            "prediction": "AUTHENTIC",
            "confidence": 0.12,
            "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc)
        },
        {
            "scan_id": "scan_b",
            "filename": "photo2.png",
            "prediction": "TAMPERED",
            "confidence": 0.88,
            "created_at": datetime(2026, 1, 2, tzinfo=timezone.utc)
        },
        {
            "scan_id": "scan_c",
            "filename": "photo3.webp",
            "prediction": "AI_GENERATED",
            "confidence": 0.95,
            "created_at": datetime(2026, 1, 3, tzinfo=timezone.utc)
        },
    ]
    
    response = client.get("/api/v1/history")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["total"] == 3
    assert len(data["items"]) == 3
    # Verify each item has the required fields
    for item in data["items"]:
        assert "scan_id" in item
        assert "filename" in item
        assert "verdict" in item
        assert "risk_score" in item
        assert "created_at" in item

# 24. Test analyze with pipeline ValueError returns 422
@patch("app.api.routes._find_uploaded_file")
@patch("app.services.hybrid_pipeline.run_detection_pipeline")
def test_analyze_pipeline_validation_error(mock_pipeline, mock_find, client, mock_firestore):
    mock_find.return_value = "/tmp/smartzi_uploads/dummy.png"
    mock_pipeline.side_effect = ValueError("Image is fully transparent — cannot analyze")
    
    response = client.post("/api/v1/analyze", data={"upload_id": "bad_img_id", "filename": "transparent.png"})
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert "transparent" in response.json()["detail"].lower()

# 25. Test chat validation error (non-empty but invalid prompt)
@patch("app.services.gemini_service.gemini_service.chat", new_callable=AsyncMock)
def test_chat_validation_error(mock_chat, client):
    mock_chat.side_effect = ValueError("Prompt contains disallowed characters")
    
    response = client.post("/api/v1/chat", json={"prompt": "valid-looking prompt"})
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    data = response.json()
    assert data["success"] is False
    assert data["error_code"] == "VALIDATION_ERROR"
