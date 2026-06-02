# backend/tests/conftest.py
import pytest
from unittest.mock import AsyncMock, MagicMock
import sys

# Pre-emptively mock firebase_admin to prevent it from loading credentials
sys.modules['firebase_admin'] = MagicMock()
sys.modules['firebase_admin.credentials'] = MagicMock()
sys.modules['firebase_admin.firestore'] = MagicMock()
sys.modules['firebase_admin.auth'] = MagicMock()
sys.modules['firebase_admin.storage'] = MagicMock()

# Import the actual application objects
from main import app
from app.api.routes import get_current_user
from app.core.firebase import firestore_service
from app.core.startup_validator import startup_validator
from app.services.model_manager import model_manager
from app.services.trufor_service import trufor_service
from app.services.aide_service import aide_service

# Apply permanent mock overrides directly to the global singletons
startup_validator.generate_report = MagicMock(return_value={"status": "HEALTHY"})
startup_validator.validate = MagicMock(return_value={
    "torch": True, "cuda": False, "weights": True, "trufor": "ACTIVE", "mantranet": False, "gemini": True, "warnings": []
})

model_manager.initialize = AsyncMock()
model_manager.dispose = AsyncMock()

trufor_service.initialize = MagicMock(return_value=True)
trufor_service.ensure_assets = MagicMock(return_value=True)
trufor_service.analyze = MagicMock(return_value={
    "available": True, "prediction": "AUTHENTIC", "confidence": 0.1, "tampered_probability": 0.1, "heatmap_base64": ""
})

aide_service.initialize = MagicMock(return_value=True)
aide_service.predict = MagicMock(return_value={
    "available": True, "confidence": 0.1, "ai_probability": 0.1, "tampered_probability": 0.1, "source": "aide_detector"
})

@pytest.fixture
def mock_user():
    return {
        "uid": "test_user_123",
        "email": "test@smartzi.com",
        "name": "Test User"
    }

@pytest.fixture
def client(mock_user):
    # Override get_current_user dependency to bypass Firebase Auth
    async def override_get_current_user():
        return mock_user
        
    app.dependency_overrides[get_current_user] = override_get_current_user
    
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        yield c
            
    app.dependency_overrides.clear()

@pytest.fixture
def mock_firestore():
    # Mock FirestoreService methods
    firestore_service.create_scan = AsyncMock(return_value="mock_scan_id_abc")
    firestore_service.get_scan = AsyncMock()
    firestore_service.get_user_scans = AsyncMock(return_value=[])
    firestore_service.upsert_user = AsyncMock()
    firestore_service.get_user = AsyncMock()
    return firestore_service
