"""
SMARTZI - Firebase Integration
Handles Firestore DB operations and Firebase Auth token verification.
"""

import logging
from typing import Optional, Dict, Any
import firebase_admin
from firebase_admin import credentials, firestore, auth, storage
from app.core.config import settings

logger = logging.getLogger("smartzi.firebase")

# ── Firebase App Initialization ──────────────────────────────────────────────
_firebase_app: Optional[firebase_admin.App] = None


def get_firebase_app() -> firebase_admin.App:
    """Lazily initialize and return the Firebase app singleton."""
    global _firebase_app
    if _firebase_app is None:
        try:
            cred = credentials.Certificate(settings.FIREBASE_CREDENTIALS_PATH)
            _firebase_app = firebase_admin.initialize_app(
                cred,
                {"storageBucket": f"{settings.FIREBASE_PROJECT_ID}.appspot.com"},
            )
            logger.info("Firebase app initialized successfully")
        except Exception as e:
            logger.error("Firebase initialization failed: %s", e)
            raise
    return _firebase_app


def get_firestore_client():
    """Return an authenticated Firestore client."""
    get_firebase_app()
    return firestore.client()


# ── Auth ─────────────────────────────────────────────────────────────────────
async def verify_firebase_token(id_token: str) -> Optional[Dict[str, Any]]:
    """
    Verify a Firebase ID token and return decoded claims.
    Returns None if token is invalid.
    """
    try:
        get_firebase_app()
        decoded = auth.verify_id_token(id_token)
        return decoded
    except Exception as e:
        logger.warning("Token verification failed: %s", e)
        return None


# ── Firestore Operations ──────────────────────────────────────────────────────
class FirestoreService:
    """Abstraction layer for all Firestore read/write operations."""

    @property
    def db(self):
        return get_firestore_client()

    # ── Users Collection ─────────────────────────────────────────────────────
    async def upsert_user(self, uid: str, data: Dict[str, Any]) -> None:
        """Create or update a user document."""
        ref = self.db.collection("users").document(uid)
        ref.set(data, merge=True)
        logger.debug("Upserted user document: %s", uid)

    async def get_user(self, uid: str) -> Optional[Dict[str, Any]]:
        """Fetch a user document by UID."""
        doc = self.db.collection("users").document(uid).get()
        return doc.to_dict() if doc.exists else None

    # ── Scans Collection ─────────────────────────────────────────────────────
    async def create_scan(self, scan_data: Dict[str, Any]) -> str:
        """Insert a new scan document and return its ID."""
        ref = self.db.collection("scans").add(scan_data)
        scan_id = ref[1].id
        logger.debug("Created scan document: %s", scan_id)
        return scan_id

    async def get_scan(self, scan_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single scan by its document ID."""
        doc = self.db.collection("scans").document(scan_id).get()
        return doc.to_dict() if doc.exists else None

    async def get_user_scans(self, uid: str, limit: int = 20) -> list:
        """Fetch the most recent scans for a user."""
        query = (
            self.db.collection("scans")
            .where("user_id", "==", uid)
            .order_by("created_at", direction=firestore.Query.DESCENDING)
            .limit(limit)
        )
        docs = query.stream()
        return [{"id": d.id, **d.to_dict()} for d in docs]

    # ── Results Collection ────────────────────────────────────────────────────
    async def save_result(self, result_data: Dict[str, Any]) -> str:
        """Persist an analysis result document."""
        ref = self.db.collection("results").add(result_data)
        return ref[1].id


firestore_service = FirestoreService()
