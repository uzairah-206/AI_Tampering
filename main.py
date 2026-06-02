"""
SMARTZI Backend - FastAPI Entry Point
AI-powered Image Metadata & Tampering Detection
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import uvicorn

from app.api.routes import router
from app.core.config import settings
from app.core.logging import setup_logging
from app.services.model_manager import model_manager

# Initialize logging
setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: run validations & warm up models. Shutdown: release GPU/memory resources."""
    try:
        from app.core.startup_validator import startup_validator
        startup_validator.generate_report()
    except Exception as e:
        import logging
        logging.getLogger("smartzi.main").error("Startup validator failed to run: %s", e)

    await model_manager.initialize()
    yield
    await model_manager.dispose()


# Create FastAPI app instance
app = FastAPI(
    title="SMARTZI API",
    description="AI-powered Image Metadata & Tampering Detection API",
    version="1.0.0",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    lifespan=lifespan,
)

# ── Middleware ──────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS,
)

# ── Routers ─────────────────────────────────────────────────────────────────
app.include_router(router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    """Health check endpoint for deployment monitoring."""
    return {"status": "healthy", "service": "smartzi-api", "version": "1.0.0"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        workers=1,  # Use 1 worker for free-tier deployment
    )
