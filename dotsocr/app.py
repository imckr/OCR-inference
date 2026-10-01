"""
Main FastAPI application assembly.

Mount all routers, serve the dashboard, handle startup/shutdown.
"""

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from dotsocr.auth.routes import router as auth_router
from dotsocr.config import SERVED_MODEL_NAME
from dotsocr.dashboard.routes import router as dashboard_router, live_metrics
from dotsocr.database import init_db
from dotsocr.inference.engine import initialize as init_engine
from dotsocr.inference.routes import router as inference_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("dotsocr")

STATIC_DIR = Path(__file__).parent / "dashboard" / "static"


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="DotsOCR Platform",
        description="Self-hosted OCR inference platform with API key management, "
                    "usage tracking, and a beautiful dashboard.",
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
    )

    # ── CORS ──────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────────
    app.include_router(auth_router)
    app.include_router(inference_router)
    app.include_router(dashboard_router)

    # ── Dashboard SPA ─────────────────────────────────────────────
    @app.get("/")
    async def serve_dashboard():
        return FileResponse(STATIC_DIR / "index.html")

    # Serve static assets if any are added later
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # ── Startup ───────────────────────────────────────────────────
    @app.on_event("startup")
    async def startup():
        logger.info("═" * 60)
        logger.info("  DotsOCR Platform starting up")
        logger.info("═" * 60)

        # Initialize database
        init_db()
        logger.info("Database initialized")

        # Initialize inference engine
        init_engine()
        logger.info(f"Model: {SERVED_MODEL_NAME}")
        # Start live metrics collector
        live_metrics.start_collector()
        logger.info("Live metrics collector started")
        logger.info("Platform ready — Dashboard at /")

    return app


# Default app instance
app = create_app()
