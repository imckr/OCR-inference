"""
DotsOCR platform configuration.

All configuration is driven by environment variables with sensible defaults.
"""

import os
import secrets
from pathlib import Path

from dotenv import load_dotenv

# Load .env file so env vars are available via os.getenv()
load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(os.getenv("DOTSOCR_BASE_DIR", "."))
# MODEL_CACHE_DIR is the *parent* directory for all downloaded models.
# Each model gets its own subdirectory derived from MODEL_NAME.
_model_cache = Path(os.getenv("MODEL_CACHE_DIR", str(BASE_DIR / "models")))
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'dotsocr.db'}")
LOG_DIR = Path(os.getenv("LOG_DIR", str(BASE_DIR / "logs")))

# ── Server ─────────────────────────────────────────────────────────────
HOST = os.getenv("HOST", os.getenv("SERVER_HOST", "0.0.0.0"))
PORT = int(os.getenv("PORT", os.getenv("SERVER_PORT", "8009")))

# ── Model ──────────────────────────────────────────────────────────────
MODEL_NAME = os.getenv("MODEL_NAME", "PaddlePaddle/PaddleOCR-VL")
# Derive a short display name from MODEL_NAME (e.g. "rednote-hilab/dots.ocr" → "dots.ocr")
_default_served = MODEL_NAME.split("/")[-1] if "/" in MODEL_NAME else MODEL_NAME
SERVED_MODEL_NAME = os.getenv("SERVED_MODEL_NAME", _default_served)

# MODEL_DIR is where the actual weights live.
# Explicit MODEL_DIR env var takes priority (backward-compatible).
# Otherwise: <cache_dir>/<model_short_name>  e.g. ./models/dots.ocr
if os.getenv("MODEL_DIR"):
    MODEL_DIR = Path(os.getenv("MODEL_DIR"))
else:
    MODEL_DIR = _model_cache / _default_served
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "8192"))
GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.9"))
BACKEND = os.getenv("BACKEND", "auto")  # "auto", "vllm", "transformers"

# ── Auth / Security ───────────────────────────────────────────────────
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_urlsafe(64))
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))  # 24h
API_KEY_PREFIX = "dots_"

# ── Rate Limiting ─────────────────────────────────────────────────────
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
RATE_LIMIT_PER_DAY = int(os.getenv("RATE_LIMIT_PER_DAY", "1000"))

# ── Free Tier Defaults ────────────────────────────────────────────────
FREE_TIER_REQUESTS_PER_DAY = int(os.getenv("FREE_TIER_REQUESTS_PER_DAY", "100"))
FREE_TIER_REQUESTS_PER_MINUTE = int(os.getenv("FREE_TIER_REQUESTS_PER_MINUTE", "10"))

# ── Task Prompts ──────────────────────────────────────────────────────
TASK_PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
}
VALID_TASKS = list(TASK_PROMPTS.keys()) + ["auto", "doc_parser"]
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "1024"))

# ── Image Processing ──────────────────────────────────────────────────
# Maximum dimension (width or height) for images sent to the model.
# Larger images are downscaled proportionally to prevent OOM crashes.
MAX_IMAGE_DIMENSION = int(os.getenv("MAX_IMAGE_DIMENSION", "1024"))
# DPI for rendering PDF pages to images (lower = smaller images = less RAM)
PDF_RENDER_DPI = int(os.getenv("PDF_RENDER_DPI", "144"))

# ── Auto PDF prompt (comprehensive content detection → HTML) ────────
AUTO_PDF_PROMPT = "OCR:"
