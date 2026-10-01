#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  DotsOCR Platform — Start Script
#
#  Usage:
#    bash start_platform.sh              # Start with defaults (port 8009)
#    PORT=8080 bash start_platform.sh    # Custom port
# ═══════════════════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PORT="${PORT:-8009}"
HOST="${HOST:-0.0.0.0}"

echo ""
echo "  ╔═════════════════════════════════════════════════╗"
echo "  ║               OCR Inference v1.0                ║"
echo "  ║                                                 ║"
echo "  ║   Dashboard:  http://localhost:$PORT            ║"
echo "  ║   API Docs:   http://localhost:$PORT/api/docs.  ║"
echo "  ╚═════════════════════════════════════════════════╝"
echo ""

# Activate virtualenv if it exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Install package if not already installed
if ! python -c "import dotsocr" 2>/dev/null; then
    echo "Installing DotsOCR package..."
    pip install -e ".[dev]" --quiet
fi

# Initialize database
python -c "from dotsocr.database import init_db; init_db()" 2>/dev/null || true

# Start the platform
exec python -m dotsocr run --host "$HOST" --port "$PORT"
