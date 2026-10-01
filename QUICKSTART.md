# Quick Start Guide for Dots.OCR Server

## Get Started in 3 Steps

### Step 1: Setup (One-time)

```bash
cd /Users/chandrakrajwal/Documents/inferiaAI/DotsOCR
bash install.sh
```

This will:
- Create a Python virtual environment
- Install all dependencies
- Detect your GPU (if available)
- Create necessary directories

### Step 2: Start the Server

```bash
bash start.sh
```

First run will download the model (~20GB). Subsequent runs will start instantly.

**Expected output:**
```
Started server process [XXXXX]
Waiting for application startup.
Application startup complete
Uvicorn running on http://0.0.0.0:8000
```

### Step 3: Test the Server

In a new terminal:

```bash
# Check if server is healthy
curl http://localhost:8000/health

# Get model info
curl http://localhost:8000/model/info

# Or use the test client
python client.py
```

## Access the API

- **Swagger UI**: http://localhost:8000/docs ← Start here!
- **ReDoc**: http://localhost:8000/redoc

## Common Commands

```bash
# Run diagnostics
python diagnostics.py

# Use the Python client
python client.py

# Process an image
python client.py /path/to/image.png

# Stop the server (Ctrl+C in the terminal)
```

## Docker Alternative

```bash
# Build
docker build -t dots-ocr .

# Run
docker run -p 8000:8000 dots-ocr

# Or with Docker Compose
docker-compose up
```

## Next Steps

- Read [README.md](README.md) for detailed documentation
- Check [server.py](server.py) to customize the model or API
- Add your HuggingFace token: `huggingface-cli login`

## Troubleshooting

**Q: Server takes too long to start?**
- A: First run downloads 20GB+ model. Subsequent runs are fast.

**Q: Out of memory error?**
- A: Use CPU mode. Edit `start.sh` and set MODEL_DEVICE=cpu (slower)

**Q: Can't reach HuggingFace?**
- A: Check internet connection or run `diagnostics.py`

For more help, see [README.md](README.md)
