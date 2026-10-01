# Dots.OCR Local Server - Setup Guide

## What's Been Created

Your project is now complete with a full-featured local OCR server. Here's what you have:

### Core Files
- **server.py** - FastAPI server with model loading and inference endpoints
- **client.py** - Python client for testing the server
- **requirements.txt** - Python dependencies
- **Dockerfile** - Docker container configuration
- **docker-compose.yml** - Multi-container orchestration

### Documentation
- **README.md** - Comprehensive documentation
- **QUICKSTART.md** - Quick 3-step setup guide
- **SETUP_GUIDE.md** - This file

### Startup Scripts
- **install.sh** - One-time setup script
- **start.sh** - Server startup script
- **diagnostics.py** - System diagnostics tool

### Configuration
- **.env.example** - Environment template
- **.gitignore** - Git ignore patterns
- **requirements-dev.txt** - Optional dev tools

## Installation Steps

### Option A: Using Bash Scripts (Recommended for macOS)

```bash
cd /Users/chandrakrajwal/Documents/inferiaAI/DotsOCR

# Step 1: Run setup
bash install.sh

# Step 2: Start the server
bash start.sh
```

### Option B: Manual Setup

```bash
cd /Users/chandrakrajwal/Documents/inferiaAI/DotsOCR

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# (Optional) For GPU support:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Start the server
python server.py
```

### Option C: Docker

```bash
cd /Users/chandrakrajwal/Documents/inferiaAI/DotsOCR

# Build and run
docker-compose up --build

# Or manually:
docker build -t dots-ocr .
docker run -p 8000:8000 dots-ocr
```

## First Run

When you start the server for the first time:
1. The model (~20GB) will be downloaded from HuggingFace
2. It will be cached in `./models/dots_ocr/`
3. Subsequent runs will start instantly

**Time estimates:**
- First run: 5-15 minutes (download + load)
- Subsequent runs: 30-60 seconds (model loading only)

## Testing the Server

### Health Check
```bash
curl http://localhost:8000/health
```

### Access Interactive Docs
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Run Test Client
```bash
# New terminal, same directory
python client.py
```

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Server health check |
| `/model/info` | GET | Model information |
| `/ocr` | POST | Process image for OCR |
| `/ocr/text` | POST | Process text through model |
| `/docs` | GET | Swagger UI documentation |
| `/redoc` | GET | ReDoc documentation |

## Example API Calls

### Health Check
```bash
curl http://localhost:8000/health
```

### Get Model Info
```bash
curl http://localhost:8000/model/info
```

### Process Text (JSON)
```bash
curl -X POST http://localhost:8000/ocr/text \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here"}'
```

### Process Image (Form Data)
```bash
curl -X POST -F "file=@image.png" http://localhost:8000/ocr
```

### Using Python
```python
import requests

# Text processing
response = requests.post('http://localhost:8000/ocr/text',
    json={"text": "Your text here"})
print(response.json())

# Image processing
with open('image.png', 'rb') as f:
    response = requests.post('http://localhost:8000/ocr',
        files={'file': f})
    print(response.json())
```

## Project Structure

```
DotsOCR/
├── server.py              # Main FastAPI server
├── client.py              # Test client
├── diagnostics.py         # System diagnostics
├── install.sh             # Setup script
├── start.sh               # Startup script
├── requirements.txt       # Dependencies
├── requirements-dev.txt   # Dev dependencies (optional)
├── Dockerfile             # Docker config
├── docker-compose.yml     # Docker Compose config
├── .env.example          # Environment template
├── .gitignore            # Git ignore patterns
├── README.md             # Full documentation
├── QUICKSTART.md         # Quick start guide
├── SETUP_GUIDE.md        # This file
└── models/               # Downloaded models (auto-created)
    └── dots_ocr/         # Model cache directory
```

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

Edit `.env`:
```env
# Server
SERVER_HOST=0.0.0.0
SERVER_PORT=8000

# Model
MODEL_DEVICE=cuda  # or 'cpu'
MODEL_PRECISION=float16  # or 'float32'

# HuggingFace (optional)
HF_TOKEN=your_token_here
```

### HuggingFace Token (Optional)

For faster downloads or if the model requires authentication:

```bash
huggingface-cli login
# Then enter your token from https://huggingface.co/settings/tokens
```

Or set in `.env`:
```env
HF_TOKEN=hf_your_token_here
```

## Diagnostics

Check if everything is set up correctly:

```bash
python diagnostics.py
```

This will verify:
- Python environment
- Required packages
- GPU/CUDA availability
- Disk space
- Network connectivity
- Project files
- Import compatibility

## Troubleshooting

### Server Won't Start

**Error: "Address already in use"**
```bash
# Change port in start.sh or .env
# Or kill existing process:
lsof -ti:8000 | xargs kill -9
```

**Error: "Module not found"**
```bash
# Activate virtual environment:
source venv/bin/activate

# Install dependencies:
pip install -r requirements.txt
```

### Model Download Issues

**Error: "Not enough disk space"**
- Need 20GB+ free space
- Check with: `df -h`

**Error: "Connection timeout"**
- Check internet connection
- Try again later (HuggingFace servers may be busy)
- Or download manually and place in `./models/dots_ocr/`

### Out of Memory

**Error: "CUDA out of memory"**
```bash
# Use CPU mode in .env or server.py:
MODEL_DEVICE=cpu
```

**Error: "OOM Killed"**
- System is out of memory
- Close other applications
- Use a machine with 16GB+ RAM

### Slow Inference

**Tips for speed:**
1. Use GPU (CUDA) - 10-100x faster
2. Use float16 precision (faster, less memory)
3. Reduce max_length parameter
4. Use batch processing for multiple inputs

## Performance Tips

1. **GPU Setup**: Ensure CUDA is properly installed
2. **Model Precision**: Use float16 for faster inference
3. **Batch Processing**: Group multiple requests
4. **Caching**: Model cached after first download
5. **Monitoring**: Track performance with diagnostics.py

## Deployment Options

### Local Development
```bash
bash start.sh
```

### Production with Gunicorn
```bash
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker server:app
```

### Docker Container
```bash
docker-compose up
```

### Cloud Deployment
- AWS: EC2, ECS, or SageMaker
- GCP: Compute Engine or Cloud Run
- Azure: App Service or Container Instances

## Next Steps

1. Run `bash install.sh`
2. Run `bash start.sh`
3. Visit http://localhost:8000/docs
4. Test an endpoint
5. Integrate with your application

## Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Transformers Library](https://huggingface.co/docs/transformers/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [HuggingFace Hub](https://huggingface.co/)
- [Model Repository](https://huggingface.co/rednote-hilab/dots.ocr)

## Support

For issues or questions:
1. Check the README.md for detailed documentation
2. Run `python diagnostics.py` to check your setup
3. Review the model documentation on HuggingFace
4. Check server logs for error messages

## Summary

You now have a complete local OCR server that:
- Downloads and caches the dots.ocr model
- Provides REST API endpoints for inference
- Works offline (no HuggingFace API dependency)
- Supports both images and text
- Includes interactive API documentation
- Can be deployed with Docker
- Scales with multiple workers

Happy OCR-ing!
