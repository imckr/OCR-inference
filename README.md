# Dots.OCR Local Server

A local deployment of the [dots.ocr](https://huggingface.co/rednote-hilab/dots.ocr) model from HuggingFace, allowing you to run OCR locally without depending on external APIs.

## Overview

This project sets up a FastAPI-based server that downloads and runs the dots.ocr model locally. The model is automatically downloaded on first startup and cached for subsequent runs.

## Prerequisites

- Python 3.8+
- pip (Python package manager)
- At least 20GB of free disk space (for model storage)
- GPU recommended (14GB+ VRAM) for better performance. CPU mode is supported but slower.

## Installation

### 1. Clone or Setup the Project

```bash
cd /Users/chandrakrajwal/Documents/inferiaAI/DotsOCR
```

### 2. Create a Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: If you're using GPU (CUDA), you may want to install the GPU-enabled version of PyTorch:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. Setup HuggingFace Authentication (Optional but Recommended)

If the model requires authentication or if you want to avoid rate limiting:

```bash
huggingface-cli login
# Enter your HuggingFace token when prompted
```

Get your token from: https://huggingface.co/settings/tokens

## Running the Server

### Start the Server

```bash
python server.py
```

The server will:
1. Download the dots.ocr model to `./models/dots_ocr/` (first time only)
2. Load the model into memory
3. Start listening on `http://localhost:8000`

Expected output:
```
INFO:     Started server process [XXXXX]
INFO:     Waiting for application startup.
INFO:     Application startup complete
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### Access the API

- **Health Check**: `GET http://localhost:8000/health`
- **Model Info**: `GET http://localhost:8000/model/info`
- **OCR on Image**: `POST http://localhost:8000/ocr` (multipart/form-data)
- **Text Processing**: `POST http://localhost:8000/ocr/text` (JSON)

### Interactive API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### 1. Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

### 2. Model Information

```bash
curl http://localhost:8000/model/info
```

Response:
```json
{
  "model_name": "rednote-hilab/dots.ocr",
  "model_dir": "./models/dots_ocr",
  "device": "cuda",
  "torch_version": "2.1.1",
  "model_params": 1234567890
}
```

### 3. Process Image

```bash
curl -X POST -F "file=@/path/to/image.png" http://localhost:8000/ocr
```

### 4. Process Text

```bash
curl -X POST http://localhost:8000/ocr/text \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here"}'
```

## Using the Test Client

Run the included client script to test the server:

```bash
python client.py
```

To process a specific image:
```bash
python client.py /path/to/image.png
```

## Project Structure

```
DotsOCR/
├── server.py              # Main FastAPI server
├── client.py              # Test client script
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── models/
    └── dots_ocr/          # Downloaded model (auto-created)
        ├── config.json
        ├── model.safetensors
        └── ...
```

## Configuration

### Environment Variables (Optional)

Create a `.env` file to customize:

```env
# Server configuration
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
LOG_LEVEL=INFO

# Model configuration
MODEL_DEVICE=cuda  # or 'cpu'
MODEL_PRECISION=float16  # or 'float32'

# HuggingFace
HF_TOKEN=your_token_here
```

Update `server.py` to load these variables if needed.

### Changing Model

To use a different model, edit the `MODEL_NAME` variable in `server.py`:

```python
MODEL_NAME = "rednote-hilab/dots.ocr"  # Change this
```

## Troubleshooting

### Error: "CUDA out of memory"
- Use CPU mode: Edit `server.py` and change `DEVICE` 
- Or reduce batch size / model precision
- Reduce `max_length` in generation parameters

### Error: "Model not found on HuggingFace"
- Verify the model name is correct
- Check your HuggingFace credentials
- Ensure you have internet connection during first run

### Server takes too long to start
- Model download is in progress (normal for first run)
- Check disk space (need 20GB+)
- GPU initialization can take time

### API returns 503 Service Unavailable
- Model is still loading, wait a few seconds
- Check server logs for errors

## Performance Tips

1. **First Run**: Download and loading will take 5-15 minutes depending on internet and storage speed
2. **GPU Usage**: Significantly faster - 10-100x speedup for inference
3. **Caching**: Model is cached locally after first download
4. **Batch Processing**: Consider batching requests for better throughput

## Docker Deployment (Optional)

Create a `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .

EXPOSE 8000

CMD ["python", "server.py"]
```

Build and run:
```bash
docker build -t dots-ocr-server .
docker run -p 8000:8000 --gpus all dots-ocr-server  # Add --gpus for GPU support
```

## Integration Example

```python
import requests

# Start the server first: python server.py

# Health check
response = requests.get('http://localhost:8000/health')
print(response.json())

# Text processing
response = requests.post('http://localhost:8000/ocr/text', 
    json={"text": "Your text here"})
print(response.json())
```

## Deployment Options

### Option 1: Local Development
Perfect for testing and development:
```bash
python server.py
```

### Option 2: Production with Gunicorn
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 server:app
```

### Option 3: Docker
See Docker Deployment section above

### Option 4: Cloud Deployment
- **AWS**: EC2 instance or ECS
- **GCP**: Compute Engine or Cloud Run
- **Azure**: Virtual Machine or Container Instances
- **Heroku**: Not recommended (model too large for free tier)

## Model Details

- **Model**: rednote-hilab/dots.ocr
- **Type**: Vision-Language Model
- **Repository**: https://huggingface.co/rednote-hilab/dots.ocr
- **License**: Check the repository for license details

## Support and Resources

- Model Documentation: https://huggingface.co/rednote-hilab/dots.ocr
- FastAPI Docs: https://fastapi.tiangolo.com/
- Transformers Library: https://huggingface.co/docs/transformers/

## License

Refer to the original dots.ocr model license on HuggingFace.

## Notes

- This is a local deployment template. Adjust the model preprocessing and inference logic based on the actual model's requirements.
- The `/ocr` endpoint is a skeleton implementation and needs to be customized based on how the dots.ocr model expects image inputs.
- For production use, add authentication, rate limiting, and monitoring.
