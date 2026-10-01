# ── DotsOCR Platform ────────────────────────────────────────────────
# GPU-ready OCR inference platform with dashboard, auth & API keys.
#
# Build:
#   docker build -t dotsocr .
#
# Run:
#   docker run --gpus all -p 8009:8009 -v ./models:/app/models --env-file .env dotsocr
#
# CUDA 12.6 base — PaddlePaddle-GPU + PyTorch CUDA are installed on
# top of the system CUDA runtime.  The NVIDIA Container Toolkit must
# be installed on the host (provides the GPU driver to the container).
# ────────────────────────────────────────────────────────────────────

FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

# ── Python 3.12 via deadsnakes PPA + system deps ──────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
        build-essential curl libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python  python  /usr/bin/python3.12 1

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

WORKDIR /app

# ── Python deps (layer-cached until requirements.txt changes) ──────
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir hf_transfer

# ── GPU-accelerated overrides ──────────────────────────────────────
# PyTorch with CUDA 12.6 support (replaces CPU-only torch from requirements)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Ensure PyTorch's bundled NCCL is used instead of the older system copy
# (avoids "undefined symbol: ncclCommShrink" from version mismatch)
RUN apt-get purge -y --auto-remove libnccl2 libnccl-dev 2>/dev/null; \
    rm -f /usr/lib/x86_64-linux-gnu/libnccl* 2>/dev/null; true
ENV LD_LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/torch/lib:${LD_LIBRARY_PATH}"

# PaddlePaddle GPU (replaces CPU-only paddlepaddle from requirements)
# Uses the official PaddlePaddle index for CUDA 12.6 wheels.
RUN pip uninstall -y paddlepaddle && \
    pip install --no-cache-dir paddlepaddle-gpu \
        -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# ── Application ────────────────────────────────────────────────────
COPY pyproject.toml ./
COPY dotsocr/ ./dotsocr/
COPY .env* ./

# Register entry points (ocr CLI) — deps already satisfied above
RUN pip install --no-cache-dir --no-deps -e .

RUN mkdir -p /app/models /app/logs /app/data

EXPOSE 8009

HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD curl -f http://localhost:8009/health || exit 1

CMD ["ocr", "run"]
