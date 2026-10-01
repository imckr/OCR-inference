"""
OCR inference engine — extracted from server_vllm.py.

Handles model download, loading (vLLM or transformers), image decoding,
PDF → image conversion, and inference dispatch.
"""

import base64
import io
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from dotsocr.inference.md_to_html import md_to_html, normalize_model_output

from dotsocr.config import (
    AUTO_PDF_PROMPT,
    BACKEND,
    GPU_MEMORY_UTILIZATION,
    MAX_IMAGE_DIMENSION,
    MAX_MODEL_LEN,
    MODEL_DIR,
    MODEL_NAME,
    PDF_RENDER_DPI,
    SERVED_MODEL_NAME,
    TASK_PROMPTS,
    DEFAULT_MAX_TOKENS,
)

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

logger = logging.getLogger("dotsocr.engine")

# ── Check vLLM availability ───────────────────────────────────────────
VLLM_AVAILABLE = False
if BACKEND != "transformers":
    try:
        from vllm import LLM, SamplingParams

        VLLM_AVAILABLE = True
        logger.info("vLLM is available")
    except ImportError:
        logger.info("vLLM not installed — will use transformers backend")


# ── Global state ──────────────────────────────────────────────────────
_engine = None  # vLLM LLM instance
_model = None  # transformers model
_processor = None  # transformers processor
_backend: Optional[str] = None  # "vllm" | "transformers"
_device: str = "cpu"
_dtype = torch.float32
_executor = ThreadPoolExecutor(max_workers=2)


def get_backend() -> Optional[str]:
    return _backend


def get_device() -> str:
    return _device


def get_executor() -> ThreadPoolExecutor:
    return _executor


# ══════════════════════════════════════════════════════════════════════
#  Model download
# ══════════════════════════════════════════════════════════════════════
def download_model():
    has_weights = any(MODEL_DIR.glob("*.safetensors")) or any(
        MODEL_DIR.glob("*.bin")
    )
    if MODEL_DIR.exists() and has_weights:
        logger.info(f"Model present at {MODEL_DIR}")
        return
    logger.info(f"Downloading {MODEL_NAME} → {MODEL_DIR} …")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_NAME,
        local_dir=str(MODEL_DIR),
        local_dir_use_symlinks=False,
    )
    logger.info("Download complete")


# ══════════════════════════════════════════════════════════════════════
#  Backend initialisation
# ══════════════════════════════════════════════════════════════════════
def detect_device():
    global _device, _dtype
    if torch.cuda.is_available():
        _device, _dtype = "cuda", torch.bfloat16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS has poor bfloat16 kernel support — float16 is hardware-accelerated
        _device, _dtype = "mps", torch.float16
    else:
        _device, _dtype = "cpu", torch.float32
    logger.info(f"Device: {_device}  dtype: {_dtype}")


def init_vllm() -> bool:
    global _engine, _processor, _backend
    if not VLLM_AVAILABLE:
        return False
    try:
        logger.info("Initialising vLLM engine …")
        dtype_str = "bfloat16" if torch.cuda.is_available() else "float16"
        _engine = LLM(
            model=str(MODEL_DIR),
            trust_remote_code=True,
            dtype=dtype_str,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            enforce_eager=True,
        )
        _processor = AutoProcessor.from_pretrained(
            str(MODEL_DIR), trust_remote_code=True
        )
        _backend = "vllm"
        logger.info("vLLM engine ready")
        return True
    except Exception as exc:
        logger.warning(f"vLLM init failed: {exc}")
        _engine = None
        return False


def init_transformers():
    global _model, _processor, _backend
    logger.info(f"Loading transformers model on {_device} ({_dtype}) …")
    try:
        _processor = AutoProcessor.from_pretrained(
            str(MODEL_DIR), trust_remote_code=True
        )
    except TypeError as exc:
        # Qwen2.5-VL based models (e.g. dots.ocr) fail because
        # DotsVLProcessor.__init__ doesn't pass video_processor to super().
        # Fix: build the processor manually from components.
        if "video_processor" in str(exc):
            logger.info("Patching processor for Qwen2.5-VL video_processor compatibility")
            from transformers import AutoTokenizer, AutoImageProcessor, Qwen2_5_VLProcessor, Qwen2VLVideoProcessor

            image_processor = AutoImageProcessor.from_pretrained(
                str(MODEL_DIR), trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(
                str(MODEL_DIR), trust_remote_code=True
            )
            video_processor = Qwen2VLVideoProcessor()
            _processor = Qwen2_5_VLProcessor(
                image_processor=image_processor,
                tokenizer=tokenizer,
                video_processor=video_processor,
            )
            # Set custom attributes that DotsVLProcessor would have added
            _processor.image_token = getattr(tokenizer, "image_token", "<|imgpad|>")
            _processor.image_token_id = getattr(tokenizer, "image_token_id", 151665)
            # Copy the chat template from tokenizer to processor so
            # apply_chat_template() works on the processor directly.
            # Prefer the multimodal chat_template.json (handles image content
            # lists) over the text-only template in tokenizer_config.json.
            import json as _json
            _chat_tpl_file = MODEL_DIR / "chat_template.json"
            if _chat_tpl_file.exists():
                with open(_chat_tpl_file) as _f:
                    _processor.chat_template = _json.load(_f).get("chat_template")
                logger.info("Loaded multimodal chat template from chat_template.json")
            elif hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
                _processor.chat_template = tokenizer.chat_template
            logger.info("Processor built manually with video_processor fallback")
        else:
            raise
    _model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        trust_remote_code=True,
        dtype=_dtype,
    )
    # Ensure *every* parameter is in _dtype (some custom code may skip conversion)
    _model = _model.to(dtype=_dtype, device=_device).eval()
    _backend = "transformers"
    logger.info("Transformers model loaded")


def initialize():
    """Full initialization sequence — call on startup."""
    detect_device()
    download_model()
    if BACKEND in ("auto", "vllm"):
        if init_vllm():
            logger.info(f"Backend: vLLM  |  Model: {SERVED_MODEL_NAME}")
            return
    init_transformers()
    logger.info(f"Backend: transformers  |  Model: {SERVED_MODEL_NAME}")


# ══════════════════════════════════════════════════════════════════════
#  Image utilities
# ══════════════════════════════════════════════════════════════════════
# PDF magic bytes: %PDF
_PDF_MAGIC = b"%PDF"


def _is_pdf_bytes(data: bytes) -> bool:
    """Check if raw bytes are a PDF document."""
    return data[:4] == _PDF_MAGIC


def decode_image_from_url(url: str) -> List[Image.Image]:
    """Decode PIL Image(s) from a data-URI, HTTP(S) URL, or raw base64.

    Returns a list of images. For normal images this is a single-element list.
    For PDFs, each page becomes a separate image.
    """
    if url.startswith("data:"):
        if "," not in url:
            raise ValueError("Malformed data URI — no comma separator found")
        header, payload = url.split(",", 1)
        payload = payload.strip().replace("\n", "").replace("\r", "").replace(" ", "")
        missing_pad = len(payload) % 4
        if missing_pad:
            payload += "=" * (4 - missing_pad)
        try:
            img_bytes = base64.b64decode(payload, validate=True)
        except Exception:
            img_bytes = base64.urlsafe_b64decode(payload)

        # Detect PDF by MIME type in header or by magic bytes
        is_pdf = "application/pdf" in header.lower() or _is_pdf_bytes(img_bytes)
        if is_pdf:
            logger.info("Detected PDF in data URI — converting pages to images")
            return pdf_to_images(img_bytes)

        return [Image.open(io.BytesIO(img_bytes)).convert("RGB")]

    if url.startswith(("http://", "https://")):
        import requests as _req

        resp = _req.get(url, timeout=30)
        resp.raise_for_status()
        raw = resp.content

        if _is_pdf_bytes(raw) or "application/pdf" in resp.headers.get("content-type", "").lower():
            logger.info("Detected PDF from URL — converting pages to images")
            return pdf_to_images(raw)

        return [Image.open(io.BytesIO(raw)).convert("RGB")]

    # Raw base64
    try:
        payload = url.strip().replace("\n", "").replace("\r", "").replace(" ", "")
        missing_pad = len(payload) % 4
        if missing_pad:
            payload += "=" * (4 - missing_pad)
        img_bytes = base64.b64decode(payload, validate=True)

        if _is_pdf_bytes(img_bytes):
            logger.info("Detected PDF in raw base64 — converting pages to images")
            return pdf_to_images(img_bytes)

        return [Image.open(io.BytesIO(img_bytes)).convert("RGB")]
    except Exception:
        pass

    raise ValueError(f"Unsupported image URL scheme: {url[:80]}")


def resize_for_inference(image: Image.Image, max_dim: int = MAX_IMAGE_DIMENSION) -> Image.Image:
    """Downscale image so its longest side ≤ max_dim. Returns a new image
    (or the same object if already small enough). Uses LANCZOS for quality."""
    w, h = image.size
    if w <= max_dim and h <= max_dim:
        return image
    scale = max_dim / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    logger.info(f"Resizing image {w}x{h} → {new_w}x{new_h} (max_dim={max_dim})")
    return image.resize((new_w, new_h), Image.LANCZOS)


def image_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode()
    buf.close()
    return b64


def pdf_to_images(pdf_bytes: bytes, dpi: int = PDF_RENDER_DPI) -> List[Image.Image]:
    """Render PDF pages to PIL images at the configured DPI, auto-resized."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images: List[Image.Image] = []
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        # Immediately resize to prevent huge images from eating RAM
        img = resize_for_inference(img)
        images.append(img)
    doc.close()
    return images


# ══════════════════════════════════════════════════════════════════════
#  Message parsing
# ══════════════════════════════════════════════════════════════════════
def parse_messages(messages) -> tuple[List[Dict], List[Image.Image]]:
    """Convert ChatMessage list → plain dicts + extracted PIL images.

    PDFs are automatically detected and expanded: each page becomes a
    separate image entry in the content parts.
    """
    images: List[Image.Image] = []
    dicts: List[Dict] = []

    for msg in messages:
        content = msg.content if hasattr(msg, "content") else msg.get("content")
        role = msg.role if hasattr(msg, "role") else msg.get("role")

        if isinstance(content, str):
            dicts.append({"role": role, "content": content})
        else:
            parts: List[Dict] = []
            for p in content:
                p_type = p.type if hasattr(p, "type") else p.get("type")
                if p_type == "text":
                    text = p.text if hasattr(p, "text") else p.get("text")
                    if text:
                        parts.append({"type": "text", "text": text})
                elif p_type == "image_url":
                    img_url = p.image_url if hasattr(p, "image_url") else p.get("image_url")
                    url = img_url.url if hasattr(img_url, "url") else img_url.get("url")
                    decoded_images = decode_image_from_url(url)
                    for img in decoded_images:
                        # Resize before anything else to cap memory
                        img = resize_for_inference(img)
                        images.append(img)
                        b64 = image_to_base64(img)
                        parts.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64}"},
                            }
                        )
            dicts.append({"role": role, "content": parts})

    return dicts, images


# ══════════════════════════════════════════════════════════════════════
#  Inference backends
# ══════════════════════════════════════════════════════════════════════
def _infer_vllm(
    messages_dicts: List[Dict],
    max_tokens: int,
    temperature: float,
    top_p: float,
    stop: Optional[List[str]] = None,
) -> tuple[str, int, int]:
    params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop or [],
    )
    outputs = _engine.chat(messages=messages_dicts, sampling_params=params)
    out = outputs[0]
    text = normalize_model_output(out.outputs[0].text)
    prompt_tokens = len(out.prompt_token_ids)
    completion_tokens = len(out.outputs[0].token_ids)
    return text, prompt_tokens, completion_tokens


def _infer_transformers(
    messages_dicts: List[Dict],
    images: List[Image.Image],
    max_tokens: int,
) -> tuple[str, int, int]:
    import gc
    import signal

    proc_messages: List[Dict] = []
    img_idx = 0
    for msg in messages_dicts:
        if isinstance(msg["content"], str):
            proc_messages.append(msg)
        else:
            content = []
            for part in msg["content"]:
                if part["type"] == "text":
                    content.append({"type": "text", "text": part["text"]})
                elif part["type"] == "image_url":
                    content.append({"type": "image", "image": images[img_idx]})
                    img_idx += 1
            proc_messages.append({"role": msg["role"], "content": content})

    t_prep = time.time()
    inputs = _processor.apply_chat_template(
        proc_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(_device)

    # Ensure all float tensors match the model dtype (e.g. bfloat16 → float16)
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point() and v.dtype != _dtype:
            inputs[k] = v.to(_dtype)

    prompt_tokens = inputs["input_ids"].shape[-1]
    prep_time = time.time() - t_prep
    logger.info(
        f"Prompt tokens: {prompt_tokens} | Preprocessing: {prep_time:.1f}s | "
        f"Max new tokens: {max_tokens}"
    )

    t_gen = time.time()
    with torch.inference_mode():
        output_ids = _model.generate(**inputs, max_new_tokens=max_tokens)
    gen_time = time.time() - t_gen

    completion_tokens = output_ids.shape[-1] - prompt_tokens
    tokens_per_sec = completion_tokens / gen_time if gen_time > 0 else 0
    logger.info(
        f"Generation: {gen_time:.1f}s | {completion_tokens} tokens | "
        f"{tokens_per_sec:.1f} tok/s"
    )

    # Decode only the newly generated tokens (skip the echoed prompt)
    generated_ids = output_ids[:, prompt_tokens:]
    result = normalize_model_output(
        _processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    )

    # Fallback: if generated-only decode is empty or suspiciously short,
    # decode the full output and strip the prompt/chat markers
    if not result:
        full_decode = _processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        result = full_decode
        for sep in ("assistant\n", "Assistant\n", "Assistant:"):
            if sep in result:
                result = result.split(sep)[-1].strip()
        logger.debug(f"Used full-decode fallback ({len(result)} chars)")

    # ── Free GPU/MPS memory immediately ──
    del inputs, output_ids, generated_ids
    if _device == "mps":
        torch.mps.empty_cache()
    elif _device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return result, prompt_tokens, completion_tokens


def run_inference(
    messages_dicts: List[Dict],
    images: List[Image.Image],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.0,
    top_p: float = 1.0,
    stop: Optional[List[str]] = None,
) -> tuple[str, int, int]:
    """Dispatch to the active backend. Returns (text, prompt_tokens, completion_tokens).

    When images are present, routes through PaddleOCRVL for structured
    document parsing (LaTeX, tables, layout-aware markdown).
    """
    if images:
        parts = []
        for img in images:
            md_text, _ = process_document_image(img, max_tokens)
            parts.append(md_text)
        combined = "\n\n".join(parts)
        return combined, 0, len(combined.split())

    if _backend == "vllm":
        return _infer_vllm(messages_dicts, max_tokens, temperature, top_p, stop)
    return _infer_transformers(messages_dicts, images, max_tokens)


# ── PaddleOCRVL pipeline singleton ─────────────────────────────────────
_vl_pipeline = None


def _get_vl_pipeline():
    """Lazy-initialize the official PaddleOCRVL pipeline."""
    global _vl_pipeline
    if _vl_pipeline is not None:
        return _vl_pipeline

    import os
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    import paddle
    from paddleocr import PaddleOCRVL

    # Auto-detect: gpu:0 when CUDA is compiled in and a GPU is present, else cpu
    if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
        device = "gpu:0"
    else:
        device = "cpu"

    logger.info("Initializing PaddleOCRVL pipeline (device=%s) …", device)
    _vl_pipeline = PaddleOCRVL(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        device=device,
    )
    logger.info("PaddleOCRVL pipeline ready")
    return _vl_pipeline


def process_document_image(
    image: Image.Image, max_tokens: int
) -> tuple[str, float]:
    """Parse a document image using the official PaddleOCRVL pipeline.

    Returns (markdown_text, elapsed_seconds).
    """
    import tempfile
    t0 = time.time()

    pipeline = _get_vl_pipeline()

    # PaddleOCRVL.predict() accepts file paths or numpy arrays.
    # Save the PIL image to a temp file for compatibility.
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.save(tmp, format="PNG")
        tmp_path = tmp.name

    try:
        results = pipeline.predict(
            tmp_path,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            max_new_tokens=max_tokens,
        )
    finally:
        os.remove(tmp_path)

    # Extract markdown text from the result
    markdown_parts = []
    for res in results:
        md_info = res.markdown
        md_text = md_info.get("markdown_texts", "")
        if md_text:
            markdown_parts.append(md_text)

    markdown = "\n\n".join(markdown_parts)
    elapsed = time.time() - t0
    logger.info(f"PaddleOCRVL: processed in {elapsed:.1f}s, {len(markdown)} chars")
    return markdown, elapsed


def process_single_image(
    image: Image.Image, task: str, max_tokens: int
) -> tuple[str, float]:
    """Run OCR on a single image via PaddleOCRVL.

    All task types are routed through the PaddleOCRVL pipeline which
    auto-detects layout regions (text, tables, formulas, charts) in a
    single pass. The `task` parameter is kept for API compatibility but
    does not change inference behaviour.
    """
    return process_document_image(image, max_tokens)


def process_pdf_to_html(
    pdf_bytes: bytes, max_tokens: int = DEFAULT_MAX_TOKENS
) -> tuple[str, List[Dict], float]:
    """Process a full PDF → structured HTML output via PaddleOCRVL.

    Returns (html_string, pages_list, total_elapsed).
    """
    import tempfile

    t0 = time.time()
    pipeline = _get_vl_pipeline()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        results = pipeline.predict(
            tmp_path,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            max_new_tokens=max_tokens,
        )
    finally:
        os.remove(tmp_path)

    pages: List[Dict] = []
    total_elapsed = time.time() - t0

    for i, res in enumerate(results):
        md_info = res.markdown
        md_text = md_info.get("markdown_texts", "")
        pages.append({
            "page": i + 1,
            "content": md_text,
            "elapsed_seconds": round(total_elapsed / max(len(results), 1), 1),
        })

    # Build combined HTML document
    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head><meta charset=\"UTF-8\"><style>",
        "body { font-family: system-ui, -apple-system, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; color: #1a1a1a; }",
        ".page-break { border-top: 2px dashed #ccc; margin: 32px 0; padding-top: 8px; color: #888; font-size: 13px; }",
        "table { border-collapse: collapse; width: 100%; margin: 16px 0; }",
        "th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: left; }",
        "th { background: #f5f5f5; font-weight: 600; }",
        "pre, code { background: #f5f5f5; padding: 2px 6px; border-radius: 4px; font-size: 14px; }",
        "pre { padding: 12px; overflow-x: auto; }",
        "img { max-width: 100%; }",
        "</style></head>",
        "<body>",
    ]

    for p in pages:
        page_num = p["page"]
        if page_num > 1:
            html_parts.append(f"<div class='page-break'>Page {page_num}</div>")
        # Convert markdown model output → HTML
        content_html = md_to_html(p["content"])
        html_parts.append(f"<section data-page='{page_num}'>{content_html}</section>")

    html_parts.append("</body></html>")
    html_output = "\n".join(html_parts)

    return html_output, pages, total_elapsed


def get_model_info() -> Dict[str, Any]:
    """Return model metadata."""
    info: Dict[str, Any] = {
        "model_name": MODEL_NAME,
        "served_as": SERVED_MODEL_NAME,
        "backend": _backend,
        "device": _device,
        "dtype": str(_dtype),
        "torch_version": torch.__version__,
        "tasks": list(TASK_PROMPTS.keys()),
    }
    if _model is not None:
        info["model_params"] = sum(p.numel() for p in _model.parameters())
    return info
