"""
Inference API routes — OpenAI-compatible chat completions + file upload.

All inference endpoints require a valid API key.
"""

import asyncio
import io
import json
import logging
import time
import uuid
from typing import AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
from sqlalchemy.orm import Session

from dotsocr.auth.dependencies import validate_api_key
from dotsocr.config import DEFAULT_MAX_TOKENS, SERVED_MODEL_NAME, TASK_PROMPTS, VALID_TASKS
from dotsocr.database import APIKey, User, get_db
from dotsocr.inference.engine import (
    get_backend,
    get_device,
    get_executor,
    get_model_info,
    parse_messages,
    pdf_to_images,
    process_pdf_to_html,
    process_single_image,
    run_inference,
)
from dotsocr.inference.md_to_html import md_to_html
from dotsocr.inference.schemas import ChatCompletionRequest
from dotsocr.middleware.rate_limit import rate_limiter
from dotsocr.middleware.usage import record_usage
from dotsocr.dashboard.routes import live_metrics

logger = logging.getLogger("dotsocr.api")

router = APIRouter(tags=["inference"])


# ── Helpers ───────────────────────────────────────────────────────────
def _make_completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def _build_response(
    text: str, prompt_tokens: int, completion_tokens: int, model: str
) -> Dict:
    return {
        "id": _make_completion_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


async def _stream_response(
    text: str, prompt_tokens: int, completion_tokens: int, model: str
) -> AsyncGenerator[str, None]:
    cid = _make_completion_id()
    created = int(time.time())

    chunk = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
        ],
    }
    yield f"data: {json.dumps(chunk)}\n\n"

    chunk_size = 40
    for i in range(0, len(text), chunk_size):
        piece = text[i : i + chunk_size]
        chunk = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {"index": 0, "delta": {"content": piece}, "finish_reason": None}
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0)

    chunk = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"


# ══════════════════════════════════════════════════════════════════════
#  POST /v1/chat/completions
# ══════════════════════════════════════════════════════════════════════
@router.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    req: Request = None,
    auth: tuple[User, APIKey] = Depends(validate_api_key),
    db: Session = Depends(get_db),
):
    """OpenAI-compatible chat completions with vision support."""
    user, api_key = auth
    backend = get_backend()

    if backend is None:
        raise HTTPException(503, "Model not loaded")

    # Rate limit check
    rate_limiter.check(api_key)

    req_id = uuid.uuid4().hex[:8]
    t_start = time.time()
    logger.info(
        f"[{req_id}] POST /v1/chat/completions  user={user.email}  "
        f"key={api_key.key_prefix}...  model={request.model}  "
        f"backend={backend}  device={get_device()}"
    )

    try:
        messages_dicts, images = parse_messages(request.messages)
        logger.info(f"[{req_id}] Parsed {len(images)} image(s), {len(messages_dicts)} message(s)")
    except Exception as exc:
        logger.error(f"[{req_id}] Failed to parse messages: {exc}", exc_info=True)
        record_usage(
            db, user.id, api_key.id, "/v1/chat/completions",
            status_code=400, error_message=str(exc),
            ip_address=req.client.host if req else None,
        )
        raise HTTPException(400, f"Failed to parse messages: {exc}")

    stop = (
        [request.stop] if isinstance(request.stop, str) else (request.stop or None)
    )

    try:
        loop = asyncio.get_event_loop()
        text, p_tok, c_tok = await loop.run_in_executor(
            get_executor(),
            run_inference,
            messages_dicts,
            images,
            request.max_tokens or DEFAULT_MAX_TOKENS,
            request.temperature or 0.0,
            request.top_p or 1.0,
            stop,
        )
    except Exception as exc:
        elapsed = time.time() - t_start
        logger.error(f"[{req_id}] Inference error after {elapsed:.1f}s: {exc}", exc_info=True)
        record_usage(
            db, user.id, api_key.id, "/v1/chat/completions",
            elapsed_seconds=elapsed, status_code=500,
            error_message=str(exc),
            ip_address=req.client.host if req else None,
        )
        raise HTTPException(500, f"Inference error: {exc}")

    elapsed = time.time() - t_start
    logger.info(f"[{req_id}] Done in {elapsed:.1f}s  tokens={p_tok}p+{c_tok}c")

    # Record usage
    record_usage(
        db, user.id, api_key.id, "/v1/chat/completions",
        prompt_tokens=p_tok, completion_tokens=c_tok,
        elapsed_seconds=elapsed,
        ip_address=req.client.host if req else None,
    )

    # Live metrics
    live_metrics.record_request(elapsed, is_error=False)

    # Convert markdown → HTML if requested
    output_text = text
    if request.response_format and request.response_format.type == "html":
        output_text = md_to_html(text)

    model_name = request.model or SERVED_MODEL_NAME

    if request.stream:
        return StreamingResponse(
            _stream_response(output_text, p_tok, c_tok, model_name),
            media_type="text/event-stream",
        )

    return JSONResponse(_build_response(output_text, p_tok, c_tok, model_name))


# ══════════════════════════════════════════════════════════════════════
#  GET /v1/models
# ══════════════════════════════════════════════════════════════════════
@router.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": SERVED_MODEL_NAME,
                "object": "model",
                "created": 1700000000,
                "owned_by": "paddlepaddle",
                "permission": [],
            }
        ],
    }


# ══════════════════════════════════════════════════════════════════════
#  POST /convert — backward-compatible file upload OCR
# ══════════════════════════════════════════════════════════════════════
@router.post("/convert")
async def convert_file(
    file: UploadFile = File(...),
    task: str = "ocr",
    max_tokens: int = DEFAULT_MAX_TOKENS,
    req: Request = None,
    auth: tuple[User, APIKey] = Depends(validate_api_key),
    db: Session = Depends(get_db),
):
    """Upload an image or PDF for OCR processing."""
    user, api_key = auth
    backend = get_backend()

    if backend is None:
        raise HTTPException(503, "Model not loaded")

    rate_limiter.check(api_key)

    task = task.lower()
    if task not in VALID_TASKS:
        raise HTTPException(400, f"Invalid task '{task}'. Choose from: {VALID_TASKS}")

    contents = await file.read()
    filename = file.filename or "upload"
    is_pdf = filename.lower().endswith(".pdf") or file.content_type == "application/pdf"

    loop = asyncio.get_event_loop()
    t_start = time.time()

    try:
        if is_pdf:
            # ── PDF: auto-detect all content types, output HTML ──
            effective_task = task if task != "auto" else "auto"

            if effective_task == "auto":
                # Comprehensive processing → HTML document
                html_output, pages, total_elapsed = await loop.run_in_executor(
                    get_executor(), process_pdf_to_html, contents, max_tokens
                )

                record_usage(
                    db, user.id, api_key.id, "/convert",
                    elapsed_seconds=total_elapsed, task="auto",
                    ip_address=req.client.host if req else None,
                )
                live_metrics.record_request(total_elapsed, is_error=False)

                return JSONResponse({
                    "filename": filename,
                    "task": "auto",
                    "format": "html",
                    "total_pages": len(pages),
                    "content": html_output,
                    "pages": pages,
                    "elapsed_seconds": round(total_elapsed, 1),
                    "status": "success",
                })
            else:
                # Specific task for each page
                images = pdf_to_images(contents)
                logger.info(f"PDF '{filename}' — {len(images)} page(s)")

                pages = []
                total_elapsed = 0.0

                for i, img in enumerate(images):
                    logger.info(f"  Page {i+1}/{len(images)} ({img.size[0]}x{img.size[1]}) task={task}")
                    result, elapsed = await loop.run_in_executor(
                        get_executor(), process_single_image, img, task, max_tokens
                    )
                    total_elapsed += elapsed
                    pages.append({
                        "page": i + 1,
                        "content": result,
                        "elapsed_seconds": round(elapsed, 1),
                    })

                combined = "\n\n".join(
                    f"--- Page {p['page']} ---\n{p['content']}" for p in pages
                )
                # Convert each page's markdown to HTML
                for p in pages:
                    p["content_html"] = md_to_html(p["content"])
                combined_html = "\n".join(
                    "<section data-page='{}'>{}</section>".format(p["page"], p["content_html"]) for p in pages
                )

                record_usage(
                    db, user.id, api_key.id, "/convert",
                    elapsed_seconds=total_elapsed, task=task,
                    ip_address=req.client.host if req else None,
                )
                live_metrics.record_request(total_elapsed, is_error=False)

                return JSONResponse({
                    "filename": filename,
                    "task": task,
                    "format": "html",
                    "total_pages": len(images),
                    "content": combined,
                    "content_html": combined_html,
                    "pages": pages,
                    "elapsed_seconds": round(total_elapsed, 1),
                    "status": "success",
                })
        else:
            # ── Image file ──
            effective_task = task if task != "auto" else "doc_parser"
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            logger.info(f"Processing '{filename}' ({image.size[0]}x{image.size[1]}) task={effective_task}")
            result, elapsed = await loop.run_in_executor(
                get_executor(), process_single_image, image, effective_task, max_tokens
            )

            record_usage(
                db, user.id, api_key.id, "/convert",
                elapsed_seconds=elapsed, task=effective_task,
                ip_address=req.client.host if req else None,
            )
            live_metrics.record_request(elapsed, is_error=False)

            return JSONResponse({
                "filename": filename,
                "task": effective_task,
                "format": "html",
                "content": result,
                "content_html": md_to_html(result),
                "elapsed_seconds": round(elapsed, 1),
                "status": "success",
            })
    except HTTPException:
        raise
    except Exception as exc:
        elapsed = time.time() - t_start
        logger.error(f"Processing error: {exc}", exc_info=True)
        record_usage(
            db, user.id, api_key.id, "/convert",
            elapsed_seconds=elapsed, status_code=400,
            error_message=str(exc), task=task,
            ip_address=req.client.host if req else None,
        )
        raise HTTPException(400, str(exc))


# ══════════════════════════════════════════════════════════════════════
#  Health / info (public)
# ══════════════════════════════════════════════════════════════════════
@router.get("/health")
async def health():
    return {
        "status": "healthy" if get_backend() else "loading",
        "backend": get_backend(),
        "model": SERVED_MODEL_NAME,
        "device": get_device(),
        "tasks": VALID_TASKS,
    }


@router.get("/model/info")
async def model_info():
    if get_backend() is None:
        raise HTTPException(503, "Model not loaded")
    return get_model_info()
