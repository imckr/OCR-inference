"""Document parsing pipeline: layout analysis → per-region VLM recognition.

Implements the PaddleOCR-VL two-stage approach:
1. PP-DocLayoutV2 detects layout regions (text, table, formula, chart, …)
2. Each region is cropped and sent to the VLM with the appropriate task prompt
3. Results are aggregated in reading order into structured Markdown
"""

import gc
import logging
import time
from pathlib import Path
from typing import List, Optional

from PIL import Image

from dotsocr.config import MODEL_DIR, TASK_PROMPTS, DEFAULT_MAX_TOKENS
from dotsocr.inference.layout import LayoutRegion, detect_layout

logger = logging.getLogger("dotsocr.doc_parser")

_LAYOUT_MODEL_DIR = MODEL_DIR.parent / "PP-DocLayoutV2"

# Padding (pixels) added around each cropped region for VLM context
_CROP_PADDING = 4

# Labels that produce headings in the Markdown output
_HEADING_LABELS = {"doc_title", "paragraph_title"}

# Labels to skip entirely (no VLM call, no output)
_SKIP_LABELS = {"image", "footer_image", "header_image", "seal", "vision_footnote"}


def _crop_region(image: Image.Image, region: LayoutRegion) -> Image.Image:
    """Crop a region from the image with a small padding border."""
    w, h = image.size
    x1 = max(0, int(region.x1) - _CROP_PADDING)
    y1 = max(0, int(region.y1) - _CROP_PADDING)
    x2 = min(w, int(region.x2) + _CROP_PADDING)
    y2 = min(h, int(region.y2) + _CROP_PADDING)
    return image.crop((x1, y1, x2, y2))


def _format_region_output(label: str, text: str) -> str:
    """Wrap recognized text with appropriate Markdown formatting."""
    text = text.strip()
    if not text:
        return ""

    if label in _HEADING_LABELS:
        level = "##" if label == "doc_title" else "###"
        return f"{level} {text}"

    if label == "footnote":
        return f"> {text}"

    # Tables and formulas are already in their canonical format from the VLM
    return text


def parse_document(
    image: Image.Image,
    infer_fn,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    layout_threshold: float = 0.5,
) -> tuple[str, float]:
    """Parse a document image using layout analysis + per-region VLM recognition.

    Args:
        image: Full-page document image.
        infer_fn: Callable(image, task, max_tokens) → (text, elapsed_seconds).
                  This is typically ``engine.process_single_image``.
        max_tokens: Maximum generation tokens per region.
        layout_threshold: Confidence threshold for layout detection.

    Returns:
        (markdown_text, total_elapsed_seconds)
    """
    t_start = time.time()

    if not _LAYOUT_MODEL_DIR.exists():
        logger.warning(
            f"PP-DocLayoutV2 not found at {_LAYOUT_MODEL_DIR} — "
            f"falling back to full-page OCR"
        )
        return infer_fn(image, "ocr", max_tokens)

    # Stage 1: Layout analysis (CPU, fast)
    t_layout = time.time()
    regions = detect_layout(image, _LAYOUT_MODEL_DIR, threshold=layout_threshold)
    layout_time = time.time() - t_layout
    logger.info(f"Layout analysis: {len(regions)} regions in {layout_time:.2f}s")

    if not regions:
        logger.info("No layout regions detected — falling back to full-page OCR")
        return infer_fn(image, "ocr", max_tokens)

    # Stage 2: Per-region VLM recognition
    output_parts: List[str] = []
    total_infer_time = 0.0

    for i, region in enumerate(regions):
        task = region.task
        if task is None or region.label in _SKIP_LABELS:
            logger.debug(f"  Region {i}: {region.label} — skipped")
            continue

        prompt = TASK_PROMPTS.get(task)
        if not prompt:
            logger.warning(f"  Region {i}: unknown task '{task}' for label '{region.label}'")
            continue

        cropped = _crop_region(image, region)
        logger.info(
            f"  Region {i}: {region.label} ({region.score:.2f}) "
            f"[{region.x1:.0f},{region.y1:.0f},{region.x2:.0f},{region.y2:.0f}] "
            f"→ {cropped.size[0]}x{cropped.size[1]} task={task}"
        )

        text, elapsed = infer_fn(cropped, task, max_tokens)
        total_infer_time += elapsed

        formatted = _format_region_output(region.label, text)
        if formatted:
            output_parts.append(formatted)

    gc.collect()

    markdown = "\n\n".join(output_parts)
    total_elapsed = time.time() - t_start

    logger.info(
        f"Document parsing complete: {len(output_parts)} sections, "
        f"layout={layout_time:.1f}s, inference={total_infer_time:.1f}s, "
        f"total={total_elapsed:.1f}s"
    )

    return markdown, total_elapsed
