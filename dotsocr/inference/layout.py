"""PP-DocLayoutV2 layout analysis via PaddlePaddle Inference.

Detects semantic regions (text, table, formula, chart, etc.) and their
bounding boxes in a document page image. This is the first stage of the
PaddleOCR-VL two-stage pipeline.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger("dotsocr.layout")

# ── Label list from PP-DocLayoutV2 config.json ────────────────────────
LAYOUT_LABELS = [
    "abstract", "algorithm", "aside_text", "chart", "content",
    "display_formula", "doc_title", "figure_title", "footer",
    "footer_image", "footnote", "formula_number", "header",
    "header_image", "image", "inline_formula", "number",
    "paragraph_title", "reference", "reference_content", "seal",
    "table", "text", "vertical_text", "vision_footnote",
]

# Map each layout label to the VLM task prompt name (None → skip region)
LABEL_TO_TASK = {
    "text": "ocr",
    "content": "ocr",
    "abstract": "ocr",
    "aside_text": "ocr",
    "doc_title": "ocr",
    "figure_title": "ocr",
    "footnote": "ocr",
    "footer": "ocr",
    "header": "ocr",
    "paragraph_title": "ocr",
    "reference": "ocr",
    "reference_content": "ocr",
    "vertical_text": "ocr",
    "number": "ocr",
    "formula_number": "ocr",
    "algorithm": "ocr",
    "table": "table",
    "display_formula": "formula",
    "inline_formula": "formula",
    "chart": "chart",
    # Visual regions — skip VLM recognition
    "image": None,
    "footer_image": None,
    "header_image": None,
    "seal": None,
    "vision_footnote": None,
}

_INPUT_SIZE = 800
_CONFIDENCE_THRESHOLD = 0.5


@dataclass
class LayoutRegion:
    label: str
    score: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def task(self) -> Optional[str]:
        return LABEL_TO_TASK.get(self.label)

    @property
    def area(self) -> float:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)


# ── Module-level predictor (loaded once) ──────────────────────────────
_predictor = None
_model_dir: Optional[Path] = None


def _load_predictor(model_dir: Path):
    """Load the PP-DocLayoutV2 PaddlePaddle inference predictor."""
    global _predictor, _model_dir

    if _predictor is not None and _model_dir == model_dir:
        return

    import paddle.inference as pi

    pdmodel = str(model_dir / "inference.pdmodel")
    pdiparams = str(model_dir / "inference.pdiparams")

    config = pi.Config(pdmodel, pdiparams)
    config.disable_gpu()
    config.switch_ir_optim(True)
    config.enable_memory_optim()
    # Suppress verbose IR optimization logs on repeated runs
    config.switch_use_feed_fetch_ops(False)

    _predictor = pi.create_predictor(config)
    _model_dir = model_dir
    logger.info(f"PP-DocLayoutV2 loaded from {model_dir}")


def _preprocess(image: Image.Image) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare image for PP-DocLayoutV2.

    Preprocessing from config.json:
    1. Resize to 800×800 (no aspect ratio preservation)
    2. NormalizeImage: mean=0, std=1 (effectively just cast to float32)
    3. Permute: HWC → CHW
    """
    orig_w, orig_h = image.size

    img = image.resize((_INPUT_SIZE, _INPUT_SIZE), Image.Resampling.BILINEAR)
    img_array = np.array(img, dtype=np.float32)  # (800, 800, 3)
    img_array = img_array.transpose(2, 0, 1)  # (3, 800, 800)
    img_array = img_array[np.newaxis, ...]  # (1, 3, 800, 800)

    im_shape = np.array([[float(orig_h), float(orig_w)]], dtype=np.float32)
    scale_factor = np.array([
        [float(orig_h) / _INPUT_SIZE, float(orig_w) / _INPUT_SIZE]
    ], dtype=np.float32)

    return img_array, im_shape, scale_factor


def detect_layout(
    image: Image.Image,
    model_dir: Path,
    threshold: float = _CONFIDENCE_THRESHOLD,
) -> List[LayoutRegion]:
    """Run layout analysis and return detected regions sorted in reading order.

    Reading order: top-to-bottom, then left-to-right within similar vertical
    positions (rows grouped by y-center proximity).
    """
    _load_predictor(model_dir)

    img_tensor, im_shape, scale_factor = _preprocess(image)

    for name, data in [
        ("image", img_tensor),
        ("im_shape", im_shape),
        ("scale_factor", scale_factor),
    ]:
        handle = _predictor.get_input_handle(name)
        handle.reshape(list(data.shape))
        handle.copy_from_cpu(data)

    _predictor.run()

    out_names = _predictor.get_output_names()
    detections = _predictor.get_output_handle(out_names[0]).copy_to_cpu()
    # detections shape: (N, 6+) → [class_id, score, x1, y1, x2, y2, ...]

    orig_w, orig_h = image.size
    regions: List[LayoutRegion] = []

    for det in detections:
        class_id = int(det[0])
        score = float(det[1])

        if score < threshold:
            continue
        if class_id < 0 or class_id >= len(LAYOUT_LABELS):
            continue

        label = LAYOUT_LABELS[class_id]
        # Clamp bbox to image bounds
        x1 = max(0.0, float(det[2]))
        y1 = max(0.0, float(det[3]))
        x2 = min(float(orig_w), float(det[4]))
        y2 = min(float(orig_h), float(det[5]))

        if x2 <= x1 or y2 <= y1:
            continue

        regions.append(LayoutRegion(
            label=label, score=score,
            x1=x1, y1=y1, x2=x2, y2=y2,
        ))

    # Sort in reading order: group by vertical position, then left-to-right
    regions.sort(key=lambda r: (r.y1, r.x1))

    logger.info(
        f"Layout analysis: {len(regions)} regions from {len(detections)} queries "
        f"(threshold={threshold})"
    )
    for r in regions:
        logger.debug(
            f"  {r.label} ({r.score:.2f}) [{r.x1:.0f},{r.y1:.0f},{r.x2:.0f},{r.y2:.0f}]"
        )

    return regions
