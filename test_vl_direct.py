#!/usr/bin/env python3
"""Direct PaddleOCRVL test bypassing engine.py."""

import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

from PIL import Image, ImageDraw

# Create test image
img = Image.new("RGB", (800, 600), "white")
draw = ImageDraw.Draw(img)
draw.text((50, 30), "Multi-Head Attention", fill="black")
draw.text((50, 70), "The attention function is:", fill="black")
draw.text((50, 110), "Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V", fill="black")
draw.text((50, 160), "Where Q, K, V are the query, key, and value matrices.", fill="black")

tmp_path = "/tmp/test_ocr_simple.png"
img.save(tmp_path)
print(f"Image saved: {tmp_path}")

from paddleocr import PaddleOCRVL

print("Creating pipeline...")
pipeline = PaddleOCRVL(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
)
print("Pipeline created.")

print("Running predict with file path...")
try:
    results = pipeline.predict(
        tmp_path,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
    )
    for res in results:
        print(f"Result type: {type(res)}")
        print(f"Result dir keys: {[k for k in dir(res) if not k.startswith('_')]}")
        md = res.markdown
        print(f"markdown type: {type(md)}")
        print(f"markdown keys: {list(md.keys()) if isinstance(md, dict) else 'N/A'}")
        md_text = md.get("markdown_texts", "")
        print(f"markdown_texts length: {len(md_text)}")
        print("--- MARKDOWN ---")
        print(md_text)
        print("--- END ---")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

os.remove(tmp_path)
