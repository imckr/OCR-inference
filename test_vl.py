#!/usr/bin/env python3
"""Quick test: PaddleOCRVL process_document_image with a synthetic doc image."""

import sys, os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
sys.path.insert(0, ".")

from PIL import Image, ImageDraw

# Create a simple document image with text + table
img = Image.new("RGB", (800, 600), "white")
draw = ImageDraw.Draw(img)

draw.text((50, 30), "Multi-Head Attention", fill="black")
draw.text((50, 70), "The attention function is:", fill="black")
draw.text((50, 110), "Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V", fill="black")
draw.text((50, 160), "Where Q, K, V are the query, key, and value matrices.", fill="black")
draw.text((50, 220), "Table: Model Comparison", fill="black")

# Draw a simple table
for y in [260, 300, 340, 380]:
    draw.line([(50, y), (700, y)], fill="black")
draw.line([(50, 260), (50, 380)], fill="black")
draw.line([(200, 260), (200, 380)], fill="black")
draw.line([(400, 260), (400, 380)], fill="black")
draw.line([(700, 260), (700, 380)], fill="black")

draw.text((60, 268), "Model", fill="black")
draw.text((210, 268), "Parameters", fill="black")
draw.text((410, 268), "Accuracy", fill="black")
draw.text((60, 308), "BERT", fill="black")
draw.text((210, 308), "110M", fill="black")
draw.text((410, 308), "88.9%", fill="black")
draw.text((60, 348), "GPT-2", fill="black")
draw.text((210, 348), "1.5B", fill="black")
draw.text((410, 348), "92.1%", fill="black")

tmp_path = "/tmp/test_ocr_doc.png"
img.save(tmp_path)
print(f"Test image saved: {tmp_path}")

from dotsocr.inference.engine import process_document_image

test_img = Image.open(tmp_path).convert("RGB")
print("Running process_document_image...")
markdown, elapsed = process_document_image(test_img, 4096)
print(f"Elapsed: {elapsed:.1f}s")
print(f"Output length: {len(markdown)} chars")
print("--- OUTPUT START ---")
print(markdown)
print("--- OUTPUT END ---")
