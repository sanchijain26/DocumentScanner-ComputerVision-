"""
exporter.py
-----------
MODULE 3: Output & Export

Takes the flattened document image plus its OCRResult and writes results
to disk in the formats a user actually wants to keep: a plain-text file,
a PDF report containing the scanned image and extracted text, a JSON record
(useful for downstream processing / batch reporting), and the flattened
scan image itself.
"""

import json
import os

import cv2
import numpy as np
from fpdf import FPDF

from src.logger_config import get_logger
from src.ocr_engine import OCRResult

logger = get_logger(__name__)


def save_scan_image(image: np.ndarray, output_path: str) -> str:
    """Persist the flattened/enhanced scan as an image file."""
    cv2.imwrite(output_path, image)
    logger.info("Saved scan image to '%s'", output_path)
    return output_path


def save_text(result: OCRResult, output_path: str) -> str:
    """Write extracted text to a plain .txt file."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result.full_text)
    logger.info("Saved extracted text to '%s'", output_path)
    return output_path


def save_json(result: OCRResult, source_image: str, output_path: str) -> str:
    """Write a structured JSON record: source file, lines, and confidences."""
    payload = {
        "source_image": source_image,
        "average_confidence": round(result.average_confidence, 4),
        "line_count": len(result.lines),
        "lines": [
            {"text": line.text, "confidence": round(line.confidence, 4)}
            for line in result.lines
        ],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("Saved JSON report to '%s'", output_path)
    return output_path


def save_pdf(result: OCRResult, scan_image_path: str, output_path: str) -> str:
    """
    Build a simple PDF containing the scanned image followed by the
    extracted text, produced without extra native dependencies.
    """
    pdf = FPDF()
    pdf.add_page()

    if os.path.isfile(scan_image_path):
        pdf.image(scan_image_path, x=10, y=10, w=190)
        pdf.add_page()

    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 6, result.full_text)

    pdf.output(output_path)
    logger.info("Saved PDF report to '%s'", output_path)
    return output_path


def export_all(result: OCRResult, scan_image: np.ndarray, source_image: str, output_dir: str) -> dict:
    """
    Convenience wrapper that writes every supported output format for one
    processed document and returns their paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(source_image))[0]

    paths = {}
    paths["scan_image"] = save_scan_image(
        scan_image, os.path.join(output_dir, f"{base_name}_scan.png")
    )
    paths["text"] = save_text(result, os.path.join(output_dir, f"{base_name}.txt"))
    paths["json"] = save_json(
        result, source_image, os.path.join(output_dir, f"{base_name}.json")
    )
    paths["pdf"] = save_pdf(
        result, paths["scan_image"], os.path.join(output_dir, f"{base_name}.pdf")
    )
    return paths
