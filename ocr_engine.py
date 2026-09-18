"""
ocr_engine.py
-------------
MODULE 2: Optical Character Recognition

Wraps EasyOCR to extract text from a preprocessed (flattened, enhanced)
document image. Keeps the OCR backend isolated behind a small interface
(`OCREngine`) so it could be swapped for Tesseract or another engine
without touching the rest of the pipeline (maintainability / modularity).
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np

from src.logger_config import get_logger
from src.utils import OCRExtractionError

logger = get_logger(__name__)


@dataclass
class OCRLine:
    """A single recognised line of text with its confidence and bounding box."""

    text: str
    confidence: float
    bbox: list = field(default_factory=list)


@dataclass
class OCRResult:
    """Aggregate OCR result for one image."""

    lines: List[OCRLine]

    @property
    def full_text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    @property
    def average_confidence(self) -> float:
        if not self.lines:
            return 0.0
        return sum(line.confidence for line in self.lines) / len(self.lines)


class OCREngine:
    """
    Thin wrapper around EasyOCR's Reader.

    The reader is lazily instantiated (and cached) on first use, since
    loading the recognition model is the most expensive step in the whole
    pipeline (performance non-functional requirement).
    """

    _reader = None  # class-level cache shared across instances

    def __init__(self, languages: List[str] = None, gpu: bool = False):
        self.languages = languages or ["en"]
        self.gpu = gpu

    def _get_reader(self):
        if OCREngine._reader is None:
            try:
                import easyocr  # imported lazily so tests can mock this module
            except ModuleNotFoundError as exc:
                logger.error("EasyOCR is not installed; OCR cannot start")
                raise OCRExtractionError(
                    "EasyOCR is not installed. Please install the project dependencies "
                    "with: pip install -r requirements.txt"
                ) from exc

            logger.info("Initialising EasyOCR reader for languages=%s (gpu=%s)",
                        self.languages, self.gpu)
            try:
                OCREngine._reader = easyocr.Reader(self.languages, gpu=self.gpu)
            except Exception as exc:
                logger.exception("EasyOCR reader could not be initialised")
                raise OCRExtractionError(
                    "EasyOCR could not be initialised. Check that its model files are "
                    "available and try again."
                ) from exc
        return OCREngine._reader

    def extract(self, image: np.ndarray, min_confidence: float = 0.3) -> OCRResult:
        """
        Run OCR on `image` (a numpy array, e.g. from preprocessing.scan_document)
        and return an OCRResult. Lines below `min_confidence` are discarded
        as noise.
        """
        reader = self._get_reader()
        raw_results = reader.readtext(image)

        lines = [
            OCRLine(text=text.strip(), confidence=float(conf), bbox=bbox)
            for (bbox, text, conf) in raw_results
            if conf >= min_confidence and text.strip()
        ]

        if not lines:
            logger.warning("OCR produced no text above confidence threshold %.2f", min_confidence)
            raise OCRExtractionError(
                "No text could be extracted from the image (try a clearer photo, "
                "better lighting, or lower --min-confidence)."
            )

        logger.info("Extracted %d line(s) of text, avg confidence %.2f",
                    len(lines), sum(l.confidence for l in lines) / len(lines))

        return OCRResult(lines=lines)
