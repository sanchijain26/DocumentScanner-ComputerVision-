"""
Unit tests for src/ocr_engine.py.

The EasyOCR reader itself (which downloads a neural network model) is
mocked out so these tests run fast and offline -- we are testing our
wrapper logic (filtering, aggregation, error handling), not EasyOCR itself.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.ocr_engine import OCREngine, OCRResult
from src.utils import OCRExtractionError


@pytest.fixture(autouse=True)
def reset_reader_cache():
    """Ensure the class-level cached reader doesn't leak between tests."""
    OCREngine._reader = None
    yield
    OCREngine._reader = None


def test_extract_filters_low_confidence_lines(monkeypatch):
    fake_reader = MagicMock()
    fake_reader.readtext.return_value = [
        ([[0, 0], [10, 0], [10, 10], [0, 10]], "Hello", 0.95),
        ([[0, 0], [10, 0], [10, 10], [0, 10]], "??garbled??", 0.10),
    ]
    engine = OCREngine()
    monkeypatch.setattr(engine, "_get_reader", lambda: fake_reader)

    result = engine.extract(np.zeros((10, 10, 3), dtype=np.uint8), min_confidence=0.5)

    assert isinstance(result, OCRResult)
    assert len(result.lines) == 1
    assert result.lines[0].text == "Hello"


def test_extract_raises_when_no_text_found(monkeypatch):
    fake_reader = MagicMock()
    fake_reader.readtext.return_value = []
    engine = OCREngine()
    monkeypatch.setattr(engine, "_get_reader", lambda: fake_reader)

    with pytest.raises(OCRExtractionError):
        engine.extract(np.zeros((10, 10, 3), dtype=np.uint8))


def test_ocr_result_full_text_joins_lines():
    from src.ocr_engine import OCRLine

    result = OCRResult(lines=[OCRLine("Line one", 0.9), OCRLine("Line two", 0.8)])
    assert result.full_text == "Line one\nLine two"


def test_ocr_result_average_confidence():
    from src.ocr_engine import OCRLine

    result = OCRResult(lines=[OCRLine("A", 1.0), OCRLine("B", 0.5)])
    assert result.average_confidence == pytest.approx(0.75)


def test_ocr_result_average_confidence_empty():
    result = OCRResult(lines=[])
    assert result.average_confidence == 0.0
