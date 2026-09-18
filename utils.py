"""
utils.py
--------
Shared helpers, custom exceptions, and validation routines used across
the pipeline.

Non-functional requirements addressed: Error handling strategy, Reliability.
"""

import os

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


class DocumentScannerError(Exception):
    """Base exception for all pipeline-specific errors."""


class InvalidFileError(DocumentScannerError):
    """Raised when an input path does not point to a supported image file."""


class DocumentNotFoundError(DocumentScannerError):
    """Raised when no document-like contour can be detected in an image."""


class OCRExtractionError(DocumentScannerError):
    """Raised when the OCR engine fails to produce any usable text."""


def validate_image_path(path: str) -> str:
    """
    Confirm that `path` exists and has a supported image extension.
    Returns the path unchanged if valid; raises InvalidFileError otherwise.
    """
    if not os.path.isfile(path):
        raise InvalidFileError(f"File not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise InvalidFileError(
            f"Unsupported file type '{ext}'. Supported types: "
            f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return path


def ensure_dir(path: str) -> str:
    """Create `path` (and parents) if it does not already exist."""
    os.makedirs(path, exist_ok=True)
    return path


def list_images_in_dir(directory: str) -> list:
    """Return a sorted list of supported image file paths within `directory`."""
    if not os.path.isdir(directory):
        raise InvalidFileError(f"Directory not found: {directory}")

    images = [
        os.path.join(directory, f)
        for f in sorted(os.listdir(directory))
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ]
    if not images:
        raise InvalidFileError(f"No supported images found in {directory}")
    return images
