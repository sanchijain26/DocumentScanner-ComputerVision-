"""
MODULE 1: Document Preprocessing

OpenCV-based document detection, perspective correction ("flattening"),
and scan enhancement.
"""

import cv2
import numpy as np

from src.logger_config import get_logger
from src.utils import (
    DocumentNotFoundError,
    validate_image_path,
)

logger = get_logger(__name__)


def resize_for_processing(image: np.ndarray, target_height: int = 800):
    """Resize an image to a manageable height and return (image, scale_ratio)."""
    if image is None or image.size == 0:
        raise ValueError("Image is empty.")

    height = image.shape[0]
    if height <= target_height:
        return image.copy(), 1.0

    ratio = target_height / float(height)
    width = int(image.shape[1] * ratio)
    resized = cv2.resize(image, (width, target_height), interpolation=cv2.INTER_AREA)
    return resized, ratio


def detect_edges(image: np.ndarray) -> np.ndarray:
    """Convert an image to grayscale, blur it, and detect Canny edges."""
    if image is None or image.size == 0:
        raise ValueError("Image is empty.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blurred, 75, 200)


def order_points(points: np.ndarray) -> np.ndarray:
    """Return four points in top-left, top-right, bottom-right, bottom-left order."""
    pts = np.asarray(points, dtype="float32").reshape(4, 2)
    ordered = np.zeros((4, 2), dtype="float32")

    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)

    ordered[0] = pts[np.argmin(sums)]   # top-left
    ordered[2] = pts[np.argmax(sums)]   # bottom-right
    ordered[1] = pts[np.argmin(diffs)]  # top-right
    ordered[3] = pts[np.argmax(diffs)]  # bottom-left
    return ordered


def four_point_transform(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a perspective transform using four document corner points."""
    rect = order_points(points)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(round(width_a)), int(round(width_b)))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(round(height_a)), int(round(height_b)))

    if max_width < 1 or max_height < 1:
        raise ValueError("Invalid document dimensions.")

    destination = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1],
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(rect, destination)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


def _find_document_contour(edges: np.ndarray):
    """
    Find the largest plausible four-corner document contour.

    A small amount of morphological closing is used before contour detection
    so broken page borders in photographs can still form a single contour.
    The function also falls back to the largest convex contour's minimum-area
    rectangle when a clean four-point approximation is not available.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    closed = cv2.dilate(closed, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(
        closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    image_area = edges.shape[0] * edges.shape[1]
    min_area = image_area * 0.05

    best_contour = None
    best_area = 0

    for contour in contours[:30]:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(contour, 0.03 * perimeter, True)

        if len(approximation) == 4 and cv2.isContourConvex(approximation):
            if area > best_area:
                best_contour = approximation.reshape(4, 2)
                best_area = area

    if best_contour is not None:
        return best_contour

    # Robust fallback: photographs of plain white pages often have no visible
    # border, so Canny edges may only capture the printed text. In that case,
    # isolate the bright page from a darker background and recover its contour.
    # This is particularly useful for phone photographs of white documents.
    gray = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    # The edge image itself is not suitable for brightness segmentation;
    # therefore this branch is handled by the caller through the optional
    # page contour helper below.
    return None


def _find_page_from_brightness(image: np.ndarray):
    """Detect a bright document page when it has no drawn border."""
    if image is None or image.size == 0:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    # Keep only very bright pixels, which separates a white sheet from a
    # typical darker desk/background.
    _, mask = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = gray.shape[0] * gray.shape[1]

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        area = cv2.contourArea(contour)
        if area < image_area * 0.25:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype("float32")

        # A nearly rectangular bright region can still be represented by its
        # minimum-area rectangle.
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        return np.asarray(box, dtype="float32")

    return None


def enhance_scan(image: np.ndarray) -> np.ndarray:
    """Convert a flattened document to a clean binary-like scan."""
    if image is None or image.size == 0:
        raise ValueError("Image is empty.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        10,
    )


def scan_document(path: str) -> np.ndarray:
    """
    Load an image, detect its document boundary, correct perspective,
    and enhance the resulting scan.
    """
    validate_image_path(path)

    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"OpenCV could not read image: {path}")

    resized, ratio = resize_for_processing(image)
    edges = detect_edges(resized)
    points = _find_document_contour(edges)

    # If the page has no printed/physical border, recover it from its
    # brightness against the surrounding background.
    if points is None:
        points = _find_page_from_brightness(resized)

    if points is None:
        logger.warning("No document contour detected in '%s'", path)
        raise DocumentNotFoundError(
            "No document-like four-corner contour was detected. "
            "Try a clearer image with the full document visible."
        )

    # Map points back to the original image coordinates.
    if ratio != 1.0:
        points = points.astype("float32") / ratio

    warped = four_point_transform(image, points)
    scan = enhance_scan(warped)

    logger.info("Document detected and perspective corrected for '%s'", path)
    return scan
