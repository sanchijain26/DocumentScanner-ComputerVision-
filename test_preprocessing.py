"""
Unit tests for src/preprocessing.py.

These tests use small synthetic images generated with numpy/OpenCV so they
run instantly with no external test fixtures required.
"""

import numpy as np
import pytest

from src.preprocessing import (
    detect_edges,
    enhance_scan,
    four_point_transform,
    order_points,
    resize_for_processing,
)
from src.utils import InvalidFileError, validate_image_path


def make_blank_image(h=400, w=300, color=(255, 255, 255)):
    img = np.full((h, w, 3), color, dtype=np.uint8)
    return img


def test_validate_image_path_rejects_missing_file():
    with pytest.raises(InvalidFileError):
        validate_image_path("does_not_exist.jpg")


def test_validate_image_path_rejects_bad_extension(tmp_path):
    bad_file = tmp_path / "notes.txt"
    bad_file.write_text("hello")
    with pytest.raises(InvalidFileError):
        validate_image_path(str(bad_file))


def test_resize_for_processing_downscales_large_image():
    img = make_blank_image(h=1600, w=1200)
    resized, ratio = resize_for_processing(img, target_height=800)
    assert resized.shape[0] == 800
    assert ratio == pytest.approx(0.5)


def test_resize_for_processing_leaves_small_image_unchanged():
    img = make_blank_image(h=400, w=300)
    resized, ratio = resize_for_processing(img, target_height=800)
    assert resized.shape == img.shape
    assert ratio == 1.0


def test_order_points_returns_tl_tr_br_bl():
    pts = np.array([[10, 10], [200, 10], [200, 300], [10, 300]])
    ordered = order_points(pts)
    top_left, top_right, bottom_right, bottom_left = ordered
    assert top_left[0] < top_right[0]
    assert bottom_left[1] > top_left[1]


def test_four_point_transform_produces_rectangular_output():
    img = make_blank_image(h=400, w=300, color=(0, 0, 0))
    pts = np.array([[10, 10], [280, 10], [280, 380], [10, 380]])
    warped = four_point_transform(img, pts)
    assert warped.shape[0] > 0
    assert warped.shape[1] > 0


def test_detect_edges_returns_single_channel_output():
    img = make_blank_image()
    edges = detect_edges(img)
    assert edges.ndim == 2


def test_enhance_scan_returns_binary_like_image():
    img = make_blank_image()
    scan = enhance_scan(img)
    assert scan.ndim == 2
    unique_values = set(np.unique(scan).tolist())
    assert unique_values.issubset({0, 255})
