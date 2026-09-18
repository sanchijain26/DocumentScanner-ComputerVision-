"""
cli.py
------
Command-line interface that wires Module 1 (preprocessing), Module 2 (OCR),
and Module 3 (export) into a single, runnable pipeline.

Usage:
    python -m src.cli --input path/to/image.jpg --output output/
    python -m src.cli --input-dir sample_images/ --output output/
    python -m src.cli --input path/to/image.jpg --output output/ --min-confidence 0.4
"""

import argparse
import sys
import time

from src.exporter import export_all
from src.logger_config import get_logger
from src.ocr_engine import OCREngine
from src.preprocessing import scan_document
from src.utils import DocumentScannerError, list_images_in_dir

logger = get_logger(__name__)


def process_single_image(path: str, output_dir: str, engine: OCREngine, min_confidence: float) -> bool:
    """Run the full pipeline on one image. Returns True on success."""
    start = time.time()
    try:
        logger.info("Processing '%s'...", path)
        scan = scan_document(path)
        result = engine.extract(scan, min_confidence=min_confidence)
        paths = export_all(result, scan, path, output_dir)

        elapsed = time.time() - start
        logger.info(
            "Done with '%s' in %.2fs | avg confidence %.2f | outputs: %s",
            path, elapsed, result.average_confidence, list(paths.values())
        )
        print(f"\n[OK] {path}")
        print(f"     Lines extracted : {len(result.lines)}")
        print(f"     Avg confidence  : {result.average_confidence:.2f}")
        print(f"     Text saved to   : {paths['text']}")
        print(f"     PDF saved to    : {paths['pdf']}")
        return True

    except DocumentScannerError as exc:
        logger.error("Failed to process '%s': %s", path, exc)
        print(f"\n[FAILED] {path} -> {exc}")
        return False


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="document-scanner-ocr",
        description="Detect, flatten, and OCR a photographed document from the command line.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", help="Path to a single image file.")
    group.add_argument("--input-dir", help="Path to a directory of images (batch mode).")

    parser.add_argument(
        "--output", default="output", help="Directory to write results to (default: output/)."
    )
    parser.add_argument(
        "--min-confidence", type=float, default=0.3,
        help="Minimum OCR confidence (0-1) to keep a recognised line (default: 0.3)."
    )
    parser.add_argument(
        "--languages", nargs="+", default=["en"],
        help="Languages for OCR, e.g. --languages en fr (default: en)."
    )
    parser.add_argument(
        "--gpu", action="store_true", help="Use GPU acceleration for OCR if available."
    )
    return parser


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    engine = OCREngine(languages=args.languages, gpu=args.gpu)

    if args.input:
        targets = [args.input]
    else:
        try:
            targets = list_images_in_dir(args.input_dir)
        except DocumentScannerError as exc:
            print(f"[FAILED] {exc}")
            return 1

    print(f"Processing {len(targets)} image(s) -> output directory: '{args.output}'\n")

    successes = 0
    for path in targets:
        if process_single_image(path, args.output, engine, args.min_confidence):
            successes += 1

    print(f"\nSummary: {successes}/{len(targets)} succeeded. Log: output/pipeline.log")
    return 0 if successes == len(targets) else 2


if __name__ == "__main__":
    sys.exit(main())
