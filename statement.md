# `statement.md`

# Project Statement

## Document Scanner Using OCR

**Name:** Mohit Poonia  
**Registration No.:** 24BAI10966  
**Domain:** Computer Vision

## Problem Statement

Photographs of documents may contain tilted pages, unwanted backgrounds, uneven lighting, and other distortions. These issues can make the image difficult to use as a scanned document and can also affect text recognition.

The aim of this project is to develop a simple document scanner that detects a document from an image, corrects its perspective, improves the scanned image, and extracts its text using OCR.

## Proposed Solution

The project is implemented in Python using OpenCV and EasyOCR. It is divided into three main modules.

### Module 1 — Preprocessing

The system validates the input image, detects document edges and boundaries, identifies the document corners, corrects the perspective, and enhances the resulting scan.

### Module 2 — OCR

The processed document is passed to EasyOCR for text extraction. OCR results are stored with confidence values, and low-confidence results can be filtered using a selected threshold.

### Module 3 — Export

The processed scan and OCR results are exported as PNG, TXT, JSON, and PDF files. The system also maintains a pipeline log.

## Functional Requirements

1. The system should accept and validate supported image files.
2. It should detect and straighten the document using Computer Vision techniques.
3. It should extract text using EasyOCR.
4. It should allow OCR confidence filtering.
5. It should export the processed results in multiple formats.
6. It should handle common processing errors clearly.

## Non-Functional Requirements

- **Performance:** Process normal document images within a reasonable time.
- **Reliability:** Handle invalid input and processing failures properly.
- **Usability:** Provide simple command-line options and clear messages.
- **Maintainability:** Keep preprocessing, OCR, exporting, and utilities in separate modules.
- **Testing:** Use automated tests to check important functions.
- **Logging:** Record important pipeline events.

## Technology Stack

- Python
- OpenCV
- NumPy
- EasyOCR
- FPDF2
- pytest

## Testing

The project contains automated tests for preprocessing and OCR functionality. The current test suite contains **13 tests**, and the development test run completed with:

```text
13 passed

A synthetic document image was also used to check the preprocessing pipeline.

Scope

The project demonstrates a basic Computer Vision pipeline for document detection, perspective correction, image enhancement, OCR integration, and result export.

Future Scope

The project can be extended with a graphical interface, camera capture, automatic orientation correction, improved document detection, additional OCR engines, and better PDF generation.

Conclusion

Document Scanner Using OCR demonstrates how image processing and OCR can be combined to convert a document photograph into a cleaner digital scan with extracted text. The modular structure also makes the project easier to test and extend.

Declaration

I, Mohit Poonia, Registration No. 24BAI10966, submit this project as an academic Computer Vision project titled Document Scanner Using OCR.

Name: Mohit Poonia
Registration No.: 24BAI10966
