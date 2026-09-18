A Computer Vision project that converts document images into clean scanned documents and extracts text using OCR.

Author: Sanchita Jain 
Registration No.: 24BAI10497

Course: Computer Vision.

🚀 Features

📷 Detects documents from photographs automatically
🔲 Detects document boundaries using contours and edge detection
📐 Corrects perspective and flattens the document
🖼️ Enhances the scan using adaptive thresholding
🔤 Extracts text using EasyOCR
🎯 Filters OCR results using a configurable confidence threshold
📁 Supports single-image and batch processing
📄 Exports results as PNG, TXT, JSON, and PDF
⚠️ Includes custom error handling
📝 Provides timestamped logging
🧪 Includes automated pytest unit tests
🛠️ Technologies Used

Python
OpenCV
NumPy
EasyOCR
FPDF2
pytest
🔄 Workflow

Input Image → Preprocessing → Document Detection → Perspective Correction → Scan Enhancement → OCR → Export

The preprocessing stage uses grayscale conversion, Gaussian blur, Canny edge detection, contour detection, morphological operations, perspective transformation, and adaptive thresholding. The processed document is then passed to EasyOCR for text extraction.

📤 Output Formats

The project generates:

"*_scan.png" — enhanced scanned document
"*.txt" — extracted text
"*.json" — structured OCR results and confidence values
"*.pdf" — scan and extracted text in PDF format
🧪 Testing

The project includes 13 automated pytest tests covering preprocessing and OCR-wrapper functionality. All 13 tests passed in the development environment.

📌 Future Scope

Graphical User Interface (GUI)
Direct camera capture
Automatic page rotation
Improved detection for complex backgrounds
Support for additional OCR engines
Searchable PDFs with a true text layer
👩‍💻 Project

Course: Computer Vision Project: Document Scanner Using OCR
