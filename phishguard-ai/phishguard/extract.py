"""File and OCR text extraction for PDF, DOCX, and Screenshots/Images."""
from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Tuple


def extract_text_from_file(file_bytes: bytes, filename: str) -> Tuple[str, str, dict]:
    """
    Extracts text from uploaded documents or image screenshots.
    Returns: (text, file_type_description, metadata)
    """
    ext = filename.lower().split(".")[-1]
    meta = {"filename": filename, "extension": ext, "bytes": len(file_bytes)}

    # 1. PDF Attachments
    if ext == "pdf":
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            pages_text = []
            for i, page in enumerate(doc):
                pages_text.append(page.get_text())
            full_text = "\n".join(pages_text).strip()
            meta["pages"] = len(doc)
            return full_text, f"PDF Document ({len(doc)} pages)", meta
        except Exception as e:
            try:
                import pypdf
                import io
                reader = pypdf.PdfReader(io.BytesIO(file_bytes))
                pages = [page.extract_text() or "" for page in reader.pages]
                full_text = "\n".join(pages).strip()
                meta["pages"] = len(reader.pages)
                return full_text, f"PDF Document ({len(reader.pages)} pages)", meta
            except Exception as e2:
                return f"[Error extracting PDF: {e2}]", "PDF Error", meta

    # 2. Word Document Attachments (.docx)
    elif ext in ["docx", "doc"]:
        try:
            import docx
            import io
            doc = docx.Document(io.BytesIO(file_bytes))
            full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip()).strip()
            meta["paragraphs"] = len(doc.paragraphs)
            return full_text, f"Word Document ({len(doc.paragraphs)} paragraphs)", meta
        except Exception as e:
            return f"[Error reading Word document: {e}]", "Word Error", meta

    # 3. Screenshot / Image OCR (.png, .jpg, .jpeg, .webp, .tiff)
    elif ext in ["png", "jpg", "jpeg", "webp", "tiff", "bmp"]:
        tesseract_bin = "/opt/homebrew/bin/tesseract"
        if not os.path.exists(tesseract_bin):
            tesseract_bin = "tesseract"
        
        with tempfile.NamedTemporaryFile(suffix="." + ext, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            # Run Tesseract OCR on image
            res = subprocess.run([tesseract_bin, tmp_path, "stdout", "--oem", "1"], capture_output=True, text=True, timeout=20)
            text = res.stdout.strip()
            meta["ocr_engine"] = "Tesseract 5.5"
            return text, "OCR Screenshot / Image Scan", meta
        except Exception as e:
            return f"[OCR Error: {e}]", "OCR Error", meta
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # 4. Plain Text (.txt, .eml)
    elif ext in ["txt", "eml"]:
        try:
            text = file_bytes.decode("utf-8", errors="ignore")
            return text, "Text / Email Export", meta
        except Exception as e:
            return str(e), "Text Error", meta

    return "", f"Unsupported file type (.{ext})", meta
