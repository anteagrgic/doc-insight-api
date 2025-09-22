# app/ingest.py
from __future__ import annotations
from typing import List
from .config import settings

# Minimal helperi koje koristi app/api.py:
# - parse_image: OCR na slikama (EasyOCR)
# - parse_text:   decode plain texta

# EasyOCR se učitava lijeno (tek kad zatreba)
_easyocr_reader = None

def _get_ocr_reader():
    """Lazy init EasyOCR readera s jezicima iz settingsa/ENV-a."""
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        langs: List[str] = [x.strip() for x in settings.OCR_LANGS.split(",") if x.strip()]
        _easyocr_reader = easyocr.Reader(langs, gpu=False)
    return _easyocr_reader


def parse_image(data: bytes) -> str:
    """
    OCR nad slikom (PNG/JPG/TIFF...). Vraća spojen tekst linija.
    """
    reader = _get_ocr_reader()
    try:
        lines = reader.readtext(data, detail=0)
    except Exception:
        # U rijetkim slučajevima EasyOCR očekuje path; fallbackamo preko numpy decode-a
        import numpy as np
        import cv2
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        lines = reader.readtext(img, detail=0)
    text = " ".join(line.strip() for line in lines if str(line).strip())
    return text


def parse_text(data: bytes) -> str:
    """
    Dekodiraj plain text. Prvo UTF-8, zatim latin-1 (bez dizanja exceptiona).
    """
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="ignore")


# ---------------------------------------------------------------------
# Legacy/removed:
# - parse_pdf / parse_pdf_pages / chunk_docs / ingest_files
#   Ingestion PDF-ova sada radi isključivo u /upload (app/api.py) preko PyMuPDF.
#   Ako zatreba OCR nad PDF skenom, može se dodati u app/api.py prilikom uploada.
# ---------------------------------------------------------------------
