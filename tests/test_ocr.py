"""Scanned-page path: rasterized text with no text layer → OCR → blocks with confidence."""

import numpy as np
import pymupdf
import pytest

from locallens.ingest import classify, pdf_text
from locallens.ingest.ocr import OcrLine, available_engines, get_engine, lines_to_blocks
from locallens.ingest.preprocess import estimate_skew, rotate, to_gray

HAS_OCR = bool(available_engines())


@pytest.fixture
def scanned_pdf(tmp_path):
    """Render a text page to an image and wrap it in a PDF with NO text layer."""
    src = pymupdf.open()
    page = src.new_page()
    page.insert_text((72, 90), "Invoice 2024-0042", fontsize=20)
    page.insert_textbox(
        pymupdf.Rect(72, 130, 540, 400),
        "Total amount due is 820.00 USD for dental cleaning performed on 11 March 2024. "
        "Payment is expected within thirty days of the invoice date.",
        fontsize=12,
    )
    pix = page.get_pixmap(dpi=150)
    out = pymupdf.open()
    p = out.new_page(width=page.rect.width, height=page.rect.height)
    p.insert_image(p.rect, pixmap=pix)
    path = tmp_path / "scan.pdf"
    out.save(path)
    return path


def test_classify_thresholds():
    assert classify.classify_text_layer(0) == "scanned"
    assert classify.classify_text_layer(5000) == "digital"
    assert classify.refine_after_ocr(0.95, 10) == "scanned"
    assert classify.refine_after_ocr(0.4, 10) == "handwritten"
    assert classify.refine_after_ocr(None, 0) == "scanned"


def test_lines_group_into_paragraphs():
    lines = [
        OcrLine("line one", 0.9, (10, 10, 200, 22)),
        OcrLine("line two", 0.8, (10, 24, 200, 36)),
        OcrLine("new para", 1.0, (10, 70, 200, 82)),
    ]
    blocks = lines_to_blocks(lines, "d:p1", scale=0.5)
    assert [b.text for b in blocks] == ["line one line two", "new para"]
    assert blocks[0].confidence == pytest.approx(0.85)
    assert blocks[0].bbox == (5.0, 5.0, 100.0, 18.0)
    assert blocks[0].source == "ocr"


def test_deskew_recovers_rotation(scanned_pdf):
    gray = to_gray(np.frombuffer(
        (pix := pymupdf.open(scanned_pdf)[0].get_pixmap(dpi=100, colorspace=pymupdf.csGRAY)).samples,
        dtype=np.uint8).reshape(pix.h, pix.w))
    skewed = rotate(gray, 3.0)
    assert abs(abs(estimate_skew(skewed)) - 3.0) < 1.0


def test_scanned_pdf_without_ocr_is_marked_scanned(scanned_pdf):
    doc = pdf_text.extract(scanned_pdf, "scan01", "application/pdf", ocr=None)
    assert doc.pages[0].kind == "scanned" and doc.pages[0].blocks == []


@pytest.mark.skipif(not HAS_OCR, reason="no OCR engine on this machine")
def test_scanned_pdf_is_ocrd(scanned_pdf):
    _, engine = get_engine()
    doc = pdf_text.extract(scanned_pdf, "scan01", "application/pdf", ocr=engine)
    page = doc.pages[0]
    assert page.kind == "scanned"
    text = " ".join(b.text for b in page.blocks)
    assert "820.00" in text and "2024-0042" in text
    assert all(b.source == "ocr" and b.confidence is not None for b in page.blocks)
    # bbox is back in PDF points: inside the page
    for b in page.blocks:
        assert 0 <= b.bbox[0] < b.bbox[2] <= page.width + 1 and 0 <= b.bbox[1] < b.bbox[3] <= page.height + 1
