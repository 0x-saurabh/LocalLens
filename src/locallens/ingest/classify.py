"""Decide how a page should be read."""

from __future__ import annotations

from locallens.models import PageKind

MIN_CHARS_FOR_DIGITAL = 200
HANDWRITING_CONF_THRESHOLD = 0.6   # mean OCR confidence below this → probably handwriting


def classify_text_layer(text_chars: int) -> PageKind:
    return "digital" if text_chars >= MIN_CHARS_FOR_DIGITAL else "scanned"


def refine_after_ocr(mean_confidence: float | None, n_lines: int) -> PageKind:
    """After OCR of a non-digital page: printed scan vs. handwriting (Phase 4 routes the latter to a VLM)."""
    if n_lines == 0:
        return "scanned"
    return "handwritten" if (mean_confidence or 0.0) < HANDWRITING_CONF_THRESHOLD else "scanned"
