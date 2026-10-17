"""Tesseract OCR fallback (cross-platform). Requires the `tesseract` binary + pytesseract."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pytesseract

from locallens.ingest.ocr import OcrLine
from locallens.ingest.preprocess import binarize


def recognize(gray: np.ndarray, languages: list[str] | None = None) -> list[OcrLine]:
    lang = "+".join(l.split("-")[0] for l in languages) if languages else "eng"
    data = pytesseract.image_to_data(binarize(gray), lang=lang, output_type=pytesseract.Output.DICT)
    words: dict[tuple, list[int]] = defaultdict(list)
    for i, txt in enumerate(data["text"]):
        if txt.strip() and float(data["conf"][i]) >= 0:
            words[(data["block_num"][i], data["par_num"][i], data["line_num"][i])].append(i)
    lines: list[OcrLine] = []
    for idxs in words.values():
        text = " ".join(data["text"][i] for i in idxs)
        conf = sum(float(data["conf"][i]) for i in idxs) / len(idxs) / 100.0
        x0 = min(data["left"][i] for i in idxs)
        y0 = min(data["top"][i] for i in idxs)
        x1 = max(data["left"][i] + data["width"][i] for i in idxs)
        y1 = max(data["top"][i] + data["height"][i] for i in idxs)
        lines.append(OcrLine(text=text, confidence=conf, bbox=(x0, y0, x1, y1)))
    return lines
