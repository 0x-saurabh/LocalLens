"""Apple Vision OCR (macOS). Fully on-device."""

from __future__ import annotations

import cv2
import numpy as np
import Vision
from Foundation import NSData

from locallens.ingest.ocr import OcrLine


def recognize(gray: np.ndarray, languages: list[str] | None = None) -> list[OcrLine]:
    h, w = gray.shape[:2]
    ok, png = cv2.imencode(".png", gray)
    if not ok:
        return []
    data = NSData.dataWithBytes_length_(png.tobytes(), len(png))
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, None)

    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    req.setUsesLanguageCorrection_(True)
    if languages:
        req.setRecognitionLanguages_(languages)
    elif hasattr(req, "setAutomaticallyDetectsLanguage_"):
        req.setAutomaticallyDetectsLanguage_(True)

    success, err = handler.performRequests_error_([req], None)
    if not success:
        raise RuntimeError(f"Vision OCR failed: {err}")

    lines: list[OcrLine] = []
    for obs in req.results() or []:
        cands = obs.topCandidates_(1)
        if not cands:
            continue
        c = cands[0]
        bb = obs.boundingBox()  # normalized, origin bottom-left
        x0 = bb.origin.x * w
        x1 = (bb.origin.x + bb.size.width) * w
        y1 = (1.0 - bb.origin.y) * h
        y0 = (1.0 - bb.origin.y - bb.size.height) * h
        lines.append(OcrLine(text=str(c.string()), confidence=float(c.confidence()), bbox=(x0, y0, x1, y1)))
    return lines
