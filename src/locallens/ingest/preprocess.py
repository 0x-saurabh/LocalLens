"""Image cleanup before OCR: grayscale, deskew, light denoise.

We deliberately do NOT binarize for Apple Vision (it prefers grayscale);
Tesseract gets a binarized copy via `binarize()`.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MAX_DESKEW_DEG = 15.0
MIN_DESKEW_DEG = 0.3


@dataclass
class Preprocessed:
    image: np.ndarray        # grayscale uint8
    skew_deg: float          # rotation applied (0 if none)


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def estimate_skew(gray: np.ndarray) -> float:
    """Angle (degrees) of the dominant text orientation, via min-area rect of ink pixels."""
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # thicken strokes so lines of text merge into bars; this makes minAreaRect stable
    ink = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3)))
    coords = cv2.findNonZero(ink)
    if coords is None or len(coords) < 50:
        return 0.0
    angle = cv2.minAreaRect(coords)[-1]
    if angle > 45:
        angle -= 90
    elif angle < -45:
        angle += 90
    return float(angle)


def rotate(gray: np.ndarray, angle: float) -> np.ndarray:
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def preprocess(img: np.ndarray) -> Preprocessed:
    gray = to_gray(img)
    gray = cv2.fastNlMeansDenoising(gray, None, h=7)
    angle = estimate_skew(gray)
    if MIN_DESKEW_DEG <= abs(angle) <= MAX_DESKEW_DEG:
        gray = rotate(gray, angle)
    else:
        angle = 0.0
    return Preprocessed(image=gray, skew_deg=angle)


def binarize(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
