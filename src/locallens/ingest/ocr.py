"""OCR engine selection and line→block grouping shared by all engines."""

from __future__ import annotations

import platform
import shutil
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from locallens.models import Block


@dataclass
class OcrLine:
    text: str
    confidence: float                              # 0..1
    bbox: tuple[float, float, float, float]        # pixels, top-left origin


OcrEngine = Callable[[np.ndarray], list[OcrLine]]


def available_engines() -> list[str]:
    names = []
    if platform.system() == "Darwin":
        try:
            import Vision  # noqa: F401
            names.append("vision")
        except ImportError:
            pass
    if shutil.which("tesseract"):
        try:
            import pytesseract  # noqa: F401
            names.append("tesseract")
        except ImportError:
            pass
    return names


def get_engine(preferred: str = "auto") -> tuple[str, OcrEngine]:
    engines = available_engines()
    if not engines:
        raise RuntimeError(
            "No OCR engine available. On macOS install pyobjc-framework-Vision; "
            "elsewhere `brew/apt install tesseract` and `uv sync --extra tesseract`."
        )
    name = engines[0] if preferred == "auto" else preferred
    if name not in engines:
        raise RuntimeError(f"OCR engine {name!r} not available; have {engines}")
    if name == "vision":
        from locallens.ingest.ocr_vision import recognize
        return name, recognize
    from locallens.ingest.ocr_tesseract import recognize
    return name, recognize


def lines_to_blocks(lines: list[OcrLine], id_prefix: str, scale: float = 1.0,
                    heading_path: list[str] | None = None) -> list[Block]:
    """Group OCR lines into paragraph blocks by vertical gap. `scale` maps pixels→PDF points."""
    if not lines:
        return []
    lines = sorted(lines, key=lambda l: (round(l.bbox[1] / 10), l.bbox[0]))
    heights = [l.bbox[3] - l.bbox[1] for l in lines]
    line_h = statistics.median(heights) or 1.0

    groups: list[list[OcrLine]] = [[lines[0]]]
    for prev, cur in pairwise(lines):
        gap = cur.bbox[1] - prev.bbox[3]
        new_para = gap > 0.7 * line_h or cur.bbox[0] > prev.bbox[2]  # big gap or next column
        (groups.append([cur]) if new_para else groups[-1].append(cur))

    blocks: list[Block] = []
    for i, g in enumerate(groups):
        parts: list[str] = []
        for l in g:
            t = l.text.strip()
            if parts and parts[-1].endswith("-") and t[:1].islower():
                parts[-1] = parts[-1][:-1] + t      # re-join a word hyphenated at the line end
            else:
                parts.append(t)
        text = " ".join(parts)
        x0 = min(l.bbox[0] for l in g) * scale
        y0 = min(l.bbox[1] for l in g) * scale
        x1 = max(l.bbox[2] for l in g) * scale
        y1 = max(l.bbox[3] for l in g) * scale
        blocks.append(Block(
            id=f"{id_prefix}:o{i}", type="paragraph", text=text, bbox=(x0, y0, x1, y1),
            source="ocr", confidence=round(statistics.fmean(l.confidence for l in g), 3),
            heading_path=list(heading_path or []),
        ))
    return blocks
