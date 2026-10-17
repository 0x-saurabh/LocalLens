"""Local vision-language model transcription (handwriting, messy scans, scanned tables).

Runs through Ollama on 127.0.0.1. Output is parsed into paragraph and table blocks.
Confidence is derived from the share of words the model itself marked unreadable,
cross-checked against OCR agreement when OCR text is available. VLM output is capped
at 0.85 because a transcription without coordinates can never be fully verified.
"""

from __future__ import annotations

import re
from contextlib import suppress
from dataclasses import dataclass

import cv2
import numpy as np
import ollama

from locallens import config
from locallens.models import Block

PROMPT = (
    "Transcribe ALL text in this image exactly as written, line by line, in reading order. "
    "Rules: do not summarize, translate, correct or add anything. Keep the original language. "
    "Write [?] in place of any word you cannot read. "
    "If the image contains a table, output it as a Markdown table with a header row. "
    "Separate paragraphs with a blank line. Output only the transcription."
)
MAX_CONF = 0.85
MAX_SIDE = 1280          # Qwen2.5-VL token budget: keep the long side at or under this
NUM_CTX = 8192
_UNREADABLE = re.compile(r"\[\?+\]")
_WORD = re.compile(r"\w+")


@dataclass
class VlmResult:
    text: str
    confidence: float
    unreadable_ratio: float
    ocr_agreement: float | None


class Transcriber:
    def __init__(self, cfg: config.Config | None = None):
        self.cfg = cfg or config.load()
        self.client = ollama.Client(host=self.cfg.ollama_host)
        self.model = self.cfg.models.vlm
        self.errors: list[str] = []

    def available(self) -> bool:
        try:
            names = {m.model for m in self.client.list().models}
        except Exception:  # noqa: BLE001
            return False
        base = self.model.split(":")[0]
        return any(n == self.model or n.split(":")[0] == base for n in names)

    def transcribe(self, image: np.ndarray, ocr_text: str | None = None) -> VlmResult:
        h, w = image.shape[:2]
        scale = MAX_SIDE / max(h, w)
        if scale < 1.0:
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        ok, png = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError("could not encode page image")
        res = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": PROMPT, "images": [png.tobytes()]}],
            options={"temperature": 0, "num_ctx": NUM_CTX},
        )
        text = (res.message.content or "").strip()
        return VlmResult(text=text, **score(text, ocr_text))

    def unload(self) -> None:
        """Free the VLM's RAM so the chat LLM can load without pressure."""
        with suppress(Exception):  # Ollama may already have evicted it
            self.client.generate(model=self.model, prompt="", keep_alive=0)


def score(text: str, ocr_text: str | None) -> dict:
    words = _WORD.findall(text)
    n_unreadable = len(_UNREADABLE.findall(text))
    total = len(words) + n_unreadable
    unreadable_ratio = n_unreadable / total if total else 1.0
    agreement = None
    if ocr_text:
        a = {w.lower() for w in _WORD.findall(text)}
        b = {w.lower() for w in _WORD.findall(ocr_text)}
        agreement = len(a & b) / len(a | b) if a | b else 0.0
    conf = (1.0 - unreadable_ratio) * MAX_CONF
    if agreement is not None and agreement > 0.8:
        conf = max(conf, min(MAX_CONF, 0.6 + agreement * 0.25))  # corroborated by OCR
    return {"confidence": round(conf, 3), "unreadable_ratio": round(unreadable_ratio, 3),
            "ocr_agreement": None if agreement is None else round(agreement, 3)}


def _parse_md_table(lines: list[str]) -> list[list[str]] | None:
    rows = []
    for ln in lines:
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
            continue  # separator row
        rows.append(cells)
    if len(rows) < 2:
        return None
    width = max(len(r) for r in rows)
    return [r + [""] * (width - len(r)) for r in rows]


def to_blocks(result: VlmResult, id_prefix: str, page_bbox: tuple[float, float, float, float],
              handwriting: bool, heading_path: list[str] | None = None) -> list[Block]:
    """Split transcription into paragraph/handwriting blocks and markdown tables."""
    blocks: list[Block] = []
    para: list[str] = []
    table: list[str] = []

    def flush_para() -> None:
        if para:
            text = " ".join(l.strip() for l in para).strip()
            if text:
                blocks.append(Block(
                    id=f"{id_prefix}:v{len(blocks)}", type="handwriting" if handwriting else "paragraph",
                    text=text, bbox=page_bbox, source="vlm", confidence=result.confidence,
                    heading_path=list(heading_path or []),
                ))
            para.clear()

    def flush_table() -> None:
        if table:
            rows = _parse_md_table(table)
            if rows:
                md = "| " + " | ".join(rows[0]) + " |\n|" + "---|" * len(rows[0]) + "\n"
                md += "\n".join("| " + " | ".join(r) + " |" for r in rows[1:])
                blocks.append(Block(
                    id=f"{id_prefix}:v{len(blocks)}", type="table", text=md, rows=rows, bbox=page_bbox,
                    source="vlm", confidence=result.confidence, heading_path=list(heading_path or []),
                ))
            table.clear()

    for line in result.text.splitlines():
        if line.strip().startswith("|"):
            flush_para()
            table.append(line)
        elif not line.strip():
            flush_para()
            flush_table()
        else:
            flush_table()
            para.append(line)
    flush_para()
    flush_table()
    return blocks
