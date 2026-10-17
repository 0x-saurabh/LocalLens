"""PDF/image extraction via PyMuPDF: text-layer blocks with heading detection,
and OCR for pages whose text layer is too thin (scans, photos, handwriting).
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
import pymupdf

from locallens.ingest import classify
from locallens.ingest.ocr import OcrEngine, lines_to_blocks
from locallens.ingest.preprocess import preprocess
from locallens.ingest.vlm import Transcriber, to_blocks
from locallens.models import Block, Document, Page

OCR_DPI = 200

HEADING_SIZE_RATIO = 1.15
HEADING_MAX_CHARS = 120
_WS = re.compile(r"[ \t ]+")
# "S E C T I O N 0 3" -> "SECTION 03": runs of 4+ single characters separated by spaces
_SPACED = re.compile(r"\b(?:\S ){3,}\S\b")


def _collapse_spaced(text: str) -> str:
    def join(m: re.Match) -> str:
        run = m.group(0).replace(" ", "")
        return re.sub(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])", " ", run)
    return _SPACED.sub(join, text)


def _clean(text: str) -> str:
    text = text.replace("­", "")              # soft hyphen
    text = re.sub(r"-\n(?=[a-z])", "", text)       # de-hyphenate line breaks
    text = text.replace("\n", " ")
    return _collapse_spaced(_WS.sub(" ", text).strip())


def _block_font_size(block: dict) -> float:
    sizes = [span["size"] for line in block.get("lines", []) for span in line.get("spans", [])]
    return statistics.median(sizes) if sizes else 0.0


def _block_text(block: dict) -> str:
    return "\n".join(
        "".join(span["text"] for span in line.get("spans", [])) for line in block.get("lines", [])
    )


def _rasterize(page: pymupdf.Page, dpi: int) -> np.ndarray:
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w)


def ocr_page(page: pymupdf.Page, pno: int, doc_id: str, ocr: OcrEngine, dpi: int = OCR_DPI) -> tuple[list[Block], float | None]:
    """Rasterize → preprocess → OCR → blocks in PDF point coordinates. Returns (blocks, mean_conf)."""
    pre = preprocess(_rasterize(page, dpi))
    lines = ocr(pre.image)
    blocks = lines_to_blocks(lines, id_prefix=f"{doc_id[:12]}:p{pno}", scale=72.0 / dpi)
    conf = statistics.fmean(l.confidence for l in lines) if lines else None
    return blocks, conf


def _cell(v) -> str:
    return _clean(str(v)) if v is not None else ""


def _filled(row: list[str]) -> int:
    return sum(bool(c) for c in row)


def find_tables(page: pymupdf.Page, pno: int, doc_id: str, heading_path: list[str]) -> list[Block]:
    """Digital tables via PyMuPDF's finder.

    The finder tends to over-extend the table region into surrounding prose, which shows
    up as leading/trailing rows with a single huge cell. We trim those rows and shrink the
    bbox to the kept rows, so the prose stays with the paragraph blocks.
    """
    blocks: list[Block] = []
    try:
        found = page.find_tables()
    except Exception:  # noqa: BLE001  (finder can choke on odd pages)
        return blocks
    for ti, t in enumerate(found.tables):
        raw = [[_cell(c) for c in r] for r in t.extract()]
        keep = list(range(len(raw)))
        while keep and _filled(raw[keep[0]]) <= 1:
            keep.pop(0)
        while keep and _filled(raw[keep[-1]]) <= 1:
            keep.pop()
        rows = [raw[i] for i in keep if any(raw[i])]
        if len(rows) < 2 or len(rows[0]) < 2 or sum(_filled(r) >= 2 for r in rows) < 2:
            continue  # caption fragments, decorative rules, or a single-row stub
        if sum(len(c) for r in rows for c in r) < 20:
            continue  # junk like "sm _ / ts"
        try:
            row_boxes = [t.rows[i].bbox for i in keep]
            bbox = (min(b[0] for b in row_boxes), min(b[1] for b in row_boxes),
                    max(b[2] for b in row_boxes), max(b[3] for b in row_boxes))
        except (AttributeError, IndexError):
            bbox = tuple(t.bbox)
        if keep and keep[0] == 0 and t.header.external:
            names = [_cell(n) for n in t.header.names]
            # only trust an external header if most cells are filled (otherwise it's a caption/heading)
            if len(names) == len(rows[0]) and _filled(names) >= max(2, len(names) // 2 + 1):
                rows.insert(0, names)
        md = "| " + " | ".join(rows[0]) + " |\n|" + "---|" * len(rows[0]) + "\n"
        md += "\n".join("| " + " | ".join(r) + " |" for r in rows[1:])
        blocks.append(Block(
            id=f"{doc_id[:12]}:p{pno}:t{ti}", type="table", text=md, rows=rows,
            bbox=bbox, source="textlayer", heading_path=list(heading_path),
        ))
    return blocks


def _inside(inner, outer, tol: float = 2.0) -> bool:
    return (inner[0] >= outer[0] - tol and inner[1] >= outer[1] - tol
            and inner[2] <= outer[2] + tol and inner[3] <= outer[3] + tol)


def vlm_page(page: pymupdf.Page, pno: int, doc_id: str, vlm: Transcriber, ocr_blocks: list[Block],
             handwriting: bool, dpi: int = 150) -> list[Block]:
    """Transcribe a page with the vision model; falls back to the OCR blocks on failure/empty output."""
    img = _rasterize(page, dpi)
    ocr_text = " ".join(b.text for b in ocr_blocks) or None
    try:
        result = vlm.transcribe(img, ocr_text)
    except Exception as exc:  # noqa: BLE001
        if hasattr(vlm, "errors"):
            vlm.errors.append(f"page {pno}: {exc.__class__.__name__}: {str(exc)[:160]}")
        return ocr_blocks
    blocks = to_blocks(result, id_prefix=f"{doc_id[:12]}:p{pno}", page_bbox=tuple(page.rect),
                       handwriting=handwriting)
    return blocks or ocr_blocks


def extract(path: Path, doc_id: str, mime: str, ocr: OcrEngine | None = None,
            vlm: Transcriber | None = None, vlm_mode: str = "handwriting") -> Document:
    pdf = pymupdf.open(path)
    document = Document(id=doc_id, path=str(path), mime=mime, page_count=pdf.page_count)
    heading_path: list[str] = []

    # body font size across the whole document, so headings are judged consistently
    chars_by_size: Counter[float] = Counter()
    raw_pages: list[list[dict]] = []
    for page in pdf:
        blocks = [b for b in page.get_text("dict")["blocks"] if b.get("type") == 0]
        raw_pages.append(blocks)
        for b in blocks:
            n = len(_block_text(b).strip())
            if n:
                chars_by_size[round(_block_font_size(b), 1)] += n
    # body size = the font size that carries the most characters (headings are few chars)
    body_size = chars_by_size.most_common(1)[0][0] if chars_by_size else 10.0

    for pno, (page, blocks) in enumerate(zip(pdf, raw_pages), start=1):
        p = Page(number=pno, width=page.rect.width, height=page.rect.height)
        text_chars = sum(len(_block_text(b).strip()) for b in blocks)
        if classify.classify_text_layer(text_chars) != "digital":
            p.kind = "scanned"
            if ocr is not None:
                p.blocks, conf = ocr_page(page, pno, doc_id, ocr)
                p.kind = classify.refine_after_ocr(conf, len(p.blocks))
            if vlm is not None and vlm_mode != "off" and (vlm_mode == "all-scans" or p.kind == "handwritten"):
                p.blocks = vlm_page(page, pno, doc_id, vlm, p.blocks, handwriting=(p.kind == "handwritten"))
            document.pages.append(p)
            continue

        page_start_path = list(heading_path)
        tables = find_tables(page, pno, doc_id, heading_path)
        table_boxes = [t.bbox for t in tables]
        page_blocks: list[Block] = list(tables)
        for i, b in enumerate(blocks):
            if any(_inside(b["bbox"], tb) for tb in table_boxes):
                continue  # cell text already captured by the table block
            text = _clean(_block_text(b))
            if not text:
                continue
            size = _block_font_size(b)
            is_heading = (
                size >= body_size * HEADING_SIZE_RATIO
                and len(text) <= HEADING_MAX_CHARS
                and not text.endswith((".", ",", ";"))
            )
            if is_heading:
                # bigger font → shallower level. Keep it simple: replace path at a depth by size.
                level = 0 if size >= body_size * 1.6 else 1 if size >= body_size * 1.3 else 2
                heading_path = heading_path[:level] + [text]
            page_blocks.append(
                Block(
                    id=f"{doc_id[:12]}:p{pno}:b{i}",
                    type="heading" if is_heading else "paragraph",
                    text=text,
                    bbox=tuple(b["bbox"]),
                    source="textlayer",
                    heading_path=list(heading_path),
                )
            )
        # reading order: top-to-bottom by bbox, tables interleaved with text
        p.blocks = sorted(page_blocks, key=lambda b: (round(b.bbox[1]), b.bbox[0]) if b.bbox else (0, 0))
        # tables inherit the heading path of the nearest text block above them
        running = page_start_path
        for b in p.blocks:
            if b.type == "table":
                b.heading_path = list(running)
            else:
                running = b.heading_path
        document.pages.append(p)

    pdf.close()
    return document
