"""Structure-aware chunking over the document model.

Rules: never cross a heading boundary, keep tables whole (Phase 3), prepend the
heading path so each chunk carries its context, overlap by a sentence tail.
"""

from __future__ import annotations

import re
from pathlib import Path

from locallens.models import Block, Chunk, Document, Page

CHARS_PER_TOKEN = 4
_SENT_END = re.compile(r"(?<=[.!?])\s+")


def _tail(text: str, max_chars: int) -> str:
    """Last few sentences of `text`, at most `max_chars` long."""
    sentences = _SENT_END.split(text)
    out: list[str] = []
    total = 0
    for s in reversed(sentences):
        if total + len(s) > max_chars:
            break
        out.insert(0, s)
        total += len(s) + 1
    return " ".join(out)


def _split_oversized(block: Block, max_chars: int) -> list[Block]:
    """Split a single block longer than max_chars into sentence-aligned pieces."""
    if len(block.text) <= max_chars or block.type == "table":
        return [block]
    pieces: list[Block] = []
    buf: list[str] = []
    size = 0
    for sent in _SENT_END.split(block.text):
        if size + len(sent) > max_chars and buf:
            pieces.append(block.model_copy(update={"id": f"{block.id}.{len(pieces)}", "text": " ".join(buf)}))
            buf, size = [], 0
        buf.append(sent)
        size += len(sent) + 1
    if buf:
        pieces.append(block.model_copy(update={"id": f"{block.id}.{len(pieces)}", "text": " ".join(buf)}))
    return pieces


def _table_chunks(doc: Document, page: int, block: Block, start: int) -> list[Chunk]:
    """One chunk for the whole table, plus one per data row with the header repeated.

    Row chunks are what make "what was the amount for X" and aggregation questions work.
    """
    prefix = "[" + " > ".join([Path(doc.path).stem, *block.heading_path]) + "]\n"
    common = {"doc_id": doc.id, "page": page, "block_ids": [block.id], "source": block.source,
              "confidence": block.confidence, "heading_path": list(block.heading_path)}
    out = [Chunk(id=f"{doc.id[:12]}:p{page}:c{start}", text=prefix + block.text, **common)]
    rows = block.rows or []
    if len(rows) > 2:
        header = rows[0]
        for ri, row in enumerate(rows[1:], start=1):
            pairs = "; ".join(f"{h}: {v}" for h, v in zip(header, row) if v)
            text = f"{prefix}Table row {ri}: {pairs}"
            out.append(Chunk(id=f"{doc.id[:12]}:p{page}:c{start}:r{ri}", text=text, **common))
    return out


def chunk_document(doc: Document, chunk_tokens: int = 400, overlap: float = 0.15) -> list[Chunk]:
    max_chars = chunk_tokens * CHARS_PER_TOKEN
    overlap_chars = int(max_chars * overlap)
    chunks: list[Chunk] = []

    for page in doc.pages:
        buf: list[Block] = []
        buf_len = 0
        carry = ""  # overlap text from the previous chunk on this page
        current_path: list[str] | None = None

        def flush(page: Page = page) -> None:  # bind the loop variable
            nonlocal buf, buf_len, carry
            if not buf:
                return
            body = " ".join(b.text for b in buf)
            # document name + heading path give the embedder context ("the cgbse document", "chapter 3")
            prefix = " > ".join([Path(doc.path).stem, *buf[0].heading_path])
            text = f"[{prefix}]\n" + (f"{carry} " if carry else "") + body
            confs = [b.confidence for b in buf if b.confidence is not None]
            chunks.append(
                Chunk(
                    id=f"{doc.id[:12]}:p{page.number}:c{len(chunks)}",
                    doc_id=doc.id,
                    page=page.number,
                    text=text,
                    block_ids=[b.id for b in buf],
                    source=buf[0].source,
                    confidence=min(confs) if confs else None,
                    heading_path=list(buf[0].heading_path),
                )
            )
            carry = _tail(body, overlap_chars)
            buf, buf_len = [], 0

        blocks = [piece for b in page.blocks for piece in _split_oversized(b, max_chars)]
        for block in blocks:
            if block.type == "table":
                flush()
                carry = ""
                chunks.extend(_table_chunks(doc, page.number, block, len(chunks)))
                continue
            if current_path is not None and block.heading_path != current_path:
                flush()
                carry = ""                      # no overlap across headings
            current_path = block.heading_path
            if buf_len + len(block.text) > max_chars and buf:
                flush()
            buf.append(block)
            buf_len += len(block.text) + 1
        flush()

    return chunks
