"""The document model. Every extraction route (text layer, OCR, VLM) produces this."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

PageKind = Literal["digital", "scanned", "handwritten", "mixed"]
BlockType = Literal["heading", "paragraph", "list", "table", "figure", "handwriting"]
Source = Literal["textlayer", "ocr", "vlm"]


class Block(BaseModel):
    id: str
    type: BlockType
    text: str                      # markdown for tables
    rows: list[list[str]] | None = None   # tables only: header row first
    bbox: tuple[float, float, float, float] | None = None
    source: Source = "textlayer"
    confidence: float | None = None
    heading_path: list[str] = Field(default_factory=list)


class Page(BaseModel):
    number: int                    # 1-based
    width: float
    height: float
    kind: PageKind = "digital"
    blocks: list[Block] = Field(default_factory=list)


class Document(BaseModel):
    id: str                        # sha256 of file bytes
    path: str
    mime: str
    page_count: int
    pages: list[Page] = Field(default_factory=list)


class Chunk(BaseModel):
    id: str
    doc_id: str
    page: int
    text: str
    block_ids: list[str]
    source: Source
    confidence: float | None = None
    heading_path: list[str] = Field(default_factory=list)
