"""Find candidate files, hash them, detect type."""

from __future__ import annotations

import hashlib
import mimetypes
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

SUPPORTED = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}  # all opened by PyMuPDF


@dataclass(frozen=True)
class Candidate:
    path: Path
    sha256: str
    mime: str


def sha256_of(path: Path, bufsize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def discover(paths: list[Path]) -> Iterator[Candidate]:
    for root in paths:
        files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
        for p in files:
            if p.suffix.lower() not in SUPPORTED or p.name.startswith("."):
                continue
            mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
            yield Candidate(path=p.resolve(), sha256=sha256_of(p), mime=mime)
