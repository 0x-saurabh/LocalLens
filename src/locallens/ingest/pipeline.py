"""Orchestrates discover → extract → chunk → embed → store."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from locallens import config
from locallens.chunk import chunk_document
from locallens.embed import Embedder
from locallens.ingest import discover as disc
from locallens.ingest import ocr as ocr_mod
from locallens.ingest import pdf_text
from locallens.ingest.vlm import Transcriber
from locallens.store import Store


@dataclass
class IngestReport:
    files_seen: int = 0
    files_indexed: int = 0
    files_skipped_unchanged: int = 0
    files_replaced: int = 0
    pages_digital: int = 0
    pages_ocr: int = 0
    pages_handwritten: int = 0
    pages_vlm: int = 0
    vlm_model: str | None = None
    ocr_engine: str | None = None
    chunks: int = 0
    tables: int = 0
    errors: list[tuple[Path, str]] = field(default_factory=list)


def ingest(paths: list[Path], cfg: config.Config, store: Store, embedder: Embedder,
           log: Callable[[str], None] = lambda s: None, force: bool = False,
           vlm_mode: str | None = None) -> IngestReport:
    rep = IngestReport()
    engine = None
    try:
        rep.ocr_engine, engine = ocr_mod.get_engine(cfg.ocr_engine)
    except RuntimeError as exc:
        log(f"[yellow]OCR unavailable:[/] {exc} — scanned pages will be skipped")
    vlm_mode = vlm_mode or cfg.vlm_mode
    vlm = None
    if vlm_mode != "off":
        t = Transcriber(cfg)
        if t.available():
            vlm, rep.vlm_model = t, t.model
        else:
            log(f"[yellow]VLM {t.model} not pulled[/] — handwriting will keep plain OCR text (run `locallens setup`)")
    for cand in disc.discover(paths):
        rep.files_seen += 1
        if store.has_document(cand.sha256) and not force:
            rep.files_skipped_unchanged += 1
            continue
        old_id = store.doc_id_for_path(str(cand.path))
        if old_id and old_id != cand.sha256:
            store.delete_document(old_id)       # same path, new content → replace the stale version
            rep.files_replaced += 1
        try:
            doc = pdf_text.extract(cand.path, cand.sha256, cand.mime, ocr=engine, vlm=vlm, vlm_mode=vlm_mode)
            chunks = chunk_document(doc, cfg.retrieval.chunk_tokens, cfg.retrieval.chunk_overlap)
            vectors = embedder.embed_passages([c.text for c in chunks]) if chunks else []
            if force:
                store.delete_document(cand.sha256)
            store.add_chunks(chunks, vectors)
            digital = sum(p.kind == "digital" for p in doc.pages)
            handwritten = sum(p.kind == "handwritten" for p in doc.pages)
            vlm_pages = sum(any(b.source == "vlm" for b in p.blocks) for p in doc.pages)
            rep.pages_vlm += vlm_pages
            scanned = doc.page_count - digital
            confs = [b.confidence for p in doc.pages for b in p.blocks if b.confidence is not None]
            mean_conf = sum(confs) / len(confs) if confs else None
            store.record_document(cand.sha256, str(cand.path), cand.mime, doc.page_count,
                                  digital, scanned, len(chunks))
            rep.files_indexed += 1
            rep.pages_digital += digital
            rep.pages_ocr += scanned
            rep.pages_handwritten += handwritten
            rep.chunks += len(chunks)
            n_tables = sum(b.type == "table" for p in doc.pages for b in p.blocks)
            rep.tables += n_tables
            detail = f"{digital} digital" + (f", {n_tables} tables" if n_tables else "")
            if scanned:
                detail += f", {scanned} OCR" + (f" (conf {mean_conf:.2f})" if mean_conf is not None else " (no text found)")
            if handwritten:
                detail += f", {handwritten} handwritten"
            if vlm_pages:
                detail += f", {vlm_pages} via VLM"
            log(f"indexed {cand.path.name}: {detail}, {len(chunks)} chunks")
            if vlm is not None and vlm.errors:
                for e in vlm.errors:
                    log(f"  [yellow]VLM fell back to OCR[/] {e}")
                vlm.errors.clear()
        except Exception as exc:  # noqa: BLE001
            rep.errors.append((cand.path, f"{exc.__class__.__name__}: {exc}"))
            log(f"[red]failed[/] {cand.path.name}: {exc}")
    if rep.files_indexed:
        store.rebuild_fts()
    if vlm is not None:
        vlm.unload()
    return rep
