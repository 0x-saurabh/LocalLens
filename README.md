# LocalLens

Fully offline, privacy-first RAG: ask questions over your PDFs, scans, tables and
handwritten notes. Zero data ever leaves your device.

Design: [docs/DESIGN.md](docs/DESIGN.md)

## Quick start

```bash
uv sync                       # create .venv with Python 3.12 and deps
uv run locallens doctor       # check Ollama, models, offline readiness
uv run locallens setup        # one-time: pull local models (needs internet ONCE)
uv run locallens ingest ~/Documents/some-folder   # PDFs, scans, PNG/JPG photos
uv run locallens ingest ~/Scans --vlm all-scans     # force the vision model on every scanned page
uv run locallens ask "What does chapter 3 say about normalization?"
uv run locallens chat         # interactive chat loop in the terminal
```

## Status

- Phase 0 (environment, offline guard) — done
- Phase 1 (digital PDFs: ingest, ask, chat, status) — done
- Phase 2 (scans and images: Apple Vision OCR, deskew, confidence, Tesseract fallback) — done
- Phase 3 (tables as row chunks, BM25 + vector hybrid retrieval) — done
- Phase 4 (handwriting and scanned tables via local vision model, `--vlm all-scans`) — done
- Phase 5 (reranker, eval harness) — next

Run the tests with `uv run pytest`. The end-to-end test needs Ollama running and is skipped otherwise.
