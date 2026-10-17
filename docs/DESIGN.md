# LocalLens — Design Document

> A fully offline, privacy-first RAG system that answers questions from your PDFs,
> scans, tables and handwritten notes. Zero data ever leaves your device.

Target machine (dev): Apple M4, 16 GB unified memory, macOS 26.

---

## 1. Goals and non-goals

**Goals**
- Ask natural-language questions over a folder of personal documents and get a
  grounded answer with page-level citations.
- Handle four input classes well: digital PDFs, scanned PDFs/images, tables
  (digital and scanned), handwritten notes.
- 100% offline after a one-time model download. Provable, not just promised
  (see §7).
- Run comfortably on a 16 GB laptop with no GPU server.

**Non-goals (v1)**
- Multi-user / server deployment.
- Real-time collaboration, cloud sync.
- Perfect handwriting recognition. We aim for "useful with confidence flags".
- Non-English documents (design for it, but don't test it in v1).

---

## 2. User experience (what "done" looks like)

```
$ locallens ingest ~/Documents/taxes ~/Scans/notebook-2025
  ▸ 42 files · 318 pages
  ▸ 211 digital · 84 scanned · 23 handwritten · 57 tables detected
  ▸ indexed in 6m 12s  (models: vision-ocr, qwen2.5vl:3b, bge-m3)

$ locallens ask "What was my total medical spend in 2024 and which receipts support it?"
  Total 2024 medical spend: $3,412.50 across 6 receipts.
  Sources:
    [1] receipts/2024-03-11-dentist.pdf  p.1  (table row: "Total  $820.00")
    [2] scans/pharmacy-2024.pdf          p.3  (OCR, conf 0.91)
    ...
  ⚠ One handwritten note (notebook p.14) mentions "~$200 copay" — low OCR confidence, not included.

$ locallens serve   # optional local web UI at http://127.0.0.1:7860
```

Key UX principles:
- Every answer cites document + page + the extracted snippet.
- Low-confidence extractions (handwriting, bad scans) are surfaced, never silently trusted.
- Re-running `ingest` is incremental: unchanged files are skipped (content hash).

---

## 2a. Interface decision (2026-10-18): CLI first, chat is a CLI command

LocalLens is built as a Python package whose only interface in v1 is the shell:

| Command             | What it does                                                   |
|---------------------|----------------------------------------------------------------|
| `locallens setup`   | one-time model download (the ONLY command allowed on the network) |
| `locallens doctor`  | verify Ollama, models, offline guard                           |
| `locallens ingest`  | index files/folders, incremental                               |
| `locallens ask`     | one question → cited answer, exits (scriptable, pipe-friendly) |
| `locallens chat`    | multi-turn chat loop in the terminal (this IS the chatbot)     |
| `locallens status`  | what is indexed                                                |

Why not a desktop app (Tauri + React) first:
- The hard, differentiating work is ingestion quality and retrieval. A UI adds
  zero to that and costs weeks of packaging, IPC and two-language plumbing.
- A CLI is testable, scriptable and lets you iterate on the pipeline daily.
- `chat` gives the conversational experience without any frontend.

The core stays UI-agnostic (`ingest/`, `retrieve.py`, `generate.py` expose plain
Python functions). When the pipeline is good, wrap the same package with a Tauri
or web shell bound to 127.0.0.1 (Phase 6). Nothing has to be rewritten.

Stack choices vs. the alternative plan (Tauri/FAISS/Tesseract/llama.cpp):
- Ollama instead of raw llama.cpp: same engine underneath, far less plumbing,
  and it serves LLM, embeddings and the VLM through one local API.
- LanceDB instead of FAISS: FAISS is vectors only; LanceDB gives vectors + BM25
  full-text in one embedded store, which hybrid retrieval needs anyway.
- Apple Vision OCR first, Tesseract as fallback: much better on real scans on
  this machine; Tesseract keeps the door open for Linux/Windows.
- LLM: `qwen3:8b` is already pulled locally, so Phase 1 uses it.

---

## 3. Architecture

```
                 ┌──────────────────────────────────────────────────────────┐
                 │                        INGEST                            │
 files ─────────▶│ 1. Discover + hash   2. Classify pages   3. Extract       │
                 │    (pdf/png/jpg)        digital/scan/hw     text/OCR/VLM  │
                 │ 4. Tables → markdown 5. Build Document model (provenance)│
                 │ 6. Chunk (structure-aware) 7. Embed + BM25 8. Store       │
                 └──────────────────────────┬───────────────────────────────┘
                                            │  LanceDB (vectors + FTS) + SQLite (metadata)
                 ┌──────────────────────────▼───────────────────────────────┐
                 │                         QUERY                            │
 question ──────▶│ 1. Hybrid retrieve (vector ∪ BM25, top-40)               │
                 │ 2. Rerank (cross-encoder, top-8)                         │
                 │ 3. Prompt assembly with numbered sources                 │
                 │ 4. Local LLM generation (streamed)                       │
                 │ 5. Citation verification (snippets must exist)          │
                 └──────────────────────────────────────────────────────────┘
```

### 3.1 Document model (the heart of the system)

Everything is normalized into this before chunking, so the retrieval side never
cares whether text came from a text layer, OCR, or a vision model.

```python
Document
  id, path, sha256, mime, page_count, ingested_at
  pages: list[Page]

Page
  number, width, height, kind: "digital" | "scanned" | "handwritten" | "mixed"
  blocks: list[Block]

Block
  id, type: "heading" | "paragraph" | "list" | "table" | "figure" | "handwriting"
  text                      # markdown for tables
  bbox (x0,y0,x1,y1)        # for later highlighting in UI
  source: "textlayer" | "ocr" | "vlm"
  confidence: float | None  # OCR/VLM only
  heading_path: list[str]   # e.g. ["Schedule A", "Medical Expenses"]
```

### 3.2 Page classification heuristic

For each PDF page:
1. Extract text layer with PyMuPDF. If chars/page > ~200 and the text is not
   garbage (ratio of alphanumerics, dictionary hit rate) → `digital`.
2. Else rasterize at 200–300 DPI and run fast OCR. If OCR yields printed-looking
   text with high mean confidence → `scanned`.
3. Else (low confidence, irregular line heights) → `handwritten` → route to VLM.
Images (png/jpg/heic) skip step 1.

### 3.3 Extraction routes

| Page kind      | Primary tool                                    | Fallback                     |
|----------------|-------------------------------------------------|------------------------------|
| digital text   | PyMuPDF text + Docling layout                   | pdfplumber                   |
| digital table  | Docling TableFormer → markdown                  | pdfplumber `extract_tables`  |
| scanned text   | Apple Vision OCR (`VNRecognizeTextRequest`, on-device, via pyobjc) | Tesseract / PaddleOCR |
| scanned table  | Docling (OCR-enabled) → markdown                | VLM transcribe to markdown   |
| handwriting    | Local VLM (Qwen2.5-VL 3B/7B via Ollama) transcribe | Apple Vision OCR (it handles neat handwriting surprisingly well) |

Why Apple Vision first on macOS: it is free, fast, fully on-device, and
noticeably better than Tesseract on real-world scans. Keep Tesseract as the
cross-platform fallback so Linux/Windows aren't blocked later.

Scan preprocessing before OCR: deskew, grayscale, adaptive threshold, light
denoise (OpenCV). Cheap and it lifts OCR accuracy a lot on phone photos.

### 3.4 Chunking (structure-aware, not fixed-size)

- Paragraph/list blocks: merge sequentially up to ~400 tokens, 15% overlap,
  never cross a heading boundary. Prepend `heading_path` to the chunk text
  (" > "-joined) so the embedding carries context.
- Tables: store **two** representations:
  1. The whole table as one markdown chunk (for "what does this table show").
  2. Row-level chunks, each with the header row repeated, e.g.
     `Date | Provider | Amount\n2024-03-11 | Dentist | $820.00` — this is what
     makes "total medical spend" questions work.
- Handwriting: chunk per note/page; attach `confidence`.
- Every chunk carries: `doc_id, page, block_ids, bbox_union, source, confidence,
  heading_path`.

### 3.5 Retrieval

- Embeddings: `bge-m3` (dense, multilingual, 8k ctx) via Ollama or
  sentence-transformers. Alternative lighter: `nomic-embed-text`.
- Lexical: BM25 via LanceDB full-text index (exact numbers, invoice IDs,
  names — vectors are weak on these).
- Fusion: Reciprocal Rank Fusion of both lists, top-40.
- Rerank: `BAAI/bge-reranker-v2-m3` cross-encoder → top-8.
- Optional: query rewriting by the LLM for multi-hop questions (v2).

### 3.6 Generation

- LLM via Ollama: `qwen2.5:7b-instruct` or `llama3.1:8b` at Q4_K_M (~5 GB).
  Start with a ~7–8B model; it is the sweet spot on 16 GB. Keep context ≤ 8k.
- Prompt: system rules (answer only from sources, cite as [n], say "not found"
  if unsupported) + numbered sources + question. Stream output.
- Post-check: every `[n]` must refer to a provided source; any quoted number
  should appear in some source chunk — otherwise append a warning. This kills
  most citation hallucinations cheaply.

### 3.7 Storage layout

```
~/.locallens/
  config.toml
  models/            # HF models, Ollama handles its own cache
  index/
    lancedb/         # chunks table: vector + text + metadata (+ FTS index)
    meta.sqlite      # documents, pages, blocks, ingest log, hashes
    pages/           # rasterized page PNGs for UI highlighting (optional)
```

Optional (v2): encrypt `index/` at rest with SQLCipher / an `age`-encrypted
volume for the "laptop gets stolen" threat model.

---

## 4. Tech stack

| Layer          | Choice                                   | Why                                          |
|----------------|------------------------------------------|----------------------------------------------|
| Language       | Python 3.12 (via `uv`)                   | ML wheel ecosystem. **Not** 3.15 alpha.      |
| PDF            | PyMuPDF                                  | Fastest text + rasterize                     |
| Layout/tables  | Docling                                  | Offline layout model + TableFormer           |
| OCR            | Apple Vision (pyobjc) / Tesseract        | On-device, accurate / portable               |
| VLM            | Ollama `qwen2.5vl:3b` (7b if RAM allows) | Handwriting + messy tables                   |
| Embeddings     | bge-m3 (Ollama or sentence-transformers) | Strong, long context                         |
| Vector + FTS   | LanceDB                                  | Embedded, no server, hybrid search built in  |
| Metadata       | SQLite                                   | Zero-ops                                     |
| Reranker       | bge-reranker-v2-m3                       | Big precision lift for little cost           |
| LLM            | Ollama (qwen2.5:7b / llama3.1:8b)        | Easy local serving, Metal accelerated        |
| CLI            | Typer + Rich                             | Fast to build, pleasant                      |
| UI (later)     | Gradio or Streamlit, bound to 127.0.0.1  | Local-only web UI                            |
| Tests          | pytest + small fixture corpus            |                                              |

Memory budget (16 GB): never hold LLM + VLM + embedder in RAM at once.
Ingest phase uses VLM + embedder; query phase uses embedder + reranker + LLM.
Ollama unloads idle models automatically (`OLLAMA_KEEP_ALIVE`), so sequence
the work rather than parallelize it.

---

## 5. Repository layout

```
locallens/
  pyproject.toml
  README.md
  docs/DESIGN.md
  src/locallens/
    __init__.py
    config.py          # paths, model names, thresholds (TOML)
    models.py          # Document / Page / Block / Chunk dataclasses
    ingest/
      discover.py      # walk dirs, hash, detect mime
      classify.py      # page kind heuristic
      pdf_text.py      # PyMuPDF text layer
      ocr_vision.py    # Apple Vision OCR
      ocr_tesseract.py # fallback
      preprocess.py    # deskew / threshold (OpenCV)
      tables.py        # Docling / pdfplumber → markdown
      vlm.py           # Ollama vision transcription
      pipeline.py      # orchestrates the above into Documents
    chunk.py
    embed.py
    store.py           # LanceDB + SQLite
    retrieve.py        # hybrid + RRF + rerank
    generate.py        # prompt, Ollama chat, citation check
    offline_guard.py   # see §7
    cli.py             # ingest / ask / status / serve
    ui.py              # later
  tests/
    fixtures/          # 1 digital pdf, 1 scan, 1 table pdf, 1 handwriting jpg
    test_*.py
  data/                # gitignored sample docs
```

---

## 6. Build plan (phased, each phase ships something usable)

**Phase 0 — Environment** ✅ done 2026-10-18
- `brew install uv tesseract`; `uv python install 3.12`; `uv init`.
- `ollama pull qwen2.5:7b-instruct nomic-embed-text qwen2.5vl:3b` (or bge-m3).
- Download HF models once (reranker, Docling models), then set
  `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.
- Write `offline_guard.py` and a test that proves no socket opens (see §7).

**Phase 1 — Digital PDFs end-to-end** ✅ done 2026-10-18 (`ingest`, `ask`, `chat`, `status` work; vector-only retrieval; a 36-page real PDF indexes in ~6 s, answers take ~30–50 s on M4 with qwen3:8b)
- discover → PyMuPDF text → simple heading-aware chunking → embed → LanceDB.
- `locallens ask` with vector-only retrieval and cited answers.
- Success: correct, cited answers on 3–5 of your own digital PDFs.


> **Known limitation after Phase 1 (measured 2026-10-18):** vector-only retrieval
> misses lexically specific questions. Example: "List the four mechanisms that
> replace ADL" — the chunk that names them (page 3, abstract) ranked 21st of 40
> by cosine similarity, outside the top-8 context window, so the LLM correctly
> answered "not in sources". This is the concrete motivation for BM25 hybrid
> retrieval (Phase 3) and the cross-encoder reranker (Phase 5). Pull those
> forward if answer quality matters more than scans right now.

**Phase 2 — Scans** ✅ done 2026-10-18 (Apple Vision OCR via pyobjc, Tesseract fallback
behind `uv sync --extra tesseract`; OpenCV denoise + deskew; page kinds digital/scanned/handwritten
by text-layer density then OCR confidence; images .png/.jpg/.tif/.bmp/.gif ingest like PDFs;
a real 1-page scanned certificate OCR'd at 0.98 mean confidence in ~15 s incl. model warm-up.
Chunks now carry the document name in their context prefix so "the X document" questions retrieve.)
- Page classification, OpenCV preprocessing, Apple Vision OCR with confidence,
  Tesseract fallback. Store `source` + `confidence` on blocks.
- Success: a phone-photographed receipt and a scanned letter are answerable.

**Phase 3 — Tables + hybrid retrieval** ✅ done 2026-10-18
- Digital tables via PyMuPDF `find_tables` (no Docling dependency yet): leading/trailing
  single-cell rows trimmed, bbox shrunk to kept rows so neighbouring prose stays prose;
  junk/stub tables rejected. 17 real tables found in the 36-page whitepaper, 0 false positives.
- Each table → one whole-table markdown chunk + one chunk per row with `header: value` pairs.
  Aggregation ("total travel cost") and exact lookup ("electricity in March") both verified.
- BM25 via LanceDB native FTS, fused with dense by RRF, with the top-4 of each retriever
  guaranteed a seat (plain RRF buried a rank-4 BM25 hit at fused rank 30). Context is 10 chunks.
- Known limits: whitespace-aligned tables without ruling lines sometimes merge columns into
  the first cell (text still searchable); scanned tables are plain OCR text until the VLM
  route (Phase 4); Docling remains the upgrade path for layout-heavy PDFs.
- The "four mechanisms" recall failure from Phase 1 is fixed, but only because the chunk
  was BM25 rank 4 — a cross-encoder reranker (Phase 5) is the robust fix, not wider k.
- Docling integration; markdown tables; whole-table + row-level chunks with
  repeated headers. BM25 added for exact-value matching; RRF fusion.
- Success: "what is the amount on row X", and simple aggregation questions.

**Phase 4 — Handwriting + scanned tables via VLM** ✅ done 2026-10-18
- `qwen2.5vl:3b` through Ollama. Strict transcription prompt; `[?]` for unreadable words;
  Markdown tables in the output become table blocks → whole-table + row chunks (same path as
  digital tables). Images are downscaled to ≤1280 px on the long side (a 150 DPI page blew the
  4k context; errors now surface in the ingest log instead of silently falling back to OCR).
- Confidence: `(1 − unreadable_ratio) × 0.85`, lifted when OCR agrees >80 %. VLM text is
  capped at 0.85 and carries `source="vlm"` so answers can flag it.
- Routing (`--vlm`, or `vlm_mode` in config): `handwriting` (default: only pages whose OCR
  mean confidence < 0.6), `all-scans` (every non-digital page), `off`.
- Measured: handwritten-style note 1 page ≈ 80 s incl. model load; dense scanned certificate
  1 page ≈ 5 min (long output at 3B on M4) → 2 tables, 10 row chunks, correct headers.
- ⚠ Unvalidated: the 0.6 auto-routing threshold. Apple Vision reports 1.0 confidence even on
  cursive *fonts* (Snell Roundhand), so synthetic tests cannot trigger the handwriting route;
  a real handwriting sample is needed. Until then `--vlm all-scans` is the reliable switch.
- The VLM is unloaded (`keep_alive=0`) after ingest so the 8B chat model fits in 16 GB.
- VLM route via Ollama with a strict transcription prompt ("transcribe exactly,
  mark unreadable words as [?]"). Confidence = VLM self-report + OCR agreement.
- Surface low-confidence sources in answers.
- Success: readable notebook pages are searchable; messy ones are flagged.

**Phase 5 — Quality (ongoing)**
- Reranker; eval harness with a golden Q&A set (~30 questions across the four
  doc types); metrics: retrieval hit@8, answer correctness (manual/LLM-judged
  locally), citation validity rate.
- Incremental ingest (hash skip), delete/re-index, `locallens status`.

**Phase 6 — UX & packaging**
- Local web UI with page-image preview and highlighted bbox of the cited block.
- Watch-folder mode. `pipx`/`uv tool` install. Optional at-rest encryption.

---

## 7. Privacy guarantee — make it provable

"Zero data leaves your device" must be enforceable, not aspirational:

1. **Offline guard**: at startup, `offline_guard.enable()` monkeypatches
   `socket.socket.connect` to raise unless the target is `127.0.0.1`/`::1`
   (Ollama). Run in all modes except the explicit `locallens setup` download step.
2. **Env**: `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `OLLAMA_HOST=127.0.0.1`.
3. **Test**: `tests/test_offline.py` ingests a fixture and asks a question with
   a network-denying socket patch; any outbound attempt fails the suite.
4. **Audit mode**: `locallens doctor` prints every model path and confirms they
   resolve locally; prints the bound UI address.
5. No telemetry, no crash reporting, no update checks. Ever.

---

## 8. Known hard parts and how we'll handle them

| Risk                                      | Mitigation                                                    |
|-------------------------------------------|---------------------------------------------------------------|
| Handwriting accuracy                      | VLM route + confidence flags + "not confident" in answers      |
| VLM speed on 16 GB (10–30 s/page for 7B)  | Default to 3B VLM; only route truly handwritten pages to it    |
| RAM pressure                              | Sequential model use; Q4 quants; Ollama keep-alive tuning      |
| Tables losing header context              | Row-chunks with repeated header; whole-table chunk too         |
| Numbers/IDs missed by vectors             | BM25 hybrid                                                   |
| Citation hallucination                    | Post-generation citation/number verification                   |
| Python 3.15 alpha breaks wheels           | Pin 3.12 via uv                                               |
| Docling model download at runtime         | Pre-fetch in `setup`, then run offline                         |

---

## 9. Stretch ideas (after v1)

- Multi-turn chat with conversation memory (local SQLite).
- Per-folder "collections" and permissions.
- Image/figure Q&A via the VLM at query time.
- MLX backend (Apple-native) instead of Ollama for faster VLM inference.
- Cross-platform packaging (Linux/Windows with Tesseract/PaddleOCR).
