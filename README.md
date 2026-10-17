# LocalLens

Ask questions about your own documents, entirely on your own machine.

LocalLens indexes PDFs, scanned pages, phone photos, tables and handwritten notes, then
answers natural-language questions with page-level citations. Nothing is uploaded anywhere.
After a one-time model download, the tool blocks every outbound network connection while
it runs, so "offline" is enforced rather than promised.

```
$ locallens ask "What was the total travel cost in the March expense report?"

╭─ answer ─────────────────────────────────────────────────────────────────╮
│ The total travel cost in March was $13,950 [1].                          │
╰──────────────────────────────────────────────────────────────────────────╯
                                 sources
  #   document                 page   snippet
 ● 1  expense_report.pdf          2   Table row 7: Category: Travel; Amount: 13,950
 ○ 2  expense_report.pdf          1   Monthly expense summary for March …
```

## What it handles

| Input | How it is read |
|---|---|
| Digital PDFs | Text layer, heading-aware chunks |
| Scanned PDFs and images (`.png .jpg .tif .bmp .gif`) | Deskew and denoise, then on-device OCR with per-block confidence |
| Tables, digital or scanned | Stored whole and as one chunk per row, so lookups and totals work |
| Handwritten notes | Local vision model transcribes them; unreadable words are marked `[?]` and the answer flags low confidence |

Retrieval is hybrid: vector search plus BM25 keyword search, fused and then reranked by a
cross-encoder. Keyword search is what makes exact numbers, invoice IDs and names findable.

## Requirements

- macOS on Apple Silicon is the tested platform. 16 GB of memory is enough.
  Linux should work with the Tesseract extra (see below) but is untested.
- [Ollama](https://ollama.com) installed and running. It serves the language, embedding
  and vision models locally.
- [uv](https://docs.astral.sh/uv/) to install Python 3.12 and the dependencies.
- Roughly 8 GB of disk for the default models.

## Install

```bash
git clone https://github.com/0x-saurabh/LocalLens.git
cd LocalLens
uv sync                  # creates .venv with Python 3.12 and all dependencies
uv run locallens setup   # one-time model download; the only command that uses the network
uv run locallens doctor  # confirms Ollama, models and the offline guard
```

`setup` pulls `qwen3:8b` (answers), `nomic-embed-text` (embeddings) and `qwen2.5vl:3b`
(handwriting and scanned tables) through Ollama, plus the `bge-reranker-v2-m3` reranker
from Hugging Face. It also writes `~/.locallens/config.toml`.

On Linux, or if you prefer Tesseract over Apple Vision for OCR, install the `tesseract`
binary with your package manager and run `uv sync --extra tesseract`.

## Usage

Index a folder or individual files. Re-running is incremental: unchanged files are skipped.

```bash
uv run locallens ingest ~/Documents/taxes ~/Scans/notebook-2025
uv run locallens ingest ~/Scans --vlm all-scans   # run the vision model on every scanned page
uv run locallens ingest ~/Docs --force            # re-index even if files are unchanged
```

Ask one question and exit, or open a chat loop:

```bash
uv run locallens ask "Which receipts support the 2024 medical total?"
uv run locallens ask --no-rerank "..."            # skip the cross-encoder, faster but less precise
uv run locallens chat                             # multi-turn; type /quit to leave
```

See and manage what is indexed:

```bash
uv run locallens status                           # documents, page kinds, chunk counts
uv run locallens remove ~/Documents/old.pdf       # drop a document from the index
```

Every answer lists its sources with document, page and the matching snippet. A filled
marker means the answer cited that source. If the model cites a source that does not
exist, LocalLens says so under the answer instead of letting it pass.

### The `--vlm` switch

Scanned pages go through OCR first. The vision model is slower, so by default it only
runs on pages whose OCR confidence is low, which is where handwriting ends up.

| Value | When the vision model runs |
|---|---|
| `handwriting` (default) | Only pages with mean OCR confidence below 0.6 |
| `all-scans` | Every non-digital page. Use this for scanned tables and messy scans. |
| `off` | Never. OCR text only. |

Expect roughly a minute per handwritten page and several minutes for a dense scanned
table on a 16 GB laptop. Digital PDFs index in seconds.

## Configuration

Everything lives under `~/.locallens` (override with the `LOCALLENS_HOME` environment
variable). `setup` writes `config.toml` there with these defaults:

```toml
ollama_host = "http://127.0.0.1:11434"
ocr_engine = "auto"          # auto | vision | tesseract
vlm_mode = "handwriting"     # handwriting | all-scans | off

[models]
llm = "qwen3:8b"
embed = "nomic-embed-text"
vlm = "qwen2.5vl:3b"
reranker = "BAAI/bge-reranker-v2-m3"

[retrieval]
chunk_tokens = 400
chunk_overlap = 0.15
top_k_candidates = 40
top_k_final = 10
guaranteed_per_retriever = 4
rerank = true
```

To use a different model, change the name here and run `setup` again to pull it. The
index itself is stored in `~/.locallens/index` as a LanceDB table plus a SQLite file.
Delete that folder to start over.

## Privacy guarantee

- Every command except `setup` starts by patching the socket layer so that any connection
  to an address other than `127.0.0.1` or `::1` fails. Ollama runs on localhost, so it
  still works; nothing else can reach out.
- Hugging Face and Transformers run in offline mode.
- There is no telemetry, crash reporting or update check.
- `tests/test_offline.py` ingests a document and asks a question with the guard active,
  so any outbound attempt fails the test suite.

## Troubleshooting

- **`doctor` says Ollama is unreachable.** Start it with `ollama serve` or open the
  Ollama app, then rerun `doctor`.
- **"Nothing indexed yet."** Run `ingest` on a folder first, then `status` to confirm.
- **A scanned table came back as plain text.** Re-ingest that file with `--vlm all-scans`
  and `--force`.
- **Answers are slow.** First answers include model load time. `--no-rerank` saves a few
  seconds. A smaller LLM in `config.toml` helps on tight memory.
- **No OCR engine found.** On macOS make sure `uv sync` installed the Vision framework
  bindings. Elsewhere install Tesseract and use `uv sync --extra tesseract`.

## Development

```bash
uv sync --extra dev
uv run pytest            # 20 tests; the end-to-end test is skipped unless Ollama is running
uv run ruff check src tests
uv run locallens eval eval/golden.json             # retrieval hit rate, answer accuracy, citation validity
uv run locallens eval eval/golden.json --no-generate   # retrieval metrics only, fast
```

The golden set format is documented at the top of `src/locallens/eval.py`. Reports are
written as JSON under `~/.locallens/eval`.

Architecture, design decisions and the phased build log are in
[docs/DESIGN.md](docs/DESIGN.md). In short: ingest normalises every page into a common
document model regardless of whether text came from a text layer, OCR or the vision
model; chunks keep their heading path and document name; retrieval is hybrid with
reciprocal rank fusion and a reranker; generation runs through Ollama with a citation
check afterwards.

## Roadmap

- Validate the handwriting auto-routing threshold on real handwritten samples.
- Local web UI with page previews and highlighted citations, bound to localhost.
- Watch-folder mode and `uv tool install` packaging.
- Optional encryption of the index at rest.
