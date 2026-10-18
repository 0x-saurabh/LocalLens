"""Local web server for the chat UI. Binds to 127.0.0.1 only.

Serves the Flutter web build from ui/build/web plus a small JSON API:
  POST /api/ask  {"question": "...", "rerank": true|false|null}
       → newline-delimited JSON: one "sources" event, many "token" events, one "done" event
  POST /api/upload?name=<file name>&vlm=handwriting|all-scans|off   (body: the file bytes)
       → newline-delimited JSON: "log" events while indexing, one "done" event with the report
  GET  /api/documents → [{"name", "pages", "digital_pages", "chunks", "indexed"}]
  GET  /api/health → {"ok": true, "documents": n, "chunks": n}
"""

from __future__ import annotations

import json
import mimetypes
import re
import threading
from collections.abc import Callable, Iterator
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from locallens import __version__, config
from locallens.ingest.discover import SUPPORTED, sha256_of

WEB_DIR = Path(__file__).resolve().parents[2] / "ui" / "build" / "web"
DEFAULT_PORT = 8765
MAX_BODY = 64 * 1024
MAX_UPLOAD = 200 * 1024 * 1024
VLM_MODES = {"handwriting", "all-scans", "off"}
_TYPES = {".wasm": "application/wasm", ".mjs": "text/javascript", ".js": "text/javascript",
          ".json": "application/json", ".otf": "font/otf", ".ttf": "font/ttf"}
_MARKUP = re.compile(r"\[/?(?:bold|red|green|yellow|dim|cyan)?\]")

# The browser itself refuses any request that is not to this server.
CSP = ("default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; style-src 'self' 'unsafe-inline'; "
       "img-src 'self' data: blob:; font-src 'self' data:; connect-src 'self'; worker-src 'self' blob:; "
       "frame-ancestors 'none'; base-uri 'self'; form-action 'none'")

_NOT_BUILT = b"""<!doctype html><meta charset="utf-8"><title>LocalLens</title>
<body style="font-family:system-ui;max-width:40rem;margin:4rem auto;padding:0 1rem;line-height:1.5">
<h1>The chat UI is not built yet</h1>
<p>Build it once (needs the Flutter SDK), then run <code>locallens ui</code> again:</p>
<pre>cd ui
flutter build web --no-web-resources-cdn</pre></body>"""


class Backend:
    """What the API does. Tests swap in a fake; the real one calls the same code as the CLI."""

    def __init__(self, cfg: config.Config):
        self.cfg = cfg

    def ask(self, question: str, rerank: bool | None = None) -> Iterator[dict]:
        """Same flow as `locallens ask`, as a stream of JSON-able events."""
        from locallens.embed import Embedder
        from locallens.generate import stream_answer, verify_citations
        from locallens.retrieve import retrieve
        from locallens.store import Store

        hits = retrieve(question, self.cfg, Store(self.cfg), Embedder(self.cfg), rerank=rerank)
        if not hits:
            yield {"type": "error", "message": "Nothing indexed yet. Upload a file or run `locallens ingest <folder>`."}
            return
        yield {"type": "sources", "sources": [
            {"n": h.n, "document": Path(h.doc_path).name, "page": h.page, "snippet": h.snippet(),
             "source": h.source, "confidence": h.confidence}
            for h in hits
        ]}
        answer = ""
        for token in stream_answer(question, hits, self.cfg):
            answer += token
            yield {"type": "token", "text": token}
        valid, bogus = verify_citations(answer, hits)
        yield {"type": "done", "cited": sorted(valid), "bogus": sorted(bogus)}

    def ingest(self, path: Path, vlm_mode: str, log: Callable[[str], None]) -> dict:
        """Same as `locallens ingest <path> --vlm <mode>`; returns the report."""
        from locallens.embed import Embedder
        from locallens.ingest.pipeline import ingest
        from locallens.store import Store

        rep = ingest([path], self.cfg, Store(self.cfg), Embedder(self.cfg), log=log, vlm_mode=vlm_mode)
        out = asdict(rep)
        out["errors"] = [err for _, err in rep.errors]
        return out

    def documents(self) -> list[dict]:
        from locallens.store import Store

        return [{"name": Path(d["path"]).name, "pages": d["page_count"], "digital_pages": d["digital_pages"],
                 "chunks": d["chunk_count"], "indexed": d["ingested_at"]}
                for d in Store(self.cfg).documents()]

    def health(self) -> dict:
        from locallens.store import Store

        store = Store(self.cfg)
        return {"ok": True, "version": __version__, "documents": len(store.documents()),
                "chunks": store.chunk_count()}


def safe_upload_name(raw: str) -> str | None:
    """A plain file name with a supported extension, or None. Strips any folder part."""
    name = Path(raw.replace("\\", "/")).name.strip()
    if not name or name.startswith(".") or Path(name).suffix.lower() not in SUPPORTED:
        return None
    return name


def _unique_target(folder: Path, name: str, tmp: Path) -> Path:
    """`folder/name`, or `name-2.ext`, `name-3.ext`… if a different file already has that name."""
    stem, suffix = Path(name).stem, Path(name).suffix
    target, i = folder / name, 1
    while target.exists() and sha256_of(target) != sha256_of(tmp):
        i += 1
        target = folder / f"{stem}-{i}{suffix}"
    return target


def make_handler(backend: Backend, web_dir: Path = WEB_DIR,
                 uploads_dir: Path | None = None) -> type[BaseHTTPRequestHandler]:
    model_lock = threading.Lock()   # one answer or ingest at a time: the models are the bottleneck anyway
    root = web_dir.resolve()
    uploads = uploads_dir or backend.cfg.uploads_dir

    class Handler(BaseHTTPRequestHandler):
        server_version = f"LocalLens/{__version__}"

        def log_message(self, format, *args) -> None:  # quiet by default
            pass

        def _host_ok(self) -> bool:
            # Blocks DNS rebinding: a web page on evil.example resolving to 127.0.0.1 still sends
            # its own name in Host, so it never gets an answer back.
            port = self.server.server_address[1]
            return self.headers.get("Host", "") in {f"127.0.0.1:{port}", f"localhost:{port}"}

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            # Flutter file names carry no version hash, so always revalidate after a rebuild
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", CSP)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: dict | list) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def _start_stream(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

        def _emit(self, ev: dict) -> None:
            self.wfile.write(json.dumps(ev).encode() + b"\n")
            self.wfile.flush()

        def _emit_error(self, exc: Exception) -> None:
            msg = f"{exc.__class__.__name__}: {exc}"
            if "ConnectError" in msg or "Connection refused" in msg:
                msg = "Cannot reach Ollama. Start it with `ollama serve` or open the Ollama app."
            self._emit({"type": "error", "message": msg})

        def do_GET(self) -> None:
            if not self._host_ok():
                return self._json(403, {"error": "forbidden host"})
            path = urlsplit(self.path).path
            if path == "/api/health":
                return self._json(200, backend.health())
            if path == "/api/documents":
                return self._json(200, backend.documents())
            if not (root / "index.html").is_file():
                return self._send(200, _NOT_BUILT, "text/html; charset=utf-8")
            target = (root / path.lstrip("/")).resolve()
            if not target.is_relative_to(root):
                return self._json(404, {"error": "not found"})
            if target.is_dir():
                target = target / "index.html"
            if not target.is_file():
                if Path(path).suffix:
                    return self._json(404, {"error": "not found"})
                target = root / "index.html"   # client-side route
            ctype = _TYPES.get(target.suffix) or mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            self._send(200, target.read_bytes(), ctype)

        def do_POST(self) -> None:
            if not self._host_ok():
                return self._json(403, {"error": "forbidden host"})
            url = urlsplit(self.path)
            if url.path == "/api/ask":
                return self._ask()
            if url.path == "/api/upload":
                return self._upload(parse_qs(url.query))
            return self._json(404, {"error": "not found"})

        def _ask(self) -> None:
            # JSON only: a cross-site form or simple request cannot send this without a CORS preflight
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                return self._json(415, {"error": "expected application/json"})
            length = int(self.headers.get("Content-Length") or 0)
            if not 0 < length <= MAX_BODY:
                return self._json(400, {"error": "bad request body"})
            try:
                body = json.loads(self.rfile.read(length))
                question = str(body["question"]).strip()
            except (ValueError, KeyError, TypeError):
                return self._json(400, {"error": "expected {\"question\": \"...\"}"})
            if not question:
                return self._json(400, {"error": "empty question"})
            rerank = body.get("rerank")

            self._start_stream()
            with model_lock:
                try:
                    for ev in backend.ask(question, rerank if isinstance(rerank, bool) else None):
                        self._emit(ev)
                except (BrokenPipeError, ConnectionResetError):
                    pass   # the browser went away mid-answer
                except Exception as exc:  # noqa: BLE001
                    self._emit_error(exc)

        def _upload(self, query: dict[str, list[str]]) -> None:
            # octet-stream is not a "simple" content type, so other sites need a preflight we never grant
            if self.headers.get("Content-Type", "") != "application/octet-stream":
                return self._json(415, {"error": "expected application/octet-stream"})
            name = safe_upload_name(query.get("name", [""])[0])
            if name is None:
                return self._json(400, {"error": "unsupported file type; use " + " ".join(sorted(SUPPORTED))})
            vlm = query.get("vlm", ["handwriting"])[0]
            if vlm not in VLM_MODES:
                return self._json(400, {"error": f"vlm must be one of {sorted(VLM_MODES)}"})
            length = int(self.headers.get("Content-Length") or 0)
            if not 0 < length <= MAX_UPLOAD:
                return self._json(413 if length else 400,
                                  {"error": f"file must be between 1 byte and {MAX_UPLOAD // (1024 * 1024)} MB"})

            uploads.mkdir(parents=True, exist_ok=True)
            tmp = uploads / f".upload-{threading.get_ident()}.part"
            try:
                with tmp.open("wb") as f:
                    remaining = length
                    while remaining:
                        chunk = self.rfile.read(min(remaining, 1 << 20))
                        if not chunk:
                            raise ConnectionResetError("upload interrupted")
                        f.write(chunk)
                        remaining -= len(chunk)
                target = _unique_target(uploads, name, tmp)
                tmp.replace(target)
            except (ConnectionResetError, BrokenPipeError):
                tmp.unlink(missing_ok=True)
                return
            except OSError as exc:
                tmp.unlink(missing_ok=True)
                return self._json(500, {"error": f"could not save the file: {exc.strerror}"})

            self._start_stream()
            self._emit({"type": "saved", "name": target.name})
            with model_lock:
                try:
                    report = backend.ingest(target, vlm, lambda s: self._emit({"type": "log", "text": _MARKUP.sub("", s)}))
                    self._emit({"type": "done", "name": target.name, "report": report})
                except (BrokenPipeError, ConnectionResetError):
                    pass
                except Exception as exc:  # noqa: BLE001
                    self._emit_error(exc)

    return Handler


def make_server(backend: Backend, port: int = DEFAULT_PORT, web_dir: Path = WEB_DIR,
                uploads_dir: Path | None = None) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(backend, web_dir, uploads_dir))
    server.daemon_threads = True
    return server
