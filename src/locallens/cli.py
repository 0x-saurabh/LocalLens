"""LocalLens command-line interface. Everything runs from the shell."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from locallens import __version__, config, offline_guard

app = typer.Typer(help="Fully offline, privacy-first RAG over your documents.", no_args_is_help=True)
console = Console()


@app.callback()
def _main(ctx: typer.Context) -> None:
    # Every command except `setup` runs with the network blocked.
    if ctx.invoked_subcommand != "setup":
        offline_guard.enable()


@app.command()
def doctor() -> None:
    """Check local runtime: Ollama reachable, models present, offline guard active."""
    cfg = config.load()
    table = Table(title=f"LocalLens {__version__} — doctor")
    table.add_column("Check")
    table.add_column("Status")

    table.add_row("offline guard", "[green]active[/]" if offline_guard.is_enabled() else "[red]off[/]")
    table.add_row("home", str(config.HOME))
    from locallens.ingest.ocr import available_engines
    engines = available_engines()
    table.add_row("ocr engines", f"[green]{', '.join(engines)}[/]" if engines else "[red]none[/] (install tesseract)")
    from locallens.rerank import Reranker
    table.add_row("model: reranker", f"[green]{cfg.models.reranker}[/]" if Reranker.is_cached(cfg.models.reranker)
                  else f"[yellow]{cfg.models.reranker} not cached[/] → run `locallens setup` (hybrid-only until then)")

    try:
        import ollama

        client = ollama.Client(host=cfg.ollama_host)
        names = {m.model for m in client.list().models}
        table.add_row("ollama", f"[green]reachable[/] at {cfg.ollama_host}")
        for role in ("llm", "embed", "vlm"):
            want = getattr(cfg.models, role)
            have = any(n == want or n.split(":")[0] == want.split(":")[0] for n in names)
            table.add_row(f"model: {role}", f"[green]{want}[/]" if have else f"[yellow]{want} missing[/] → run `locallens setup`")
    except Exception as exc:  # noqa: BLE001
        table.add_row("ollama", f"[red]unreachable[/] ({exc.__class__.__name__}) — is `ollama serve` running?")

    console.print(table)


@app.command()
def setup() -> None:
    """ONE-TIME: pull local models. This is the only command allowed to use the network."""
    import ollama

    cfg = config.load()
    client = ollama.Client(host=cfg.ollama_host)
    for name in (cfg.models.llm, cfg.models.embed, cfg.models.vlm):
        console.print(f"[bold]pulling[/] {name} …")
        for progress in client.pull(name, stream=True):
            if progress.completed and progress.total:
                console.print(f"  {progress.completed / progress.total:5.1%}", end="\r")
        console.print(f"  [green]done[/] {name}")
    from huggingface_hub import snapshot_download

    console.print(f"[bold]downloading[/] reranker {cfg.models.reranker} …")
    snapshot_download(cfg.models.reranker,
                      allow_patterns=["*.json", "*.safetensors", "sentencepiece.bpe.model", "tokenizer*"])
    console.print(f"  [green]done[/] {cfg.models.reranker}")
    console.print(f"config written to {config.save(cfg)}")


@app.command()
def ingest(
    paths: Annotated[list[Path], typer.Argument(exists=True)],
    force: Annotated[bool, typer.Option("--force", help="Re-index even if the file is unchanged.")] = False,
    vlm: Annotated[str | None, typer.Option("--vlm", help="handwriting (default) | all-scans | off")] = None,
) -> None:
    """Index files or folders (incremental: unchanged files are skipped)."""
    from locallens.embed import Embedder
    from locallens.ingest.pipeline import ingest as run_ingest
    from locallens.store import Store

    cfg = config.load()
    with console.status("indexing…"):
        rep = run_ingest(paths, cfg, Store(cfg), Embedder(cfg), log=console.print, force=force, vlm_mode=vlm)
    console.print(
        f"\n[bold]{rep.files_indexed}[/] indexed · {rep.files_skipped_unchanged} unchanged · "
        f"{rep.pages_digital} digital pages · {rep.pages_ocr} OCR pages"
        + (f" ({rep.pages_handwritten} handwritten)" if rep.pages_handwritten else "")
        + (f" · {rep.pages_vlm} via VLM" if rep.pages_vlm else "")
        + f" · {rep.tables} tables · {rep.chunks} chunks" + (f" · ocr={rep.ocr_engine}" if rep.ocr_engine else "")
    )
    for path, err in rep.errors:
        console.print(f"[red]error[/] {path}: {err}")


def _answer(question: str, cfg, store, embedder, rerank: bool | None = None) -> None:
    from rich.markdown import Markdown
    from rich.markup import escape
    from rich.panel import Panel

    from locallens.generate import stream_answer, verify_citations
    from locallens.retrieve import retrieve

    hits = retrieve(question, cfg, store, embedder, rerank=rerank)
    if not hits:
        console.print("[yellow]Nothing indexed yet. Run `locallens ingest <folder>` first.[/]")
        return
    answer = ""
    with console.status("thinking…"):
        for token in stream_answer(question, hits, cfg):
            answer += token
    console.print(Panel(Markdown(answer.strip()), title="answer", border_style="green"))

    valid, bogus = verify_citations(answer, hits)
    if bogus:
        console.print(f"[red]⚠ answer cites non-existent sources {sorted(bogus)}[/]")
    table = Table(title="sources", show_lines=False)
    table.add_column("#", justify="right")
    table.add_column("document")
    table.add_column("page", justify="right")
    table.add_column("snippet")
    for h in hits:
        mark = "[green]●[/]" if h.n in valid else "[dim]○[/]"
        table.add_row(f"{mark} {h.n}", escape(Path(h.doc_path).name), str(h.page), escape(h.snippet()))
    console.print(table)


@app.command()
def ask(
    question: str,
    no_rerank: Annotated[bool, typer.Option("--no-rerank", help="Skip the cross-encoder.")] = False,
) -> None:
    """Ask a single question; prints a cited answer."""
    from locallens.embed import Embedder
    from locallens.store import Store

    cfg = config.load()
    _answer(question, cfg, Store(cfg), Embedder(cfg), rerank=False if no_rerank else None)


@app.command()
def remove(paths: Annotated[list[Path], typer.Argument()]) -> None:
    """Remove documents from the index by path."""
    from locallens.store import Store

    store = Store(config.load())
    for p in paths:
        doc_id = store.doc_id_for_path(str(p.resolve()))
        if doc_id:
            store.delete_document(doc_id)
            console.print(f"removed {p}")
        else:
            console.print(f"[yellow]not indexed:[/] {p}")
    store.rebuild_fts()


@app.command()
def eval(
    golden: Annotated[
        Path, typer.Argument(exists=True, help="JSON list of golden questions (see eval/golden.json).")
    ],
    no_rerank: Annotated[bool, typer.Option("--no-rerank")] = False,
    no_generate: Annotated[bool, typer.Option("--no-generate", help="Retrieval metrics only (fast).")] = False,
    out: Annotated[Path | None, typer.Option("--out", help="Write the full JSON report here.")] = None,
) -> None:
    """Score retrieval and answers against a golden question set."""
    import json

    from locallens import eval as ev
    from locallens.embed import Embedder
    from locallens.store import Store

    cfg = config.load()
    items = json.loads(golden.read_text())
    table = Table(title="eval", show_lines=False)
    for col in ("hit", "ans", "cite", "q", "context pages"):
        table.add_column(col)

    def log(r: ev.QResult) -> None:
        hit = f"[green]#{r.hit_rank}[/]" if r.hit_rank else "[red]miss[/]"
        ans = "" if r.answer_ok is None else ("[green]ok[/]" if r.answer_ok else f"[red]missing {r.missing}[/]")
        cite = "" if r.citations_valid is None else ("[green]ok[/]" if r.citations_valid else f"[red]{r.bogus_citations}[/]")
        pages = " ".join(f"{d[:10]}:{p}" for d, p in r.context_pages[:6])
        table.add_row(hit, ans, cite, r.q[:60], pages)

    with console.status("evaluating…"):
        rep = ev.run(items, cfg, Store(cfg), Embedder(cfg), rerank=not no_rerank, generate=not no_generate, log=log)
    console.print(table)
    summary = (f"[bold]n={rep.n}[/] · hit@{rep.k}={rep.hit_at_k:.0%} · MRR={rep.mrr:.2f}"
               + (f" · answer acc={rep.answer_accuracy:.0%} · citations valid={rep.citation_validity:.0%}" if rep.answer_accuracy is not None else "")
               + f" · rerank={'on' if rep.rerank else 'off'} · retrieve {rep.mean_retrieve_s:.2f}s"
               + (f" · answer {rep.mean_answer_s:.1f}s" if not no_generate else ""))
    console.print(summary)
    out = out or (config.HOME / "eval" / f"{golden.stem}-{'rerank' if rep.rerank else 'hybrid'}-{int(__import__('time').time())}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rep.to_json())
    console.print(f"report → {out}")


@app.command()
def chat() -> None:
    """Interactive chat loop in the terminal (multi-turn, with citations)."""
    from locallens.embed import Embedder
    from locallens.store import Store

    cfg = config.load()
    store, embedder = Store(cfg), Embedder(cfg)
    console.print("[bold]LocalLens chat[/] — offline. Type a question, or /quit.")
    while True:
        try:
            q = console.input("[cyan]you ›[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q in {"/quit", "/exit", "/q"}:
            break
        _answer(q, cfg, store, embedder)


@app.command()
def ui(
    port: Annotated[int, typer.Option("--port", help="Local port for the chat page.")] = 8765,
    no_browser: Annotated[bool, typer.Option("--no-browser", help="Do not open the browser.")] = False,
) -> None:
    """Open the chat window in your browser (served on 127.0.0.1 only)."""
    import threading
    import webbrowser

    from locallens.server import WEB_DIR, Backend, make_server

    try:
        server = make_server(Backend(config.load()), port=port)
    except OSError as exc:
        console.print(f"[red]cannot use port {port}[/] ({exc.strerror}) — is `locallens ui` already running? "
                      f"Try --port {port + 1}")
        raise typer.Exit(1) from None
    url = f"http://127.0.0.1:{port}/"
    if not (WEB_DIR / "index.html").exists():
        console.print("[yellow]chat UI not built[/] — run: cd ui && flutter build web --no-web-resources-cdn")
    console.print(f"[bold]LocalLens chat[/] at {url} — offline. Press Ctrl+C to stop.")
    if not no_browser:
        threading.Timer(0.4, webbrowser.open, (url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


@app.command()
def status() -> None:
    """Show indexed documents and chunk counts."""
    from locallens.store import Store

    store = Store(config.load())
    docs = store.documents()
    table = Table(title=f"{len(docs)} documents · {store.chunk_count()} chunks")
    table.add_column("document")
    table.add_column("pages", justify="right")
    table.add_column("digital", justify="right")
    table.add_column("scanned", justify="right")
    table.add_column("chunks", justify="right")
    table.add_column("indexed")
    for d in docs:
        table.add_row(Path(d["path"]).name, str(d["page_count"]), str(d["digital_pages"]),
                      str(d["skipped_pages"]), str(d["chunk_count"]), d["ingested_at"][:16])
    console.print(table)
