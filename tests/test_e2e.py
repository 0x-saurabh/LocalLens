"""End-to-end against the local Ollama server. Skipped when it isn't running."""

import socket

import pytest

from locallens import config
from locallens.embed import Embedder
from locallens.generate import stream_answer, verify_citations
from locallens.ingest.pipeline import ingest
from locallens.retrieve import retrieve
from locallens.store import Store


def _ollama_up() -> bool:
    try:
        socket.create_connection(("127.0.0.1", 11434), timeout=0.5).close()
        return True
    except OSError:
        return False


@pytest.mark.skipif(not _ollama_up(), reason="Ollama not running")
def test_ingest_ask_cycle(sample_pdf, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "HOME", tmp_path / "home")
    cfg = config.Config()
    store, emb = Store(cfg), Embedder(cfg)

    rep = ingest([sample_pdf], cfg, store, emb)
    assert rep.files_indexed == 1 and rep.chunks > 0
    assert ingest([sample_pdf], cfg, store, emb).files_skipped_unchanged == 1

    hits = retrieve("What does a B-tree index do?", cfg, store, emb)
    assert hits and hits[0].page == 2

    answer = "".join(stream_answer("What does a B-tree index do?", hits, cfg))
    valid, bogus = verify_citations(answer, hits)
    assert valid and not bogus
    assert "sorted" in answer.lower() or "range" in answer.lower()
