"""Embeddings via the local Ollama server."""

from __future__ import annotations

import ollama

from locallens import config

BATCH = 32


class Embedder:
    def __init__(self, cfg: config.Config | None = None):
        self.cfg = cfg or config.load()
        self.client = ollama.Client(host=self.cfg.ollama_host)
        self.model = self.cfg.models.embed

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), BATCH):
            res = self.client.embed(model=self.model, input=texts[i : i + BATCH])
            out.extend(list(v) for v in res.embeddings)
        return out

    def embed_query(self, text: str) -> list[float]:
        # nomic-embed-text expects task prefixes; harmless for other models.
        prefix = "search_query: " if "nomic" in self.model else ""
        return self.embed([prefix + text])[0]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        prefix = "search_document: " if "nomic" in self.model else ""
        return self.embed([prefix + t for t in texts])
