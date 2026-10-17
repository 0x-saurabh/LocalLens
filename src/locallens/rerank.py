"""Cross-encoder reranking of retrieval candidates. Runs locally (MPS on Apple silicon, else CPU)."""

from __future__ import annotations

import os

from locallens import config


class Reranker:
    def __init__(self, cfg: config.Config | None = None):
        self.cfg = cfg or config.load()
        self.model_name = self.cfg.models.reranker
        self._model = None

    @staticmethod
    def is_cached(model_name: str) -> bool:
        """True if the model is in the local Hugging Face cache (so it loads offline)."""
        try:
            from huggingface_hub import try_to_load_from_cache
        except ImportError:
            return False
        hit = try_to_load_from_cache(model_name, "config.json")
        return isinstance(hit, str) and os.path.exists(hit)

    def available(self) -> bool:
        return self.is_cached(self.model_name)

    def _load(self):
        if self._model is None:
            import torch
            from sentence_transformers import CrossEncoder

            device = "mps" if torch.backends.mps.is_available() else "cpu"
            self._model = CrossEncoder(self.model_name, max_length=512, device=device)
        return self._model

    def rerank(self, query: str, candidates: list[dict], k: int) -> list[tuple[dict, float]]:
        if not candidates:
            return []
        model = self._load()
        scores = model.predict([(query, c["text"]) for c in candidates], batch_size=16, show_progress_bar=False)
        ranked = sorted(zip(candidates, (float(s) for s in scores)), key=lambda t: -t[1])
        return ranked[:k]
