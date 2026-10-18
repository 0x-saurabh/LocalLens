"""Hybrid retrieval: vector + BM25 full-text, fused with Reciprocal Rank Fusion. Reranking: Phase 5."""

from __future__ import annotations

from dataclasses import dataclass

from locallens import config
from locallens.embed import Embedder
from locallens.rerank import Reranker
from locallens.store import Store

_reranker: Reranker | None = None


@dataclass
class Hit:
    n: int                 # 1-based citation number
    doc_id: str
    doc_path: str
    page: int
    text: str
    source: str
    confidence: float | None
    heading_path: list[str]
    score: float

    def snippet(self, limit: int = 110) -> str:
        """Chunk text without its [doc > heading] prefix, on one line, trimmed for display."""
        text = self.text.split("\n", 1)[-1] if self.text.startswith("[") else self.text
        text = text.replace("\n", " ")
        return text[:limit] + "…" if len(text) > limit else text


RRF_K = 60


def rrf_fuse(ranked_lists: list[list[dict]], key: str = "id") -> list[tuple[dict, float]]:
    """Reciprocal Rank Fusion: score = Σ 1/(k + rank). Robust to incomparable raw scores."""
    scores: dict[str, float] = {}
    rows: dict[str, dict] = {}
    for lst in ranked_lists:
        for rank, r in enumerate(lst, start=1):
            scores[r[key]] = scores.get(r[key], 0.0) + 1.0 / (RRF_K + rank)
            rows.setdefault(r[key], r)
    return [(rows[i], sc) for i, sc in sorted(scores.items(), key=lambda kv: -kv[1])]


def select(dense: list[dict], sparse: list[dict], k: int, guaranteed: int = 3) -> list[tuple[dict, float]]:
    """Top `guaranteed` hits from each retriever always make it in; the rest are filled by RRF.

    Plain RRF buries a strong single-retriever hit (e.g. an exact-number BM25 match that the
    embedder missed) beneath mediocre items that happen to appear in both lists.
    """
    fused = rrf_fuse([dense, sparse])
    score = {r["id"]: sc for r, sc in fused}
    chosen: list[dict] = []
    seen: set[str] = set()
    for lst in (dense, sparse):
        for r in lst[:guaranteed]:
            if r["id"] not in seen:
                chosen.append(r); seen.add(r["id"])
    for r, _ in fused:
        if len(chosen) >= k:
            break
        if r["id"] not in seen:
            chosen.append(r); seen.add(r["id"])
    chosen = chosen[:k]
    return sorted(((r, score[r["id"]]) for r in chosen), key=lambda t: -t[1])


def _get_reranker(cfg: config.Config) -> Reranker | None:
    global _reranker
    if _reranker is None:
        r = Reranker(cfg)
        _reranker = r if r.available() else False  # type: ignore[assignment]
    return _reranker or None


def retrieve(question: str, cfg: config.Config, store: Store, embedder: Embedder,
             rerank: bool | None = None) -> list[Hit]:
    """Hybrid candidates → (optionally) cross-encoder rerank → top_k_final hits."""
    n = cfg.retrieval.top_k_candidates
    dense = store.search(embedder.embed_query(question), n)
    sparse = store.search_fts(question, n)
    use_rerank = cfg.retrieval.rerank if rerank is None else rerank
    reranker = _get_reranker(cfg) if use_rerank else None
    if reranker is not None:
        pool = {r["id"]: r for r in dense}
        pool.update({r["id"]: r for r in sparse})
        fused = reranker.rerank(question, list(pool.values()), cfg.retrieval.top_k_final)
    else:
        fused = select(dense, sparse, cfg.retrieval.top_k_final, cfg.retrieval.guaranteed_per_retriever)
    return [
        Hit(
            n=i + 1, doc_id=r["doc_id"], doc_path=store.doc_path(r["doc_id"]), page=r["page"],
            text=r["text"], source=r["source"], confidence=r.get("confidence"),
            heading_path=r["heading_path"], score=score,
        )
        for i, (r, score) in enumerate(fused)
    ]
