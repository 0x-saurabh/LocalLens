"""Evaluation harness: golden questions → retrieval hit rate, answer keyword accuracy, citation validity.

Golden file format (JSON list):
  {"q": "...", "doc": "expense_report",            # substring of the expected document's filename
   "pages": [1],                                    # any of these pages in context counts as a hit
   "answer_contains": ["13,950"],                   # every item (case-insensitive) must be in the answer
   "answer_any": ["Tuesday", "4 pm"]}               # optional: at least one must be present
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from locallens import config
from locallens.embed import Embedder
from locallens.generate import stream_answer, verify_citations
from locallens.retrieve import Hit, retrieve
from locallens.store import Store


@dataclass
class QResult:
    q: str
    doc: str
    expected_pages: list[int]
    hit_rank: int | None            # 1-based rank of the first expected page in context (None = miss)
    context_pages: list[tuple[str, int]]
    answer: str | None = None
    answer_ok: bool | None = None
    missing: list[str] = field(default_factory=list)
    citations_valid: bool | None = None
    bogus_citations: list[int] = field(default_factory=list)
    retrieve_s: float = 0.0
    answer_s: float = 0.0


@dataclass
class EvalReport:
    n: int
    hit_at_k: float
    mrr: float
    answer_accuracy: float | None
    citation_validity: float | None
    rerank: bool
    k: int
    mean_retrieve_s: float
    mean_answer_s: float
    results: list[QResult]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


def _hit_rank(hits: list[Hit], doc: str, pages: list[int]) -> int | None:
    for h in hits:
        if doc.lower() in Path(h.doc_path).name.lower() and (not pages or h.page in pages):
            return h.n
    return None


def run(golden: list[dict], cfg: config.Config, store: Store, embedder: Embedder,
        rerank: bool, generate: bool = True, log=lambda s: None) -> EvalReport:
    results: list[QResult] = []
    for item in golden:
        t0 = time.perf_counter()
        hits = retrieve(item["q"], cfg, store, embedder, rerank=rerank)
        t1 = time.perf_counter()
        r = QResult(
            q=item["q"], doc=item["doc"], expected_pages=item.get("pages", []),
            hit_rank=_hit_rank(hits, item["doc"], item.get("pages", [])),
            context_pages=[(Path(h.doc_path).stem[:18], h.page) for h in hits], retrieve_s=t1 - t0,
        )
        if generate:
            answer = "".join(stream_answer(item["q"], hits, cfg))
            r.answer_s = time.perf_counter() - t1
            r.answer = answer
            low = answer.lower()
            r.missing = [s for s in item.get("answer_contains", []) if s.lower() not in low]
            any_ok = (not item.get("answer_any")) or any(s.lower() in low for s in item["answer_any"])
            r.answer_ok = not r.missing and any_ok
            valid, bogus = verify_citations(answer, hits)
            r.citations_valid = not bogus and (bool(valid) or "not found" in low)
            r.bogus_citations = sorted(bogus)
        results.append(r)
        log(r)

    n = len(results) or 1
    hits_ = [r for r in results if r.hit_rank]
    answered = [r for r in results if r.answer_ok is not None]
    return EvalReport(
        n=len(results),
        hit_at_k=len(hits_) / n,
        mrr=sum(1 / r.hit_rank for r in hits_) / n,
        answer_accuracy=(sum(r.answer_ok for r in answered) / len(answered)) if answered else None,
        citation_validity=(sum(bool(r.citations_valid) for r in answered) / len(answered)) if answered else None,
        rerank=rerank, k=cfg.retrieval.top_k_final,
        mean_retrieve_s=sum(r.retrieve_s for r in results) / n,
        mean_answer_s=(sum(r.answer_s for r in results) / n) if generate else 0.0,
        results=results,
    )
