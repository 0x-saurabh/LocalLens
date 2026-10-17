import json
from pathlib import Path

from locallens.eval import _hit_rank
from locallens.rerank import Reranker
from locallens.retrieve import Hit


def _hit(n, doc, page):
    return Hit(n=n, doc_id="x", doc_path=f"/d/{doc}.pdf", page=page, text="", source="textlayer",
               confidence=None, heading_path=[], score=0.0)


def test_hit_rank_matches_doc_and_page():
    hits = [_hit(1, "other", 1), _hit(2, "expense_report_2025", 3), _hit(3, "expense_report_2025", 1)]
    assert _hit_rank(hits, "expense_report", [1]) == 3
    assert _hit_rank(hits, "expense_report", []) == 2
    assert _hit_rank(hits, "missing", [1]) is None


def test_golden_file_is_well_formed():
    items = json.loads((Path(__file__).parent.parent / "eval" / "golden.json").read_text())
    assert len(items) >= 20
    for it in items:
        assert it["q"] and "doc" in it and isinstance(it.get("pages", []), list)
        assert it.get("answer_contains") or it.get("answer_any")


def test_reranker_reports_cache_state_without_network():
    assert Reranker.is_cached("definitely/not-a-model") is False
