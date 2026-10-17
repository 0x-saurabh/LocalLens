import pymupdf
import pytest

from locallens.chunk import chunk_document
from locallens.ingest import pdf_text
from locallens.retrieve import rrf_fuse, select

ROWS = [["Date", "Provider", "Amount"], ["2024-03-11", "Dentist", "820.00"],
        ["2024-05-02", "Pharmacy", "45.50"], ["2024-09-19", "Optician", "310.00"]]


@pytest.fixture
def table_pdf(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 70), "Medical expenses 2024", fontsize=16)
    x, y0, h = [72, 180, 330, 460], 100, 22
    for r, row in enumerate(ROWS):
        for c, cell in enumerate(row):
            page.draw_rect(pymupdf.Rect(x[c], y0 + r * h, x[c + 1], y0 + (r + 1) * h), color=(0, 0, 0), width=0.8)
            page.insert_text((x[c] + 4, y0 + r * h + 15), cell, fontsize=10)
    page.insert_textbox(pymupdf.Rect(72, 220, 540, 400),
                        "All receipts were paid by card. Totals are in USD. " * 8, fontsize=10)
    path = tmp_path / "expenses.pdf"
    doc.save(path)
    return path


def test_table_block_replaces_cell_text(table_pdf):
    doc = pdf_text.extract(table_pdf, "tab001", "application/pdf")
    blocks = doc.pages[0].blocks
    tables = [b for b in blocks if b.type == "table"]
    assert len(tables) == 1 and tables[0].rows == ROWS
    assert tables[0].text.startswith("| Date | Provider | Amount |")
    paras = " ".join(b.text for b in blocks if b.type != "table")
    assert "Dentist" not in paras                 # cell text not duplicated as paragraphs
    assert blocks[0].type == "heading" and blocks[1].type == "table"   # reading order kept


def test_table_chunks_whole_and_rows(table_pdf):
    doc = pdf_text.extract(table_pdf, "tab001", "application/pdf")
    chunks = chunk_document(doc)
    rows = [c for c in chunks if ":r" in c.id]
    assert len(rows) == 3
    assert "Provider: Dentist; Amount: 820.00" in rows[0].text
    assert all(c.text.startswith("[expenses > Medical expenses 2024]") for c in rows)
    whole = next(c for c in chunks if "| Date | Provider |" in c.text)
    assert whole.block_ids == rows[0].block_ids


def test_rrf_prefers_items_in_both_lists():
    dense = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    sparse = [{"id": "c"}, {"id": "d"}]
    fused = rrf_fuse([dense, sparse])
    ids = [r["id"] for r, _ in fused]
    assert ids[:2] == ["c", "a"]          # c is in both lists, a is rank 1 in dense
    assert set(ids[2:]) == {"b", "d"}    # b and d tie (rank 2 in one list each)


def test_select_guarantees_top_hits_from_each_retriever():
    both = [{"id": f"b{i}"} for i in range(10)]                 # appear in both lists
    dense = both + [{"id": "d_only"}]
    sparse = [{"id": "s1"}, {"id": "s2"}, {"id": "s3"}, {"id": "s4"}] + both
    picked = [r["id"] for r, _ in select(dense, sparse, k=8, guaranteed=3)]
    assert {"s1", "s2", "s3"} <= set(picked)       # sparse top-3 survive despite RRF
    assert "s4" not in picked and len(picked) == 8
