import pymupdf
import pytest


@pytest.fixture
def sample_pdf(tmp_path):
    """A 3-page digital PDF with headings, body text and a near-empty 'scanned' page."""
    path = tmp_path / "dbms_notes.pdf"
    doc = pymupdf.open()
    body = (
        "Normalization is the process of organizing data to reduce redundancy. "
        "It improves data integrity by splitting tables into related tables. "
    )
    for title, para in [
        ("Chapter 1: Normalization", "First normal form (1NF) requires atomic values. " + body * 6),
        ("Chapter 2: Indexing", "A B-tree index keeps keys sorted for fast range queries. " + body * 6),
    ]:
        page = doc.new_page()
        page.insert_text((72, 72), title, fontsize=18)
        rect = pymupdf.Rect(72, 110, 540, 760)
        page.insert_textbox(rect, para, fontsize=10)
    doc.new_page()  # blank → classified as scanned
    doc.save(path)
    return path
