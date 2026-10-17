import hashlib

from locallens.chunk import chunk_document
from locallens.ingest import discover, pdf_text


def test_discover_hashes_pdfs(sample_pdf):
    cands = list(discover.discover([sample_pdf.parent]))
    assert len(cands) == 1
    assert cands[0].sha256 == hashlib.sha256(sample_pdf.read_bytes()).hexdigest()


def test_extract_headings_and_page_kinds(sample_pdf):
    doc = pdf_text.extract(sample_pdf, "abc123", "application/pdf")
    assert doc.page_count == 3
    assert [p.kind for p in doc.pages] == ["digital", "digital", "scanned"]
    headings = [b.text for p in doc.pages for b in p.blocks if b.type == "heading"]
    assert headings == ["Chapter 1: Normalization", "Chapter 2: Indexing"]
    para = next(b for b in doc.pages[1].blocks if b.type == "paragraph")
    assert para.heading_path == ["Chapter 2: Indexing"]


def test_chunks_carry_context_and_respect_size(sample_pdf):
    doc = pdf_text.extract(sample_pdf, "abc123", "application/pdf")
    chunks = chunk_document(doc, chunk_tokens=100, overlap=0.15)
    assert len(chunks) >= 4
    assert all(c.text.startswith("[dbms_notes > Chapter") for c in chunks)
    assert all(len(c.text) < 100 * 4 * 1.6 for c in chunks)  # prefix + overlap allowance
    assert {c.page for c in chunks} == {1, 2}
