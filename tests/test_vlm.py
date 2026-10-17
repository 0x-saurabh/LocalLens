import pytest

from locallens.ingest import pdf_text
from locallens.ingest.ocr import OcrLine
from locallens.ingest.vlm import VlmResult, score, to_blocks


def test_score_penalizes_unreadable_and_rewards_ocr_agreement():
    clean = score("Buy milk eggs and rice", None)
    assert clean["confidence"] == pytest.approx(0.85) and clean["unreadable_ratio"] == 0
    rough = score("Buy [?] eggs and [?]", None)
    assert rough["unreadable_ratio"] == pytest.approx(0.4) and rough["confidence"] < 0.6
    agreed = score("Buy [?] eggs and [?]", "buy milk eggs and rice")
    assert agreed["ocr_agreement"] is not None


def test_to_blocks_splits_paragraphs_and_tables():
    text = ("Grocery list for Monday\nremember the [?] bag\n\n"
            "| Item | Qty |\n|---|---|\n| Milk | 2 |\n| Eggs | 12 |\n\n"
            "Call the plumber.")
    res = VlmResult(text=text, confidence=0.7, unreadable_ratio=0.1, ocr_agreement=None)
    blocks = to_blocks(res, "d:p1", (0, 0, 595, 842), handwriting=True)
    assert [b.type for b in blocks] == ["handwriting", "table", "handwriting"]
    assert blocks[0].text == "Grocery list for Monday remember the [?] bag"
    assert blocks[1].rows == [["Item", "Qty"], ["Milk", "2"], ["Eggs", "12"]]
    assert all(b.source == "vlm" and b.confidence == 0.7 for b in blocks)


class FakeVlm:
    def __init__(self, text): self.text = text; self.calls = 0
    def transcribe(self, image, ocr_text=None):
        self.calls += 1
        return VlmResult(text=self.text, **score(self.text, ocr_text))


def low_conf_ocr(gray):
    return [OcrLine("gibberish", 0.3, (10, 10, 200, 30)), OcrLine("more", 0.4, (10, 40, 200, 60))]


def test_handwritten_pages_route_to_vlm(tmp_path):
    import pymupdf
    src = pymupdf.open(); pg = src.new_page(); pg.insert_text((72, 100), "anything", fontsize=20)
    pix = pg.get_pixmap(dpi=100)
    out = pymupdf.open(); p = out.new_page(); p.insert_image(p.rect, pixmap=pix)
    path = tmp_path / "note.pdf"; out.save(path)

    fake = FakeVlm("Dentist appointment Tuesday 4pm\nbring [?] card")
    doc = pdf_text.extract(path, "hw01", "application/pdf", ocr=low_conf_ocr, vlm=fake, vlm_mode="handwriting")
    page = doc.pages[0]
    assert page.kind == "handwritten" and fake.calls == 1
    assert page.blocks[0].source == "vlm" and page.blocks[0].type == "handwriting"
    assert "Dentist appointment" in page.blocks[0].text and page.blocks[0].confidence < 0.85

    fake2 = FakeVlm("whatever")
    doc2 = pdf_text.extract(path, "hw01", "application/pdf", ocr=low_conf_ocr, vlm=fake2, vlm_mode="off")
    assert fake2.calls == 0 and doc2.pages[0].blocks[0].source == "ocr"
