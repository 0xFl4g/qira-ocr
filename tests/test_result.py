import pytest

from qira_ocr.result import BBox, Word, Line, Block, Page, OCRResult


class TestBBox:
    def test_create(self):
        bbox = BBox(x1=0, y1=0, x2=100, y2=50)
        assert bbox.x1 == 0
        assert bbox.x2 == 100

    def test_width_height(self):
        bbox = BBox(x1=10, y1=20, x2=110, y2=70)
        assert bbox.width == 100
        assert bbox.height == 50


class TestWord:
    def test_create(self):
        word = Word(text="hello", bbox=BBox(0, 0, 50, 20), confidence=0.95)
        assert word.text == "hello"
        assert word.confidence == 0.95


class TestLine:
    def test_text_joins_words(self):
        words = [
            Word("hello", BBox(0, 0, 50, 20), 0.95),
            Word("world", BBox(60, 0, 110, 20), 0.90),
        ]
        line = Line(words=words, bbox=BBox(0, 0, 110, 20))
        assert line.text == "hello world"

    def test_confidence_is_mean(self):
        words = [
            Word("a", BBox(0, 0, 10, 10), 0.80),
            Word("b", BBox(20, 0, 30, 10), 1.0),
        ]
        line = Line(words=words, bbox=BBox(0, 0, 30, 10))
        assert line.confidence == pytest.approx(0.90)


class TestBlock:
    def test_text_joins_lines(self):
        line1 = Line(
            words=[Word("hello", BBox(0, 0, 50, 20), 0.9)],
            bbox=BBox(0, 0, 50, 20),
        )
        line2 = Line(
            words=[Word("world", BBox(0, 30, 50, 50), 0.9)],
            bbox=BBox(0, 30, 50, 50),
        )
        block = Block(lines=[line1, line2], bbox=BBox(0, 0, 50, 50))
        assert block.text == "hello\nworld"


class TestPage:
    def test_text_joins_blocks(self):
        block1 = Block(
            lines=[Line(words=[Word("first", BBox(0, 0, 50, 20), 0.9)], bbox=BBox(0, 0, 50, 20))],
            bbox=BBox(0, 0, 50, 20),
        )
        block2 = Block(
            lines=[Line(words=[Word("second", BBox(0, 50, 50, 70), 0.9)], bbox=BBox(0, 50, 50, 70))],
            bbox=BBox(0, 50, 50, 70),
        )
        page = Page(blocks=[block1, block2], width=400, height=300)
        assert page.text == "first\n\nsecond"


class TestOCRResult:
    def _make_result(self):
        word = Word("hello", BBox(0, 0, 50, 20), 0.95)
        line = Line(words=[word], bbox=BBox(0, 0, 50, 20))
        block = Block(lines=[line], bbox=BBox(0, 0, 50, 20))
        page = Page(blocks=[block], width=400, height=300)
        return OCRResult(pages=[page])

    def test_to_text(self):
        result = self._make_result()
        assert result.to_text() == "hello"

    def test_to_dict_structure(self):
        result = self._make_result()
        d = result.to_dict()
        assert isinstance(d, dict)
        assert len(d["pages"]) == 1
        assert d["pages"][0]["blocks"][0]["lines"][0]["words"][0]["text"] == "hello"
        assert d["pages"][0]["blocks"][0]["lines"][0]["words"][0]["confidence"] == 0.95
        assert "bbox" in d["pages"][0]["blocks"][0]["lines"][0]["words"][0]

    def test_to_markdown_plain(self):
        result = self._make_result()
        md = result.to_markdown()
        assert "hello" in md

    def test_to_html_plain(self):
        result = self._make_result()
        html = result.to_html()
        assert "hello" in html

    def test_empty_result(self):
        result = OCRResult(pages=[])
        assert result.to_text() == ""
        assert result.to_dict() == {"pages": []}
        assert result.to_markdown() == ""
        assert result.to_html() == ""
