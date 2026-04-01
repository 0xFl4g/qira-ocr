import pytest
from PIL import Image

from qira_ocr.structure import StructureAnalyzer, _html_table_to_markdown


class TestStructureAnalyzer:
    def test_analyze_returns_ocr_result(self, sample_image):
        analyzer = StructureAnalyzer()
        img = Image.open(sample_image)
        result = analyzer.analyze(img)
        from qira_ocr.result import OCRResult
        assert isinstance(result, OCRResult)

    def test_analyze_empty_image(self):
        analyzer = StructureAnalyzer()
        img = Image.new("RGB", (100, 100), color="white")
        result = analyzer.analyze(img)
        from qira_ocr.result import OCRResult
        assert isinstance(result, OCRResult)


class TestHtmlTableToMarkdown:
    def test_empty_string(self):
        assert _html_table_to_markdown("") == ""

    def test_simple_2x2_table(self):
        html = "<table><tr><td>A</td><td>B</td></tr><tr><td>C</td><td>D</td></tr></table>"
        result = _html_table_to_markdown(html)
        lines = result.splitlines()
        assert lines[0] == "| A | B |"
        assert lines[1] == "| --- | --- |"
        assert lines[2] == "| C | D |"

    def test_table_with_th_headers(self):
        html = "<table><tr><th>Name</th><th>Age</th></tr><tr><td>Alice</td><td>30</td></tr></table>"
        result = _html_table_to_markdown(html)
        lines = result.splitlines()
        assert lines[0] == "| Name | Age |"
        assert lines[1] == "| --- | --- |"
        assert lines[2] == "| Alice | 30 |"

    def test_non_table_html_returned_unchanged(self):
        html = "<p>No table here</p>"
        assert _html_table_to_markdown(html) == html

    def test_nested_tags_stripped(self):
        html = "<table><tr><td><b>Bold</b></td><td><em>Italic</em></td></tr></table>"
        result = _html_table_to_markdown(html)
        lines = result.splitlines()
        assert lines[0] == "| Bold | Italic |"
        assert lines[1] == "| --- | --- |"
