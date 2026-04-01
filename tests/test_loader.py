import fitz
import pytest
from PIL import Image

from qira_ocr.loader import Document, DocumentLoader, _parse_page_range


class TestDocumentFromImage:
    def test_load_png(self, sample_image):
        doc = DocumentLoader.load(sample_image)
        assert isinstance(doc, Document)
        assert len(doc.pages) == 1
        assert isinstance(doc.pages[0], Image.Image)
        assert doc.extracted_text is None

    def test_load_pil_image(self):
        img = Image.new("RGB", (200, 100), color="white")
        doc = DocumentLoader.load(img)
        assert len(doc.pages) == 1

    def test_load_bytes(self, sample_image):
        data = sample_image.read_bytes()
        doc = DocumentLoader.load(data)
        assert len(doc.pages) == 1


class TestDocumentFromPDF:
    @pytest.fixture
    def text_pdf(self, tmp_path):
        """Create a PDF with an actual text layer."""
        path = tmp_path / "text.pdf"
        doc = fitz.open()
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 72), "Hello from PDF", fontsize=12)
        doc.save(str(path))
        doc.close()
        return path

    @pytest.fixture
    def scanned_pdf(self, tmp_path):
        """Create a PDF with only an image (no text layer)."""
        path = tmp_path / "scanned.pdf"
        img = Image.new("RGB", (612, 792), color="white")
        img.save(str(path.with_suffix(".png")))
        doc = fitz.open()
        page = doc.new_page(width=612, height=792)
        page.insert_image(fitz.Rect(0, 0, 612, 792), filename=str(path.with_suffix(".png")))
        doc.save(str(path))
        doc.close()
        return path

    def test_text_pdf_extracts_text(self, text_pdf):
        doc = DocumentLoader.load(text_pdf)
        assert doc.extracted_text is not None
        assert "Hello from PDF" in doc.extracted_text
        assert len(doc.pages) == 0  # no need to OCR

    def test_scanned_pdf_rasterizes(self, scanned_pdf):
        doc = DocumentLoader.load(scanned_pdf)
        assert doc.extracted_text is None
        assert len(doc.pages) == 1
        assert isinstance(doc.pages[0], Image.Image)

    def test_page_range(self, text_pdf):
        doc = DocumentLoader.load(text_pdf, pages="1")
        assert doc.extracted_text is not None


class TestDocumentLoaderErrors:
    def test_unsupported_format(self, tmp_path):
        bad_file = tmp_path / "data.xyz"
        bad_file.write_text("not an image")
        with pytest.raises(ValueError, match="Unsupported"):
            DocumentLoader.load(bad_file)

    def test_nonexistent_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            DocumentLoader.load(tmp_path / "missing.png")


class TestParsePageRange:
    def test_single_page(self):
        assert _parse_page_range("3", 10) == [2]

    def test_range(self):
        assert _parse_page_range("1-3", 10) == [0, 1, 2]

    def test_comma_separated(self):
        assert _parse_page_range("1,3,5", 10) == [0, 2, 4]

    def test_mixed(self):
        assert _parse_page_range("1-3,5", 10) == [0, 1, 2, 4]

    def test_out_of_bounds(self):
        assert _parse_page_range("15", 10) == []

    def test_range_clamped(self):
        assert _parse_page_range("8-12", 10) == [7, 8, 9]


class TestMultiPagePDF:
    @pytest.fixture
    def three_page_pdf(self, tmp_path):
        """Create a 3-page text-layer PDF with distinct text on each page."""
        path = tmp_path / "multipage.pdf"
        pdf = fitz.open()
        for text in ["Page one content", "Page two content", "Page three content"]:
            page = pdf.new_page(width=612, height=792)
            page.insert_text((72, 72), text, fontsize=12)
        pdf.save(str(path))
        pdf.close()
        return path

    def test_load_all_pages(self, three_page_pdf):
        doc = DocumentLoader.load(three_page_pdf)
        assert doc.extracted_text is not None
        assert "Page one content" in doc.extracted_text
        assert "Page two content" in doc.extracted_text
        assert "Page three content" in doc.extracted_text

    def test_load_single_page(self, three_page_pdf):
        doc = DocumentLoader.load(three_page_pdf, pages="2")
        assert doc.extracted_text is not None
        assert "Page two content" in doc.extracted_text
        assert "Page one content" not in doc.extracted_text
        assert "Page three content" not in doc.extracted_text

    def test_load_page_range(self, three_page_pdf):
        doc = DocumentLoader.load(three_page_pdf, pages="1-2")
        assert doc.extracted_text is not None
        assert "Page one content" in doc.extracted_text
        assert "Page two content" in doc.extracted_text
        assert "Page three content" not in doc.extracted_text
