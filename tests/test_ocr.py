import fitz
import pytest
from PIL import Image

from qira_ocr import OCR
from qira_ocr.result import OCRResult


class TestOCRRead:
    def test_read_image_file(self, sample_image):
        ocr = OCR()
        result = ocr.read(sample_image)
        assert isinstance(result, OCRResult)

    def test_read_pil_image(self):
        img = Image.new("RGB", (200, 100), color="white")
        ocr = OCR()
        result = ocr.read(img)
        assert isinstance(result, OCRResult)

    def test_read_with_engine_override(self, sample_image):
        ocr = OCR()
        result = ocr.read(sample_image, engine="paddle")
        assert isinstance(result, OCRResult)

    def test_read_text_pdf(self, tmp_path):
        path = tmp_path / "text.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "PDF text content", fontsize=12)
        doc.save(str(path))
        doc.close()

        ocr = OCR()
        result = ocr.read(path)
        assert isinstance(result, OCRResult)
        assert "PDF text content" in result.to_text()

    def test_read_returns_exportable_result(self, sample_image):
        ocr = OCR()
        result = ocr.read(sample_image)
        assert isinstance(result, OCRResult)
        assert isinstance(result.to_markdown(), str)


class TestOCRFormats:
    def test_all_formats(self, sample_image):
        ocr = OCR()
        result = ocr.read(sample_image)
        assert isinstance(result.to_text(), str)
        assert isinstance(result.to_dict(), dict)
        assert isinstance(result.to_markdown(), str)
        assert isinstance(result.to_html(), str)
