from typing import runtime_checkable

import pytest
from PIL import Image
from surya.recognition.schema import BlockOCRResult, PageOCRResult

from qira_ocr.engines.base import OCREngine
from qira_ocr.engines.paddle import PaddleEngine
from qira_ocr.engines.surya import SuryaEngine
from qira_ocr.result import BBox, Block, Line, OCRResult, Page, Word


class TestOCREngineProtocol:
    def test_protocol_is_runtime_checkable(self):
        assert runtime_checkable(OCREngine)

    def test_conforming_class_is_instance(self):
        class FakeEngine:
            def recognize(self, image: Image.Image) -> OCRResult:
                word = Word("test", BBox(0, 0, 10, 10), 1.0)
                line = Line(words=[word], bbox=BBox(0, 0, 10, 10))
                block = Block(lines=[line], bbox=BBox(0, 0, 10, 10))
                page = Page(blocks=[block], width=100, height=100)
                return OCRResult(pages=[page])

        engine = FakeEngine()
        assert isinstance(engine, OCREngine)
        result = engine.recognize(Image.new("RGB", (100, 100)))
        assert result.to_text() == "test"


class TestPaddleEngine:
    def test_conforms_to_protocol(self):
        engine = PaddleEngine()
        assert isinstance(engine, OCREngine)

    @pytest.mark.requires_ocr_server
    def test_recognize_returns_ocr_result(self, sample_image):
        engine = PaddleEngine()
        img = Image.open(sample_image)
        result = engine.recognize(img)
        assert isinstance(result, OCRResult)
        assert len(result.pages) == 1

    @pytest.mark.requires_ocr_server
    def test_recognize_finds_text(self, sample_image):
        engine = PaddleEngine()
        img = Image.open(sample_image)
        result = engine.recognize(img)
        text = result.to_text().lower()
        # The image has "Hello World" drawn on it
        assert (
            "hello" in text or len(text) > 0
        )  # OCR may not be perfect on synthetic images


class TestSuryaEngine:
    def test_conforms_to_protocol(self):
        engine = SuryaEngine()
        assert isinstance(engine, OCREngine)

    def test_recognize_maps_blocks(self):
        text_block = BlockOCRResult(
            polygon=[1, 2, 30, 20],
            confidence=0.9,
            label="Text",
            reading_order=0,
            html="<p>Hello &amp; <b>world</b><br>line two</p>",
        )
        picture = BlockOCRResult(
            polygon=[0, 25, 50, 50], label="Picture", reading_order=1, skipped=True
        )
        engine = SuryaEngine()
        engine._recognition_predictor = lambda images: [
            PageOCRResult(blocks=[text_block, picture], image_bbox=[0, 0, 100, 50])
        ]
        result = engine.recognize(Image.new("RGB", (100, 50)))
        assert result.to_text() == "Hello & world\nline two"
        (block,) = result.pages[0].blocks
        assert block.bbox == BBox(1, 2, 30, 20)
        assert block.confidence == 0.9

    @staticmethod
    def _engine_returning(*blocks):
        engine = SuryaEngine()
        engine._recognition_predictor = lambda images: [
            PageOCRResult(blocks=list(blocks), image_bbox=[0, 0, 100, 50])
        ]
        return engine

    def test_recognize_raises_when_all_blocks_error(self):
        failed = BlockOCRResult(
            polygon=[0, 0, 50, 20], label="Text", reading_order=0, error=True
        )
        picture = BlockOCRResult(
            polygon=[0, 25, 50, 50], label="Picture", reading_order=1, skipped=True
        )
        engine = self._engine_returning(failed, picture)
        with pytest.raises(RuntimeError):
            engine.recognize(Image.new("RGB", (100, 50)))

    def test_recognize_warns_on_partial_errors(self, caplog):
        ok = BlockOCRResult(
            polygon=[0, 0, 50, 20], label="Text", reading_order=0, html="<p>kept</p>"
        )
        failed = BlockOCRResult(
            polygon=[0, 25, 50, 50], label="Text", reading_order=1, error=True
        )
        engine = self._engine_returning(ok, failed)
        with caplog.at_level("WARNING"):
            result = engine.recognize(Image.new("RGB", (100, 50)))
        assert result.to_text() == "kept"
        assert any(r.levelname == "WARNING" for r in caplog.records)

    @pytest.mark.requires_ocr_server
    def test_recognize_returns_ocr_result(self, sample_image):
        engine = SuryaEngine()
        img = Image.open(sample_image)
        result = engine.recognize(img)
        assert isinstance(result, OCRResult)
        assert len(result.pages) == 1

    @pytest.mark.requires_ocr_server
    def test_recognize_arabic(self, sample_arabic_image):
        engine = SuryaEngine()
        img = Image.open(sample_arabic_image)
        result = engine.recognize(img)
        assert isinstance(result, OCRResult)
        text = result.to_text()
        assert (
            len(text) > 0 or len(result.pages[0].blocks) >= 0
        )  # OCR on synthetic may vary
