"""qira-ocr: Hybrid multi-engine OCR for Arabic and English documents."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from qira_ocr.loader import DocumentLoader
from qira_ocr.result import OCRResult, BBox, Block, Line, Page, Word
from qira_ocr.router import EngineRouter

__all__ = ["OCR", "OCRResult", "BBox", "Block", "Line", "Page", "Word"]


class OCR:
    def __init__(self) -> None:
        self._router = EngineRouter()

    def read(
        self,
        source: str | Path | bytes | Image.Image,
        engine: str = "auto",
        pages: str | None = None,
    ) -> OCRResult:
        doc = DocumentLoader.load(source, pages=pages)

        # If text was extracted directly from a text-layer PDF, wrap it in OCRResult
        if doc.extracted_text is not None:
            word = Word(text=doc.extracted_text, bbox=BBox(0, 0, 0, 0), confidence=1.0)
            line = Line(words=[word], bbox=BBox(0, 0, 0, 0))
            block = Block(lines=[line], bbox=BBox(0, 0, 0, 0))
            page = Page(blocks=[block], width=0, height=0)
            return OCRResult(pages=[page])

        # OCR each page image
        all_pages: list[Page] = []
        for page_image in doc.pages:
            text_hint = None
            ocr_engine = self._router.select(engine=engine, text_hint=text_hint)
            result = ocr_engine.recognize(page_image)
            all_pages.extend(result.pages)

        return OCRResult(pages=all_pages)
