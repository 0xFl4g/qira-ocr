from __future__ import annotations

from typing import Protocol, runtime_checkable

from PIL import Image

from qira_ocr.result import OCRResult


@runtime_checkable
class OCREngine(Protocol):
    def recognize(self, image: Image.Image) -> OCRResult: ...
