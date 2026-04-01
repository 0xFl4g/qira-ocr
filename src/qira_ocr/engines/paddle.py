from __future__ import annotations

import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

from qira_ocr.result import BBox, Block, Line, OCRResult, Page, Word


class PaddleEngine:
    def __init__(self, lang: str = "en") -> None:
        self._lang = lang
        self._ocr: PaddleOCR | None = None

    def _get_ocr(self) -> PaddleOCR:
        if self._ocr is None:
            self._ocr = PaddleOCR(
                use_textline_orientation=True,
                lang=self._lang,
            )
        return self._ocr

    def recognize(self, image: Image.Image) -> OCRResult:
        ocr = self._get_ocr()
        img_array = np.array(image)
        results = ocr.predict(img_array)

        if not results:
            page = Page(blocks=[], width=image.width, height=image.height)
            return OCRResult(pages=[page])

        result = results[0]
        rec_texts = result.get("rec_texts", [])
        rec_scores = result.get("rec_scores", [])
        rec_boxes = result.get("rec_boxes", [])

        if not rec_texts:
            page = Page(blocks=[], width=image.width, height=image.height)
            return OCRResult(pages=[page])

        blocks: list[Block] = []
        for text_str, conf, box in zip(rec_texts, rec_scores, rec_boxes):
            # rec_boxes are [x1, y1, x2, y2] axis-aligned bounding boxes
            bbox = BBox(
                x1=float(box[0]),
                y1=float(box[1]),
                x2=float(box[2]),
                y2=float(box[3]),
            )

            word = Word(text=text_str, bbox=bbox, confidence=float(conf))
            line = Line(words=[word], bbox=bbox)
            block = Block(lines=[line], bbox=bbox)
            blocks.append(block)

        page = Page(blocks=blocks, width=image.width, height=image.height)
        return OCRResult(pages=[page])
