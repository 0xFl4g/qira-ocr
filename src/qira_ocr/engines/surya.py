from __future__ import annotations

import html
import re

from PIL import Image
from surya.inference import SuryaInferenceManager
from surya.recognition import RecognitionPredictor

from qira_ocr.result import BBox, Block, Line, OCRResult, Page, Word

_LINE_BREAK = re.compile(r"<br\s*/?>|</(?:p|div|li|tr|h[1-6])>", re.IGNORECASE)
_CELL_END = re.compile(r"</t[dh]>", re.IGNORECASE)
_TAG = re.compile(r"<[^>]+>")


def _html_to_lines(fragment: str) -> list[str]:
    text = _CELL_END.sub(" ", _LINE_BREAK.sub("\n", fragment))
    text = html.unescape(_TAG.sub("", text))
    return [line.strip() for line in text.splitlines() if line.strip()]


class SuryaEngine:
    def __init__(self, langs: list[str] | None = None) -> None:
        self._langs = langs or ["ar", "en"]
        self._recognition_predictor: RecognitionPredictor | None = None

    def _get_predictor(self) -> RecognitionPredictor:
        # surya 0.20+ runs OCR on a VLM server (llama-server or vllm) that the
        # inference manager spawns lazily on the first call.
        if self._recognition_predictor is None:
            self._recognition_predictor = RecognitionPredictor(SuryaInferenceManager())
        return self._recognition_predictor

    def recognize(self, image: Image.Image) -> OCRResult:
        predictions = self._get_predictor()([image])

        if not predictions:
            page = Page(blocks=[], width=image.width, height=image.height)
            return OCRResult(pages=[page])

        blocks: list[Block] = []
        for block_pred in predictions[0].blocks:
            texts = _html_to_lines(block_pred.html)
            if not texts:  # skipped visual blocks, failed calls, empty regions
                continue
            bbox = BBox(*block_pred.bbox)
            conf = block_pred.confidence if block_pred.confidence is not None else 0.0
            lines = [
                Line(words=[Word(text=text, bbox=bbox, confidence=conf)], bbox=bbox)
                for text in texts
            ]
            blocks.append(Block(lines=lines, bbox=bbox))

        page = Page(blocks=blocks, width=image.width, height=image.height)
        return OCRResult(pages=[page])
