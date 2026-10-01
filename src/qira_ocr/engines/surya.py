from __future__ import annotations

import html
import logging
import re

from PIL import Image
from surya.inference import SuryaInferenceManager
from surya.recognition import RecognitionPredictor

from qira_ocr.result import BBox, Block, Line, OCRResult, Page, Word

logger = logging.getLogger(__name__)

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

        block_preds = predictions[0].blocks
        attempted = [b for b in block_preds if not b.skipped]
        errors = sum(b.error for b in attempted)
        # surya turns inference request failures into error=True blocks with no
        # text; an all-error page means the backend is down, not an empty page.
        if errors and errors == len(attempted):
            raise RuntimeError(
                f"surya inference failed for all {errors} block(s); "
                "check llama-server/vllm or SURYA_INFERENCE_URL"
            )
        if errors:
            logger.warning(
                "surya inference failed for %d of %d block(s); their text is missing",
                errors,
                len(attempted),
            )

        blocks: list[Block] = []
        for block_pred in block_preds:
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
