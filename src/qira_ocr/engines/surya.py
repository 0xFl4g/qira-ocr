from __future__ import annotations

from PIL import Image
from surya.recognition import RecognitionPredictor, FoundationPredictor
from surya.detection import DetectionPredictor
from surya.common.surya.schema import TaskNames

from qira_ocr.result import BBox, Block, Line, OCRResult, Page, Word


class SuryaEngine:
    def __init__(self, langs: list[str] | None = None) -> None:
        self._langs = langs or ["ar", "en"]
        self._recognition_predictor: RecognitionPredictor | None = None
        self._detection_predictor: DetectionPredictor | None = None

    def _get_predictors(self) -> tuple[RecognitionPredictor, DetectionPredictor]:
        if self._recognition_predictor is None:
            foundation_predictor = FoundationPredictor()
            self._recognition_predictor = RecognitionPredictor(foundation_predictor)
            self._detection_predictor = DetectionPredictor()
        return self._recognition_predictor, self._detection_predictor

    def recognize(self, image: Image.Image) -> OCRResult:
        recognition_predictor, detection_predictor = self._get_predictors()

        predictions = recognition_predictor(
            [image],
            task_names=[TaskNames.ocr_without_boxes],
            det_predictor=detection_predictor,
        )

        if not predictions:
            page = Page(blocks=[], width=image.width, height=image.height)
            return OCRResult(pages=[page])

        page_pred = predictions[0]
        blocks: list[Block] = []

        for text_line in page_pred.text_lines:
            poly = text_line.polygon
            x_coords = [p[0] for p in poly]
            y_coords = [p[1] for p in poly]
            bbox = BBox(
                x1=min(x_coords),
                y1=min(y_coords),
                x2=max(x_coords),
                y2=max(y_coords),
            )

            conf = text_line.confidence if text_line.confidence is not None else 0.0
            word = Word(text=text_line.text, bbox=bbox, confidence=conf)
            line = Line(words=[word], bbox=bbox)
            block = Block(lines=[line], bbox=bbox)
            blocks.append(block)

        page = Page(blocks=blocks, width=image.width, height=image.height)
        return OCRResult(pages=[page])
