from __future__ import annotations

import unicodedata

from qira_ocr.engines.base import OCREngine
from qira_ocr.engines.paddle import PaddleEngine
from qira_ocr.engines.surya import SuryaEngine

ARABIC_THRESHOLD = 0.30


def detect_arabic_ratio(text: str) -> float:
    """Return the ratio of Arabic script characters to total alphabetic characters."""
    if not text:
        return 0.0
    arabic_count = 0
    alpha_count = 0
    for char in text:
        if unicodedata.category(char).startswith("L"):
            alpha_count += 1
            if "\u0600" <= char <= "\u06ff" or "\u0750" <= char <= "\u077f" or "\ufb50" <= char <= "\ufdff" or "\ufe70" <= char <= "\ufeff":
                arabic_count += 1
    if alpha_count == 0:
        return 0.0
    return arabic_count / alpha_count


class EngineRouter:
    def __init__(self) -> None:
        self._paddle: PaddleEngine | None = None
        self._surya: SuryaEngine | None = None

    def _get_paddle(self) -> PaddleEngine:
        if self._paddle is None:
            self._paddle = PaddleEngine()
        return self._paddle

    def _get_surya(self) -> SuryaEngine:
        if self._surya is None:
            self._surya = SuryaEngine()
        return self._surya

    def select(
        self,
        engine: str = "auto",
        text_hint: str | None = None,
    ) -> OCREngine:
        if engine == "paddle":
            return self._get_paddle()
        elif engine == "surya":
            return self._get_surya()
        elif engine == "auto":
            if text_hint and detect_arabic_ratio(text_hint) >= ARABIC_THRESHOLD:
                return self._get_surya()
            return self._get_paddle()
        else:
            raise ValueError(f"Unknown engine: {engine!r}. Use 'auto', 'paddle', or 'surya'.")
