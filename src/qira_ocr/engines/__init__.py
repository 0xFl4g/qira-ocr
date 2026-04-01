from qira_ocr.engines.base import OCREngine
from qira_ocr.engines.paddle import PaddleEngine
from qira_ocr.engines.surya import SuryaEngine

__all__ = ["OCREngine", "PaddleEngine", "QariEngine", "SuryaEngine"]


def QariEngine():  # noqa: N802 — lazy wrapper to avoid hard dep on qwen_vl_utils
    from qira_ocr.engines.qari import QariEngine as _QariEngine
    return _QariEngine()
