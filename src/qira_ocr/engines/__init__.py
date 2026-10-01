from qira_ocr.engines.base import OCREngine
from qira_ocr.engines.paddle import PaddleEngine
from qira_ocr.engines.surya import SuryaEngine

__all__ = ["OCREngine", "PaddleEngine", "QariEngine", "SuryaEngine"]


# Lazy wrapper to avoid a hard dependency on qwen_vl_utils (qari extra).
def QariEngine():
    from qira_ocr.engines.qari import QariEngine as _QariEngine

    return _QariEngine()
