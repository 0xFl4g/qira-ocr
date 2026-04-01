import pytest
from PIL import Image

from qira_ocr.router import EngineRouter, detect_arabic_ratio


class TestDetectArabicRatio:
    def test_english_only(self):
        ratio = detect_arabic_ratio("Hello World")
        assert ratio == 0.0

    def test_arabic_only(self):
        ratio = detect_arabic_ratio("مرحبا بالعالم")
        assert ratio > 0.9

    def test_mixed(self):
        ratio = detect_arabic_ratio("Hello مرحبا World عالم")
        assert 0.2 < ratio < 0.8

    def test_empty(self):
        ratio = detect_arabic_ratio("")
        assert ratio == 0.0


class TestEngineRouter:
    def test_manual_paddle(self):
        router = EngineRouter()
        engine = router.select(engine="paddle")
        from qira_ocr.engines.paddle import PaddleEngine
        assert isinstance(engine, PaddleEngine)

    def test_manual_surya(self):
        router = EngineRouter()
        engine = router.select(engine="surya")
        from qira_ocr.engines.surya import SuryaEngine
        assert isinstance(engine, SuryaEngine)

    def test_auto_defaults_to_paddle(self):
        router = EngineRouter()
        engine = router.select(engine="auto")
        from qira_ocr.engines.paddle import PaddleEngine
        assert isinstance(engine, PaddleEngine)

    def test_auto_arabic_text_selects_surya(self):
        router = EngineRouter()
        engine = router.select(engine="auto", text_hint="مرحبا بالعالم هذا نص عربي طويل")
        from qira_ocr.engines.surya import SuryaEngine
        assert isinstance(engine, SuryaEngine)

    def test_auto_english_text_selects_paddle(self):
        router = EngineRouter()
        engine = router.select(engine="auto", text_hint="This is an English document with lots of text")
        from qira_ocr.engines.paddle import PaddleEngine
        assert isinstance(engine, PaddleEngine)

    def test_invalid_engine_raises(self):
        router = EngineRouter()
        with pytest.raises(ValueError, match="Unknown engine"):
            router.select(engine="invalid")
