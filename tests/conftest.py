from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir():
    return FIXTURES_DIR


@pytest.fixture
def sample_image(tmp_path):
    """Create a simple test image with English text drawn on it."""
    img = Image.new("RGB", (400, 100), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), "Hello World", fill="black")
    path = tmp_path / "test.png"
    img.save(path)
    return path


@pytest.fixture
def sample_arabic_image(tmp_path):
    """Create a simple test image with Arabic text drawn on it."""
    img = Image.new("RGB", (400, 100), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), "مرحبا بالعالم", fill="black")
    path = tmp_path / "test_arabic.png"
    img.save(path)
    return path
