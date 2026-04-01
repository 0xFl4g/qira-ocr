# qira-ocr

A hybrid multi-engine OCR toolkit for Arabic and English documents. Routes between [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) (fast, strong on printed text and tables), [QARI-OCR](https://huggingface.co/NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct) (state-of-the-art Arabic with diacritics), and [Surya](https://github.com/VikParuchuri/surya) (Arabic and handwriting fallback) behind a single unified interface. Supports PDFs, images, and batch processing with output as plain text, JSON, Markdown, or HTML.

Built as a university project at the University of Sheffield.

## Project Structure

```
src/qira_ocr/
  __init__.py          # Public OCR class — single entry point
  loader.py            # PDF text extraction, rasterization, image loading
  router.py            # Automatic engine selection via Arabic script detection
  result.py            # Structured result model (Page > Block > Line > Word)
  structure.py         # Table/layout analysis using PPStructure
  cli.py               # Click CLI (qira-ocr scan ...)
  api.py               # FastAPI REST API (/ocr, /health)
  engines/
    base.py            # OCREngine protocol
    paddle.py          # PaddleOCR wrapper
    qari.py            # QARI-OCR wrapper (Arabic VLM)
    surya.py           # Surya OCR wrapper
tests/                 # 73 tests covering all modules
```

## Installation

Requires Python 3.12+. Install with [uv](https://docs.astral.sh/uv/):

```bash
uv sync              # core OCR library
uv sync --extra cli  # + command-line interface
uv sync --extra api  # + REST API server
uv sync --extra qari # + QARI-OCR engine (Arabic VLM, needs GPU)
uv sync --extra all  # everything
```

## Usage

### Python

```python
from qira_ocr import OCR

ocr = OCR()

# Basic — auto-detects engine, extracts text from PDF text layers directly
result = ocr.read("invoice.pdf")
print(result.to_text())

# Force a specific engine
result = ocr.read("arabic_note.jpg", engine="qari")   # best Arabic quality
result = ocr.read("arabic_note.jpg", engine="surya")   # lighter Arabic fallback

# Export as structured data
for page in result.pages:
    for block in page.blocks:
        print(block.text, block.bbox, block.confidence)

# Other export formats
result.to_dict()      # nested dict with bounding boxes + confidence scores
result.to_markdown()  # markdown with table formatting
result.to_html()      # html with table formatting
```

### CLI

```bash
qira-ocr scan document.pdf
qira-ocr scan photo.jpg --engine surya --format markdown
qira-ocr scan ./documents/ --output ./results/ --format json
```

### REST API

```bash
uvicorn qira_ocr.api:app
```

```
POST /ocr    file upload, optional engine/format params
GET  /health engine availability status
```

## Engine Routing

When `engine="auto"` (default), qira-ocr inspects the document to pick the best engine:

| Condition | Engine | Why |
|-----------|--------|-----|
| PDF with text layer | Direct extraction | No OCR needed |
| Low Paddle confidence (non-Latin) | QARI-OCR | State-of-the-art Arabic, 0.061 CER |
| Low confidence, QARI not installed | Surya | Lighter Arabic/handwriting fallback |
| Everything else | PaddleOCR | Fastest, strong on printed English + tables |

Override with `engine="paddle"`, `engine="surya"`, or `engine="qari"`.

## License

[MIT](LICENSE)
