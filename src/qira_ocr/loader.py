from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import fitz
from PIL import Image


SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}


@dataclass
class Document:
    pages: list[Image.Image] = field(default_factory=list)
    extracted_text: str | None = None


def _parse_page_range(pages_str: str, total: int) -> list[int]:
    """Parse page range string like '1-3' or '2' into 0-based indices."""
    result: list[int] = []
    for part in pages_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            start_idx = int(start) - 1
            end_idx = int(end)
            result.extend(range(max(0, start_idx), min(total, end_idx)))
        else:
            idx = int(part) - 1
            if 0 <= idx < total:
                result.append(idx)
    return result


class DocumentLoader:
    @staticmethod
    def load(
        source: str | Path | bytes | Image.Image,
        pages: str | None = None,
    ) -> Document:
        if isinstance(source, Image.Image):
            return Document(pages=[source])

        if isinstance(source, bytes):
            import io
            img = Image.open(io.BytesIO(source))
            return Document(pages=[img])

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if path.suffix.lower() == ".pdf":
            return DocumentLoader._load_pdf(path, pages)

        if path.suffix.lower() in SUPPORTED_IMAGE_EXTS:
            img = Image.open(path)
            return Document(pages=[img])

        raise ValueError(f"Unsupported file format: {path.suffix}")

    @staticmethod
    def _load_pdf(path: Path, pages: str | None = None) -> Document:
        pdf = fitz.open(str(path))
        total = len(pdf)
        page_indices = _parse_page_range(pages, total) if pages else list(range(total))

        # Check if PDF has a text layer
        text_parts: list[str] = []
        has_text = False
        for idx in page_indices:
            page = pdf[idx]
            text = page.get_text().strip()
            if text:
                has_text = True
                text_parts.append(text)

        if has_text:
            pdf.close()
            return Document(extracted_text="\n\n".join(text_parts))

        # No text layer — rasterize pages for OCR
        images: list[Image.Image] = []
        for idx in page_indices:
            page = pdf[idx]
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            images.append(img)

        pdf.close()
        return Document(pages=images)
