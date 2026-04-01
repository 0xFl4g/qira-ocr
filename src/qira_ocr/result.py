from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def to_dict(self) -> dict:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}


@dataclass(slots=True)
class Word:
    text: str
    bbox: BBox
    confidence: float

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "bbox": self.bbox.to_dict(),
            "confidence": self.confidence,
        }


@dataclass(slots=True)
class Line:
    words: list[Word]
    bbox: BBox

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def confidence(self) -> float:
        if not self.words:
            return 0.0
        return sum(w.confidence for w in self.words) / len(self.words)

    def to_dict(self) -> dict:
        return {
            "words": [w.to_dict() for w in self.words],
            "bbox": self.bbox.to_dict(),
            "text": self.text,
            "confidence": self.confidence,
        }


@dataclass(slots=True)
class Block:
    lines: list[Line]
    bbox: BBox
    block_type: str = "text"

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    @property
    def confidence(self) -> float:
        if not self.lines:
            return 0.0
        return sum(line.confidence for line in self.lines) / len(self.lines)

    def to_dict(self) -> dict:
        return {
            "lines": [line.to_dict() for line in self.lines],
            "bbox": self.bbox.to_dict(),
            "block_type": self.block_type,
            "text": self.text,
            "confidence": self.confidence,
        }


@dataclass(slots=True)
class Page:
    blocks: list[Block]
    width: int
    height: int

    @property
    def text(self) -> str:
        return "\n\n".join(block.text for block in self.blocks)

    @property
    def confidence(self) -> float:
        if not self.blocks:
            return 0.0
        return sum(b.confidence for b in self.blocks) / len(self.blocks)

    def to_dict(self) -> dict:
        return {
            "blocks": [b.to_dict() for b in self.blocks],
            "width": self.width,
            "height": self.height,
            "text": self.text,
            "confidence": self.confidence,
        }


@dataclass(slots=True)
class OCRResult:
    pages: list[Page]

    def to_text(self) -> str:
        return "\n\n".join(page.text for page in self.pages).strip()

    def to_dict(self) -> dict:
        return {"pages": [p.to_dict() for p in self.pages]}

    def to_markdown(self) -> str:
        parts: list[str] = []
        for page in self.pages:
            for block in page.blocks:
                if block.block_type == "table" and hasattr(block, "table_markdown"):
                    parts.append(block.table_markdown)
                else:
                    parts.append(block.text)
        return "\n\n".join(parts).strip()

    def to_html(self) -> str:
        parts: list[str] = []
        for page in self.pages:
            for block in page.blocks:
                if block.block_type == "table" and hasattr(block, "table_html"):
                    parts.append(block.table_html)
                else:
                    parts.append(f"<p>{block.text}</p>")
        return "\n".join(parts).strip()
