from __future__ import annotations

import re

import numpy as np
from PIL import Image
from paddleocr import PPStructureV3

from qira_ocr.result import BBox, Block, Line, OCRResult, Page, Word


class StructureAnalyzer:
    def __init__(self) -> None:
        self._engine: PPStructureV3 | None = None

    def _get_engine(self) -> PPStructureV3:
        if self._engine is None:
            self._engine = PPStructureV3(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
            )
        return self._engine

    def analyze(self, image: Image.Image) -> OCRResult:
        engine = self._get_engine()
        img_array = np.array(image)
        results = engine.predict(img_array)

        if not results:
            page = Page(blocks=[], width=image.width, height=image.height)
            return OCRResult(pages=[page])

        # predict() returns a list of result objects (one per image)
        # We pass a single image, so take the first result
        raw = results[0]

        # Extract the parsing result list from the result object
        parsing_res_list = raw.get("parsing_res_list", []) if hasattr(raw, "get") else []

        if not parsing_res_list:
            page = Page(blocks=[], width=image.width, height=image.height)
            return OCRResult(pages=[page])

        blocks: list[Block] = []
        for region in parsing_res_list:
            label = getattr(region, "label", "text") or "text"
            box = getattr(region, "bbox", [0, 0, 0, 0])
            content = getattr(region, "content", "") or ""

            if len(box) >= 4:
                bbox = BBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3])
            else:
                bbox = BBox(x1=0, y1=0, x2=image.width, y2=image.height)

            if label == "table":
                block = Block(lines=[], bbox=bbox, block_type="table")
                block.table_html = content  # type: ignore[attr-defined]
                block.table_markdown = _html_table_to_markdown(content)  # type: ignore[attr-defined]
                blocks.append(block)
            else:
                lines: list[Line] = []
                if content:
                    word = Word(text=content, bbox=bbox, confidence=1.0)
                    line = Line(words=[word], bbox=bbox)
                    lines.append(line)
                block = Block(lines=lines, bbox=bbox, block_type=label)
                blocks.append(block)

        page = Page(blocks=blocks, width=image.width, height=image.height)
        return OCRResult(pages=[page])


def _html_table_to_markdown(html: str) -> str:
    """Best-effort conversion of simple HTML table to Markdown."""
    if not html:
        return ""

    rows: list[list[str]] = []
    for tr_match in re.finditer(r"<tr>(.*?)</tr>", html, re.DOTALL):
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr_match.group(1), re.DOTALL)
        cells = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
        rows.append(cells)

    if not rows:
        return html

    lines: list[str] = []
    for i, row in enumerate(rows):
        lines.append("| " + " | ".join(row) + " |")
        if i == 0:
            lines.append("| " + " | ".join("---" for _ in row) + " |")
    return "\n".join(lines)
