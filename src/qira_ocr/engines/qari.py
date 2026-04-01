from __future__ import annotations

from PIL import Image

from qira_ocr.result import BBox, Block, Line, OCRResult, Page, Word

_PROMPT = (
    "Below is the image of one page of a document. "
    "Just return the plain text representation of this document "
    "as if you were reading it naturally. Do not hallucinate."
)


class QariEngine:
    def __init__(self, max_tokens: int = 2048) -> None:
        self._max_tokens = max_tokens
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        model_name = "NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct"
        import torch

        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        self._processor = AutoProcessor.from_pretrained(model_name, use_fast=True)

    def recognize(self, image: Image.Image) -> OCRResult:
        self._load()
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": _PROMPT},
                ],
            }
        ]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(next(self._model.parameters()).device)

        generated_ids = self._model.generate(**inputs, max_new_tokens=self._max_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        if not output_text:
            page = Page(blocks=[], width=image.width, height=image.height)
            return OCRResult(pages=[page])

        bbox = BBox(0, 0, image.width, image.height)
        word = Word(text=output_text, bbox=bbox, confidence=1.0)
        line = Line(words=[word], bbox=bbox)
        block = Block(lines=[line], bbox=bbox)
        page = Page(blocks=[block], width=image.width, height=image.height)
        return OCRResult(pages=[page])
