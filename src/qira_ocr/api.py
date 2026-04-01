from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

from qira_ocr import OCR

app = FastAPI(title="qira-ocr", description="Hybrid multi-engine OCR API")
_ocr = OCR()


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "engines": {
            "paddle": "available",
            "surya": "available",
        },
    }


@app.post("/ocr")
async def ocr_endpoint(
    file: UploadFile = File(...),
    engine: str = Form("auto"),
    format: str = Form("json"),
) -> JSONResponse:
    contents = await file.read()
    suffix = ""
    if file.filename:
        suffix = "." + file.filename.rsplit(".", 1)[-1] if "." in file.filename else ""

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = Path(tmp.name)

    try:
        result = _ocr.read(tmp_path, engine=engine)
    finally:
        tmp_path.unlink(missing_ok=True)

    if format == "text":
        return JSONResponse({"text": result.to_text()})
    elif format == "markdown":
        return JSONResponse({"text": result.to_markdown()})
    elif format == "html":
        return JSONResponse({"text": result.to_html()})
    else:
        return JSONResponse(result.to_dict())
