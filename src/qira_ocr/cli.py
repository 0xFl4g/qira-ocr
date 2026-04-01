from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from qira_ocr import OCR


@click.group()
def main() -> None:
    """qira-ocr: Hybrid multi-engine OCR for Arabic and English documents."""


@main.command()
@click.argument("source", type=click.Path(exists=True))
@click.option("--engine", default="auto", type=click.Choice(["auto", "paddle", "surya", "qari"]), help="OCR engine to use.")
@click.option("--format", "fmt", default="text", type=click.Choice(["text", "markdown", "html", "json"]), help="Output format.")
@click.option("--pages", default=None, help="Page range for PDFs (e.g. '1-3').")
@click.option("--output", "-o", default=None, type=click.Path(), help="Output file path. Prints to stdout if not set.")
def scan(source: str, engine: str, fmt: str, pages: str | None, output: str | None) -> None:
    """Scan a file or directory for text."""
    source_path = Path(source)

    if source_path.is_dir():
        _scan_directory(source_path, engine, fmt, pages, output)
    else:
        _scan_file(source_path, engine, fmt, pages, output)


def _scan_file(path: Path, engine: str, fmt: str, pages: str | None, output: str | None) -> None:
    ocr = OCR()
    result = ocr.read(path, engine=engine, pages=pages)

    if fmt == "text":
        text = result.to_text()
    elif fmt == "markdown":
        text = result.to_markdown()
    elif fmt == "html":
        text = result.to_html()
    elif fmt == "json":
        text = json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
    else:
        text = result.to_text()

    if output:
        Path(output).write_text(text, encoding="utf-8")
    else:
        click.echo(text)


def _scan_directory(dir_path: Path, engine: str, fmt: str, pages: str | None, output: str | None) -> None:
    extensions = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp", ".pdf"}
    files = sorted(f for f in dir_path.iterdir() if f.suffix.lower() in extensions)

    if not files:
        click.echo(f"No supported files found in {dir_path}", err=True)
        sys.exit(1)

    output_dir = Path(output) if output else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    ext_map = {"text": ".txt", "markdown": ".md", "html": ".html", "json": ".json"}

    for file in files:
        out_path = str(output_dir / (file.stem + ext_map.get(fmt, ".txt"))) if output_dir else None
        _scan_file(file, engine, fmt, pages, out_path)
        if not out_path:
            click.echo("---")
