import fitz
import pytest
from click.testing import CliRunner

from qira_ocr.cli import main


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def text_pdf(tmp_path):
    path = tmp_path / "text.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "CLI test content", fontsize=12)
    doc.save(str(path))
    doc.close()
    return path


class TestCLIScan:
    def test_scan_pdf_text_output(self, runner, text_pdf):
        result = runner.invoke(main, ["scan", str(text_pdf)])
        assert result.exit_code == 0
        assert "CLI test content" in result.output

    def test_scan_with_format_markdown(self, runner, text_pdf):
        result = runner.invoke(main, ["scan", str(text_pdf), "--format", "markdown"])
        assert result.exit_code == 0

    def test_scan_with_format_json(self, runner, text_pdf):
        result = runner.invoke(main, ["scan", str(text_pdf), "--format", "json"])
        assert result.exit_code == 0
        import json
        data = json.loads(result.output)
        assert "pages" in data

    def test_scan_with_engine_override(self, runner, text_pdf):
        result = runner.invoke(main, ["scan", str(text_pdf), "--engine", "paddle"])
        assert result.exit_code == 0

    def test_scan_nonexistent_file(self, runner):
        result = runner.invoke(main, ["scan", "/nonexistent/file.pdf"])
        assert result.exit_code != 0

    def test_scan_output_to_file(self, runner, text_pdf, tmp_path):
        out = tmp_path / "output.txt"
        result = runner.invoke(main, ["scan", str(text_pdf), "--output", str(out)])
        assert result.exit_code == 0
        assert out.exists()
        assert "CLI test content" in out.read_text()


class TestCLIBatchScan:
    @pytest.fixture
    def pdf_dir(self, tmp_path):
        """Create a directory with two text-layer PDFs."""
        for name, text in [("alpha.pdf", "Alpha document text"), ("beta.pdf", "Beta document text")]:
            path = tmp_path / name
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((72, 72), text, fontsize=12)
            doc.save(str(path))
            doc.close()
        return tmp_path

    def test_scan_directory_outputs_both_files(self, runner, pdf_dir):
        result = runner.invoke(main, ["scan", str(pdf_dir)])
        assert result.exit_code == 0
        assert "Alpha document text" in result.output
        assert "Beta document text" in result.output

    def test_scan_directory_with_output_dir(self, runner, pdf_dir, tmp_path):
        out_dir = tmp_path / "out"
        result = runner.invoke(main, ["scan", str(pdf_dir), "--output", str(out_dir), "--format", "text"])
        assert result.exit_code == 0
        assert out_dir.exists()
        alpha_out = out_dir / "alpha.txt"
        beta_out = out_dir / "beta.txt"
        assert alpha_out.exists()
        assert beta_out.exists()
        assert "Alpha document text" in alpha_out.read_text(encoding="utf-8")
        assert "Beta document text" in beta_out.read_text(encoding="utf-8")
