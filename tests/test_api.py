import fitz
import pytest
from fastapi.testclient import TestClient

from qira_ocr.api import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def text_pdf(tmp_path):
    path = tmp_path / "text.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "API test content", fontsize=12)
    doc.save(str(path))
    doc.close()
    return path


class TestHealthEndpoint:
    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "engines" in data
        assert "paddle" in data["engines"]
        assert "surya" in data["engines"]


class TestOCREndpoint:
    def test_ocr_pdf(self, client, text_pdf):
        with open(text_pdf, "rb") as f:
            response = client.post("/ocr", files={"file": ("test.pdf", f, "application/pdf")})
        assert response.status_code == 200
        data = response.json()
        assert "pages" in data
        assert "API test content" in data["pages"][0]["blocks"][0]["text"]

    def test_ocr_with_engine(self, client, text_pdf):
        with open(text_pdf, "rb") as f:
            response = client.post(
                "/ocr",
                files={"file": ("test.pdf", f, "application/pdf")},
                data={"engine": "paddle"},
            )
        assert response.status_code == 200

    def test_ocr_with_format(self, client, text_pdf):
        with open(text_pdf, "rb") as f:
            response = client.post(
                "/ocr",
                files={"file": ("test.pdf", f, "application/pdf")},
                data={"format": "markdown"},
            )
        assert response.status_code == 200
        assert "text" in response.json()

    def test_ocr_no_file(self, client):
        response = client.post("/ocr")
        assert response.status_code == 422
