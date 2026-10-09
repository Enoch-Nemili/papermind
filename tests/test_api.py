"""Tests for the HTTP upload endpoint (FastAPI)."""

from fastapi.testclient import TestClient

import app.main as api


def setup_api(tmp_path, monkeypatch, ingest=lambda path: 3):
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(api, "DATA_DIR", data)
    monkeypatch.setattr(api, "ingest_pdf_file", ingest)
    return data, TestClient(api.app)


def test_upload_cannot_write_outside_data_dir(tmp_path, monkeypatch, make_pdf):
    _, client = setup_api(tmp_path, monkeypatch)
    pdf = make_pdf(tmp_path / "x.pdf").read_bytes()

    client.post("/upload", files={"file": ("../evil.pdf", pdf, "application/pdf")})

    assert not (tmp_path / "evil.pdf").exists(), "upload escaped data/ via the filename"


def test_upload_saves_safe_name_inside_data_dir(tmp_path, monkeypatch, make_pdf):
    data, client = setup_api(tmp_path, monkeypatch)
    pdf = make_pdf(tmp_path / "x.pdf").read_bytes()

    resp = client.post("/upload", files={"file": ("../../evil.pdf", pdf, "application/pdf")})

    assert resp.json() == {"ok": True, "filename": "evil.pdf", "chunks_added": 3}
    assert (data / "evil.pdf").exists()  # landed safely inside data/


def test_upload_rejects_file_only_pretending_to_be_pdf(tmp_path, monkeypatch):
    data, client = setup_api(tmp_path, monkeypatch)
    resp = client.post("/upload", files={"file": ("fake.pdf", b"not a pdf", "application/pdf")})
    assert resp.json()["ok"] is False
    assert list(data.iterdir()) == []


def test_upload_rolls_back_when_indexing_fails(tmp_path, monkeypatch, make_pdf):
    def broken(path):
        raise RuntimeError("database unavailable")

    data, client = setup_api(tmp_path, monkeypatch, ingest=broken)
    pdf = make_pdf(tmp_path / "x.pdf").read_bytes()

    resp = client.post("/upload", files={"file": ("paper.pdf", pdf, "application/pdf")})

    assert resp.json()["ok"] is False
    assert not (data / "paper.pdf").exists()
