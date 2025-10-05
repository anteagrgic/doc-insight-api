import io
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_smoke():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert "status" in data and data["status"] == "ok"
    # prije uploada, indexi mogu biti 0
    assert "docs_indexed" in data


def test_upload_and_ask_with_source_in_metadata():
    # 1) Upload minimalnog "dokumenta"
    file_bytes = b"Neil Armstrong was the first person to walk on the Moon."
    files = [("files", ("moon.txt", file_bytes, "text/plain"))]
    r = client.post("/upload", files=files)
    assert r.status_code == 200, r.text
    up = r.json()
    assert up["chunks_indexed"] >= 1

    # 2) Ask (novi API: ključ je 'question', ne 'query')
    payload = {"question": "Who was the first person to walk on the Moon?", "k": 5}
    r = client.post("/ask", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()

    # 3) Provjere odgovora
    assert "answer" in data
    assert data["answer"] is not None
    # najčešće će sadržavati "Neil Armstrong"
    assert "sources" in data and isinstance(data["sources"], list) and len(data["sources"]) >= 1

    # 4) Provjeri metadata: page i source
    first_source = data["sources"][0]
    assert "metadata" in first_source and isinstance(first_source["metadata"], dict)
    meta = first_source["metadata"]
    assert "page" in meta and int(meta["page"]) >= 1
    assert "source" in meta and meta["source"] == "moon.txt"


def test_ner_entities_if_enabled():
    # Ovaj test NE pada ako je NER isključen; samo validira kada je ENABLE_NER=true
    payload = {"question": "Who was the first person to walk on the Moon?", "k": 5}
    r = client.post("/ask", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()

    # Ako je NER uključen, očekujemo PERSON entitet s "Armstrong"
    ents = data.get("entities")
    if ents is not None:
        assert any(e.get("label") == "PERSON" and "Armstrong" in e.get("text", "") for e in ents)
