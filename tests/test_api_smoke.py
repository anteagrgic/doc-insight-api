from fastapi.testclient import TestClient
from app.main import app  # prilagođeno importu iz app paketa

client = TestClient(app)


def test_health_and_upload_and_ask():
    # health
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"

    # upload (txt)
    files = [("files", ("note.txt", b"The capital of France is Paris.", "text/plain"))]
    r = client.post("/upload", files=files)
    assert r.status_code == 200
    up = r.json()
    assert up["chunks_indexed"] >= 1

    # ask
    r = client.post("/ask", json={"question": "What is the capital of France?", "k": 3})
    assert r.status_code == 200
    data = r.json()

    # answer mora sadržavati "Paris"
    assert "Paris" in data["answer"]
    assert len(data["sources"]) >= 1

    # provjera metadata.page i metadata.source
    first_source = data["sources"][0]
    assert "metadata" in first_source
    meta = first_source["metadata"]
    assert "page" in meta and int(meta["page"]) >= 1
    assert "source" in meta and meta["source"] == "note.txt"
