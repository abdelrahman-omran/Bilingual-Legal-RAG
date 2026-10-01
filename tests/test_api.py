from fastapi.testclient import TestClient
from legal_rag.api.app import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["documents_indexed"] == 1149

def test_ask_empty_question_returns_422():
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422