import pytest
from fastapi.testclient import TestClient

from legal_rag.api.app import app


@pytest.fixture
def client():
    """Provides a TestClient context that triggers FastAPI lifespan startup/shutdown events."""
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    """Tests the /health endpoint status code and document count format."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "documents_indexed" in data


def test_ask_endpoint_success(client):
    """Tests a valid question submission to the /ask endpoint."""
    payload = {"question": "ما هي شروط العقد؟"}
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data


def test_ask_endpoint_empty_question_validation(client):
    """Tests that an empty question returns 422 Unprocessable Entity (Pydantic validation)."""
    payload = {"question": ""}
    response = client.post("/ask", json=payload)
    assert response.status_code == 422