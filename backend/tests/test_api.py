import os

from fastapi.testclient import TestClient

os.environ["MEDICAL_DATASET_PATH"] = "backend/data/sample_medical_knowledge.json"

from app.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["rag_ready"] is True


def test_ask_emergency_path():
    response = client.post("/ask", json={"query": "I have chest pain and sweating"})
    assert response.status_code == 200
    body = response.json()
    assert body["severity"] == "high"
    assert "emergency" in body["recommended_action"].lower()
    assert body["disclaimer"] == "This is not medical advice"


def test_symptom_check_contract():
    response = client.post("/symptom-check", json={"symptoms": ["fever", "body aches", "dry cough"]})
    assert response.status_code == 200
    body = response.json()
    expected = {
        "possible_conditions",
        "explanation",
        "severity",
        "recommended_action",
        "when_to_see_doctor",
        "confidence",
        "disclaimer",
    }
    assert expected.issubset(set(body.keys()))
    assert body["disclaimer"] == "This is not medical advice"
