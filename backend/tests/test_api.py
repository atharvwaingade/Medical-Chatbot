import os
import unittest

from fastapi.testclient import TestClient

os.environ["MEDICAL_DATASET_PATH"] = "backend/data/sample_medical_knowledge.json"

from app.main import app


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertTrue(body["rag_ready"])

    def test_ask_emergency_path(self):
        response = self.client.post("/ask", json={"query": "I have chest pain and sweating"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["severity"], "high")
        self.assertIn("emergency", body["recommended_action"].lower())
        self.assertEqual(body["disclaimer"], "This is not medical advice")

    def test_symptom_check_contract(self):
        response = self.client.post("/symptom-check", json={"symptoms": ["fever", "body aches", "dry cough"]})
        self.assertEqual(response.status_code, 200)
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
        self.assertTrue(expected.issubset(set(body.keys())))
        self.assertEqual(body["disclaimer"], "This is not medical advice")


if __name__ == "__main__":
    unittest.main()
