import os
import unittest

from fastapi.testclient import TestClient

os.environ["MEDICAL_DATASET_PATH"] = "backend/data/sample_medical_knowledge.json"

from app.main import app  # noqa: E402


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Use TestClient as a context manager to trigger the lifespan
        # (startup / shutdown) so that app.state.pipeline is initialised.
        cls._ctx = TestClient(app)
        cls.client = cls._ctx.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._ctx.__exit__(None, None, None)

    # ------------------------------------------------------------------
    # /health
    # ------------------------------------------------------------------
    def test_health_ok(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertTrue(body["rag_ready"])
        self.assertIn("knowledge_entries", body)
        self.assertGreater(body["knowledge_entries"], 5)
        self.assertEqual(body["retriever_type"], "BM25")

    def test_health_has_request_id(self):
        response = self.client.get("/health")
        self.assertIn("X-Request-ID", response.headers)
        self.assertGreater(len(response.headers["X-Request-ID"]), 0)

    # ------------------------------------------------------------------
    # /ask
    # ------------------------------------------------------------------
    def test_ask_emergency_path(self):
        response = self.client.post("/ask", json={"query": "I have chest pain and sweating"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["severity"], "high")
        self.assertIn("emergency", body["recommended_action"].lower())
        self.assertIn("not medical advice", body["disclaimer"].lower())

    def test_ask_returns_contract(self):
        response = self.client.post("/ask", json={"query": "I have a runny nose and sneezing"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        required = {
            "possible_conditions",
            "explanation",
            "severity",
            "recommended_action",
            "when_to_see_doctor",
            "confidence",
            "disclaimer",
        }
        self.assertTrue(required.issubset(set(body.keys())))

    def test_ask_query_too_short(self):
        response = self.client.post("/ask", json={"query": "x"})
        self.assertEqual(response.status_code, 422)

    def test_ask_severity_valid_enum(self):
        response = self.client.post("/ask", json={"query": "I have a mild headache"})
        self.assertIn(response.json()["severity"], ("low", "medium", "high"))

    def test_ask_confidence_valid_enum(self):
        response = self.client.post("/ask", json={"query": "I feel tired and have muscle aches"})
        self.assertIn(response.json()["confidence"], ("low", "medium", "high"))

    def test_ask_disclaimer_always_present(self):
        for query in [
            "I have chest pain",
            "runny nose sneezing",
            "random nonsense symptom xyzzy",
        ]:
            body = self.client.post("/ask", json={"query": query}).json()
            self.assertIn("not medical advice", body["disclaimer"].lower(), msg=f"query={query!r}")

    # ------------------------------------------------------------------
    # /symptom-check
    # ------------------------------------------------------------------
    def test_symptom_check_contract(self):
        response = self.client.post(
            "/symptom-check", json={"symptoms": ["fever", "body aches", "dry cough"]}
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        required = {
            "possible_conditions",
            "explanation",
            "severity",
            "recommended_action",
            "when_to_see_doctor",
            "confidence",
            "disclaimer",
        }
        self.assertTrue(required.issubset(set(body.keys())))
        self.assertIn("not medical advice", body["disclaimer"].lower())

    def test_symptom_check_red_flag(self):
        response = self.client.post(
            "/symptom-check",
            json={"symptoms": ["fever", "stiff neck", "severe headache"]},
        )
        body = response.json()
        self.assertEqual(body["severity"], "high")

    def test_symptom_check_empty_list_rejected(self):
        response = self.client.post("/symptom-check", json={"symptoms": []})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
