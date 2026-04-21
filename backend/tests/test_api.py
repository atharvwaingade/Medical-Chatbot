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
        self.assertEqual(body["retriever_type"], "MedRAG-Turbo (BM25+SCS+Prev+Causal+RAPTOR)")
        self.assertIn("active_sessions", body)

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

    def test_ask_returns_session_id(self):
        response = self.client.post("/ask", json={"query": "I have a runny nose and sneezing"})
        body = response.json()
        self.assertIn("session_id", body)
        self.assertIsNotNone(body["session_id"])

    def test_ask_returns_retrieval_metadata(self):
        response = self.client.post("/ask", json={"query": "fever and cough"})
        body = response.json()
        self.assertIn("retrieval_metadata", body)
        meta = body["retrieval_metadata"]
        self.assertIn("retrieval_entropy", meta)
        self.assertIn("retriever_type", meta)

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

    def test_ask_with_session_continuity(self):
        """Second ask with the same session_id should work without error."""
        r1 = self.client.post("/ask", json={"query": "I have a headache"})
        sid = r1.json()["session_id"]
        r2 = self.client.post("/ask", json={"query": "Still have headache", "session_id": sid})
        self.assertEqual(r2.status_code, 200)
        self.assertEqual(r2.json()["session_id"], sid)

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

    # ------------------------------------------------------------------
    # /differential
    # ------------------------------------------------------------------
    def test_differential_returns_list(self):
        response = self.client.post(
            "/differential", json={"symptoms": ["fever", "cough", "fatigue"]}
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("differentials", body)
        self.assertIsInstance(body["differentials"], list)
        self.assertGreater(len(body["differentials"]), 0)

    def test_differential_contract(self):
        response = self.client.post(
            "/differential", json={"symptoms": ["headache", "nausea"]}
        )
        body = response.json()
        self.assertIn("query_metadata", body)
        self.assertIn("disclaimer", body)
        first = body["differentials"][0]
        for field in ("rank", "condition", "hybrid_score", "evidence_tier",
                      "ruling_in_symptoms", "ruling_out_symptoms"):
            self.assertIn(field, first, f"Missing field: {field}")

    def test_differential_has_icd10(self):
        response = self.client.post(
            "/differential", json={"symptoms": ["fever", "cough"]}
        )
        for item in response.json()["differentials"]:
            self.assertIn("icd10", item)

    def test_differential_ranked_by_score(self):
        response = self.client.post(
            "/differential", json={"symptoms": ["fever", "body aches", "fatigue"]}
        )
        scores = [d["hybrid_score"] for d in response.json()["differentials"]]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_differential_query_metadata_entropy(self):
        response = self.client.post(
            "/differential", json={"symptoms": ["headache", "nausea", "light sensitivity"]}
        )
        meta = response.json()["query_metadata"]
        self.assertIn("retrieval_entropy", meta)
        self.assertGreaterEqual(meta["retrieval_entropy"], 0.0)
        self.assertLessEqual(meta["retrieval_entropy"], 1.0)


if __name__ == "__main__":
    unittest.main()
