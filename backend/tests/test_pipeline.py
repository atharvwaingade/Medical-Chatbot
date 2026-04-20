"""
Unit tests for RAGPipeline (no real network calls to Groq).

These tests exercise the pipeline in isolation by mocking the GroqClient.
"""
import asyncio
import os
import unittest
from unittest.mock import AsyncMock, MagicMock

os.environ["MEDICAL_DATASET_PATH"] = "backend/data/sample_medical_knowledge.json"

from app.rag.pipeline import RAGPipeline  # noqa: E402
from app.rag.session_store import SessionStore  # noqa: E402
from app.safety.validator import SAFE_DISCLAIMER  # noqa: E402


def _make_pipeline(groq_response=None, groq_raises=False):
    """Helper: build a RAGPipeline with a mocked GroqClient."""
    groq_client = MagicMock()
    if groq_raises:
        groq_client.generate = AsyncMock(side_effect=Exception("network error"))
    else:
        groq_client.generate = AsyncMock(return_value=groq_response)
    return RAGPipeline(
        dataset_path="backend/data/sample_medical_knowledge.json",
        top_k=3,
        groq_client=groq_client,
    )


def run(coro):
    """Run an async coroutine synchronously in tests."""
    return asyncio.run(coro)


class PipelineReadyTests(unittest.TestCase):
    def test_ready_true_when_entries_loaded(self):
        p = _make_pipeline()
        self.assertTrue(p.ready)

    def test_entries_count(self):
        p = _make_pipeline()
        self.assertGreater(len(p.dataset.entries), 5)

    def test_icd10_present_in_entries(self):
        p = _make_pipeline()
        for entry in p.dataset.entries:
            self.assertIn("icd10", entry, f"Missing icd10 in: {entry['condition']}")

    def test_prevalence_present_in_entries(self):
        p = _make_pipeline()
        valid = {"common", "very common", "uncommon", "rare"}
        for entry in p.dataset.entries:
            self.assertIn(entry.get("prevalence", ""), valid,
                          f"Invalid prevalence in: {entry['condition']}")


class PipelineEmergencyTests(unittest.TestCase):
    def test_emergency_returns_high_severity(self):
        p = _make_pipeline()
        result = run(p.answer("I have chest pain and cannot breathe"))
        self.assertEqual(result["severity"], "high")

    def test_emergency_disclaimer(self):
        p = _make_pipeline()
        result = run(p.answer("I have chest pain and cannot breathe"))
        self.assertIn("not medical advice", result["disclaimer"].lower())

    def test_emergency_bypasses_groq(self):
        groq = MagicMock()
        groq.generate = AsyncMock()
        p = RAGPipeline("backend/data/sample_medical_knowledge.json", 3, groq)
        run(p.answer("I am having a seizure"))
        groq.generate.assert_not_awaited()


class PipelineFallbackTests(unittest.TestCase):
    def test_groq_none_returns_fallback(self):
        p = _make_pipeline(groq_response=None)
        result = run(p.answer("runny nose and sneezing"))
        self.assertIn("not medical advice", result["disclaimer"].lower())
        self.assertIsInstance(result["possible_conditions"], list)
        self.assertTrue(len(result["possible_conditions"]) > 0)

    def test_groq_exception_returns_fallback(self):
        p = _make_pipeline(groq_raises=True)
        result = run(p.answer("runny nose and sneezing"))
        self.assertIn("not medical advice", result["disclaimer"].lower())


class PipelineRedFlagTests(unittest.TestCase):
    def test_red_flag_symptoms_set_high_severity(self):
        p = _make_pipeline(groq_response=None)
        result = run(p.answer("", symptoms=["fever", "stiff neck", "severe headache"]))
        self.assertEqual(result["severity"], "high")


class PipelineConditionGroundingTests(unittest.TestCase):
    def test_groq_hallucinated_condition_replaced(self):
        groq_resp = {
            "possible_conditions": ["Ebola Fever"],  # not in our dataset
            "explanation": "Some explanation.",
            "severity": "medium",
            "recommended_action": "See a doctor.",
            "when_to_see_doctor": "Today.",
            "confidence": "medium",
            "disclaimer": "Not medical advice.",
        }
        p = _make_pipeline(groq_response=groq_resp)
        result = run(p.answer("fever body aches fatigue dry cough"))
        self.assertNotIn("Ebola Fever", result["possible_conditions"])

    def test_low_confidence_prefixed(self):
        groq_resp = {
            "possible_conditions": ["Influenza"],
            "explanation": "Some explanation.",
            "severity": "medium",
            "recommended_action": "Rest.",
            "when_to_see_doctor": "If worse.",
            "confidence": "low",
            "disclaimer": "Not medical advice.",
        }
        p = _make_pipeline(groq_response=groq_resp)
        result = run(p.answer("fever body aches"))
        self.assertTrue(
            result["explanation"].startswith("I am not certain"),
            msg=f"Expected uncertainty prefix, got: {result['explanation']!r}",
        )


class PipelineSourcesTests(unittest.TestCase):
    def test_sources_attached_from_retrieved_docs(self):
        groq_resp = {
            "possible_conditions": ["Influenza"],
            "explanation": "Viral illness.",
            "severity": "medium",
            "recommended_action": "Rest.",
            "when_to_see_doctor": "If worse.",
            "confidence": "high",
            "disclaimer": "Not medical advice.",
        }
        p = _make_pipeline(groq_response=groq_resp)
        result = run(p.answer("fever body aches fatigue"))
        self.assertIn("sources", result)
        self.assertIsInstance(result["sources"], list)

    def test_fallback_sources_list(self):
        p = _make_pipeline(groq_response=None)
        result = run(p.answer("fever body aches"))
        self.assertIn("sources", result)


class PipelineValidatorTests(unittest.TestCase):
    def test_invalid_severity_coerced(self):
        groq_resp = {
            "possible_conditions": ["Influenza"],
            "explanation": "Viral.",
            "severity": "EXTREME",  # invalid
            "recommended_action": "Rest.",
            "when_to_see_doctor": "Soon.",
            "confidence": "high",
            "disclaimer": "Not medical advice.",
        }
        p = _make_pipeline(groq_response=groq_resp)
        result = run(p.answer("fever body aches"))
        self.assertIn(result["severity"], ("low", "medium", "high"))

    def test_invalid_confidence_coerced(self):
        groq_resp = {
            "possible_conditions": ["Influenza"],
            "explanation": "Viral.",
            "severity": "medium",
            "recommended_action": "Rest.",
            "when_to_see_doctor": "Soon.",
            "confidence": "uncertain",  # invalid
            "disclaimer": "Not medical advice.",
        }
        p = _make_pipeline(groq_response=groq_resp)
        result = run(p.answer("fever"))
        self.assertIn(result["confidence"], ("low", "medium", "high"))


class PipelineDifferentialTests(unittest.TestCase):
    """Tests for the new differential() method."""

    def test_differential_returns_list(self):
        p = _make_pipeline()
        result = run(p.differential(symptoms=["fever", "cough", "fatigue"]))
        self.assertIn("differentials", result)
        self.assertIsInstance(result["differentials"], list)
        self.assertGreater(len(result["differentials"]), 0)

    def test_differential_has_rank_field(self):
        p = _make_pipeline()
        result = run(p.differential(symptoms=["headache", "nausea"]))
        for item in result["differentials"]:
            self.assertIn("rank", item)

    def test_differential_has_icd10(self):
        p = _make_pipeline()
        result = run(p.differential(symptoms=["fever", "cough"]))
        for item in result["differentials"]:
            self.assertIn("icd10", item)

    def test_differential_has_evidence_tier(self):
        p = _make_pipeline()
        result = run(p.differential(symptoms=["fever", "cough"]))
        for item in result["differentials"]:
            self.assertIn("evidence_tier", item)
            self.assertIn(item["evidence_tier"], [1, 2, 3, 4, 5])

    def test_differential_has_ruling_in_out(self):
        p = _make_pipeline()
        result = run(p.differential(symptoms=["fever", "body aches", "fatigue"]))
        for item in result["differentials"]:
            self.assertIn("ruling_in_symptoms", item)
            self.assertIn("ruling_out_symptoms", item)

    def test_differential_query_metadata_has_entropy(self):
        p = _make_pipeline()
        result = run(p.differential(symptoms=["headache", "nausea"]))
        meta = result["query_metadata"]
        self.assertIn("retrieval_entropy", meta)
        self.assertGreaterEqual(meta["retrieval_entropy"], 0.0)
        self.assertLessEqual(meta["retrieval_entropy"], 1.0)

    def test_differential_ranked_by_score(self):
        p = _make_pipeline()
        result = run(p.differential(symptoms=["fever", "cough", "fatigue"]))
        scores = [d["hybrid_score"] for d in result["differentials"]]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_differential_emergency_returns_empty_list(self):
        p = _make_pipeline()
        result = run(p.differential(symptoms=["chest pain"], query="I cannot breathe"))
        self.assertEqual(result["differentials"], [])


class PipelineSessionContextTests(unittest.TestCase):
    """Tests for session context injection in the answer method."""

    def test_answer_accepts_session_context_string(self):
        p = _make_pipeline(groq_response=None)
        result = run(p.answer("runny nose", session_context="[Turn 1: cough]"))
        # Should not raise and should return a valid response
        self.assertIn("possible_conditions", result)

    def test_session_store_creates_and_retrieves_session(self):
        store = SessionStore()
        sid = store.create_session()
        session = store.get_session(sid)
        self.assertIsNotNone(session)
        self.assertEqual(session.session_id, sid)

    def test_session_store_get_or_create_with_none(self):
        store = SessionStore()
        sid, session = store.get_or_create(None)
        self.assertIsNotNone(sid)
        self.assertTrue(len(sid) > 0)

    def test_session_store_records_turn(self):
        store = SessionStore()
        sid = store.create_session()
        store.record_turn(
            sid, "headache query", [],
            {"possible_conditions": ["Migraine"], "severity": "medium",
             "explanation": "headache explanation"}
        )
        session = store.get_session(sid)
        self.assertEqual(len(session.turns), 1)
        self.assertEqual(session.turns[0].conditions, ["Migraine"])

    def test_session_context_summary_non_empty_after_turn(self):
        store = SessionStore()
        sid = store.create_session()
        store.record_turn(
            sid, "headache", [],
            {"possible_conditions": ["Migraine"], "severity": "medium",
             "explanation": "Some explanation"}
        )
        session = store.get_session(sid)
        summary = session.context_summary()
        self.assertIn("Migraine", summary)

    def test_session_max_turns_enforced(self):
        from app.rag.session_store import SESSION_MAX_TURNS
        store = SessionStore()
        sid = store.create_session()
        for i in range(SESSION_MAX_TURNS + 3):
            store.record_turn(
                sid, f"query {i}", [],
                {"possible_conditions": ["Condition"], "severity": "low", "explanation": "x"}
            )
        session = store.get_session(sid)
        self.assertLessEqual(len(session.turns), SESSION_MAX_TURNS)


if __name__ == "__main__":
    unittest.main()
