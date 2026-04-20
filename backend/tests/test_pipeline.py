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
    return asyncio.get_event_loop().run_until_complete(coro)


class PipelineReadyTests(unittest.TestCase):
    def test_ready_true_when_entries_loaded(self):
        p = _make_pipeline()
        self.assertTrue(p.ready)

    def test_entries_count(self):
        p = _make_pipeline()
        self.assertGreater(len(p.dataset.entries), 5)


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
        """
        If the LLM returns a condition not in the retrieved set it must be
        replaced with the top retrieved condition.
        """
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
        # Must be from retrieved set — "Ebola Fever" should be gone
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


if __name__ == "__main__":
    unittest.main()
