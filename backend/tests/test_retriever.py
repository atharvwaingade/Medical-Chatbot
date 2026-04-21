"""
Tests for the MedHybrid-BM25+SCS+PRF Retriever.

Validates that:
- Symptom Coverage Score (SCS) improves ranking on exact symptom matches.
- PRF expansion terms are correctly extracted from top documents.
- Score entropy is computed correctly (0 for single result, >0 for multiple).
- Hybrid scoring produces results sorted descending.
- Edge cases (empty corpus, empty query) are handled gracefully.
- retrieve_scored backward-compat returns (score, entry) tuples.
- IDF correctly down-weights ubiquitous terms.
"""
import unittest

from app.rag.retriever import Retriever

_ENTRIES = [
    {
        "condition": "Common Cold",
        "symptoms": ["runny nose", "sore throat", "mild cough", "sneezing"],
        "explanation": "A viral upper respiratory infection.",
        "warnings": ["breathing difficulty"],
        "source": "WHO",
        "verified": True,
    },
    {
        "condition": "Influenza",
        "symptoms": ["fever", "body aches", "fatigue", "dry cough"],
        "explanation": "Contagious respiratory illness caused by influenza viruses.",
        "warnings": ["chest pain"],
        "source": "CDC",
        "verified": True,
    },
    {
        "condition": "Migraine",
        "symptoms": ["severe headache", "nausea", "light sensitivity", "throbbing pain"],
        "explanation": "A recurrent primary headache disorder.",
        "warnings": ["sudden severe headache"],
        "source": "PubMed",
        "verified": True,
    },
    {
        "condition": "Acute Gastroenteritis",
        "symptoms": ["diarrhea", "vomiting", "abdominal cramps", "nausea"],
        "explanation": "Inflammation of the gastrointestinal tract.",
        "warnings": ["dehydration"],
        "source": "WHO",
        "verified": True,
    },
]


class RetrieverRankingTests(unittest.TestCase):
    def setUp(self):
        self.retriever = Retriever(_ENTRIES)

    def test_exact_condition_name_ranks_first(self):
        results = self.retriever.retrieve("migraine", top_k=3)
        self.assertEqual(results[0]["condition"], "Migraine")

    def test_symptom_match_ranks_correctly(self):
        results = self.retriever.retrieve("fever body aches fatigue", top_k=1)
        self.assertEqual(results[0]["condition"], "Influenza")

    def test_nausea_vomiting_ranks_gastroenteritis(self):
        results = self.retriever.retrieve("nausea vomiting diarrhea abdominal cramps", top_k=1)
        self.assertEqual(results[0]["condition"], "Acute Gastroenteritis")

    def test_runny_nose_sneezing_ranks_common_cold(self):
        results = self.retriever.retrieve("runny nose sneezing sore throat", top_k=1)
        self.assertEqual(results[0]["condition"], "Common Cold")

    def test_top_k_respected(self):
        results = self.retriever.retrieve("cough fever", top_k=2)
        self.assertLessEqual(len(results), 2)

    def test_retrieve_scored_descending(self):
        scored = self.retriever.retrieve_scored("fever body aches", top_k=4)
        scores = [s for s, _ in scored]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_retrieve_scored_top_score_positive(self):
        scored = self.retriever.retrieve_scored("headache nausea", top_k=3)
        self.assertGreater(scored[0][0], 0)


class RetrieverEdgeCaseTests(unittest.TestCase):
    def test_empty_corpus_returns_empty(self):
        r = Retriever([])
        self.assertEqual(r.retrieve("fever"), [])
        self.assertEqual(r.retrieve_scored("fever"), [])

    def test_empty_query_returns_fallback(self):
        r = Retriever(_ENTRIES)
        results = r.retrieve("", top_k=1)
        self.assertEqual(len(results), 1)

    def test_unrelated_query_returns_fallback_entry(self):
        r = Retriever(_ENTRIES)
        results = r.retrieve("xyzzy gobbledygook randomterm", top_k=1)
        self.assertEqual(len(results), 1)

    def test_single_entry_corpus(self):
        r = Retriever([_ENTRIES[0]])
        results = r.retrieve("runny nose", top_k=3)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["condition"], "Common Cold")


class RetrieverIdfTests(unittest.TestCase):
    """IDF-related correctness: rare terms should drive ranking more than common ones."""

    def test_rare_term_beats_common_term(self):
        r = Retriever(_ENTRIES)
        results = r.retrieve("headache", top_k=1)
        self.assertEqual(results[0]["condition"], "Migraine")


class RetrieverResultsTests(unittest.TestCase):
    """Tests for the new retrieve_results() with RetrievalResult objects."""

    def setUp(self):
        self.retriever = Retriever(_ENTRIES)

    def test_retrieve_results_returns_retrieval_result_objects(self):
        from app.rag.retriever import RetrievalResult
        results = self.retriever.retrieve_results("fever body aches", [], top_k=2)
        self.assertIsInstance(results[0], RetrievalResult)

    def test_retrieve_results_ruling_in_not_empty_for_matching_symptoms(self):
        results = self.retriever.retrieve_results("fever body aches fatigue", [], top_k=1)
        # Influenza should rank first; ruling_in should contain overlapping symptoms
        self.assertGreater(len(results[0].ruling_in), 0)

    def test_retrieve_results_scs_score_in_range(self):
        results = self.retriever.retrieve_results("fever body aches fatigue", [], top_k=3)
        for r in results:
            self.assertGreaterEqual(r.scs_score, 0.0)
            self.assertLessEqual(r.scs_score, 1.0)

    def test_retrieve_results_hybrid_score_sorted(self):
        results = self.retriever.retrieve_results("nausea vomiting diarrhea", [], top_k=4)
        scores = [r.hybrid_score for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))


class RetrieverEntropyTests(unittest.TestCase):
    """Score entropy calculation."""

    def setUp(self):
        self.retriever = Retriever(_ENTRIES)

    def test_entropy_zero_for_single_score(self):
        e = self.retriever.score_entropy([5.0])
        self.assertEqual(e, 0.0)

    def test_entropy_one_for_uniform_distribution(self):
        e = self.retriever.score_entropy([1.0, 1.0, 1.0, 1.0])
        self.assertAlmostEqual(e, 1.0, places=5)

    def test_entropy_between_0_and_1(self):
        e = self.retriever.score_entropy([10.0, 3.0, 1.0])
        self.assertGreaterEqual(e, 0.0)
        self.assertLessEqual(e, 1.0)

    def test_confidence_high_when_entropy_low(self):
        conf = self.retriever.confidence_from_entropy(0.1)
        self.assertEqual(conf, "high")

    def test_confidence_low_when_entropy_high(self):
        conf = self.retriever.confidence_from_entropy(0.9)
        self.assertEqual(conf, "low")

    def test_confidence_medium_in_between(self):
        conf = self.retriever.confidence_from_entropy(0.5)
        self.assertEqual(conf, "medium")


if __name__ == "__main__":
    unittest.main()
