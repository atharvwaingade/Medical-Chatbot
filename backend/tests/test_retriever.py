"""
Tests for the BM25 Retriever.

Validates that:
- The most symptom-relevant entry ranks highest.
- IDF correctly down-weights ubiquitous terms.
- Edge cases (empty corpus, empty query) are handled gracefully.
- retrieve_scored returns (score, entry) tuples in descending order.
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
        """
        'headache' appears only in Migraine; 'nausea' appears in both Migraine
        and Gastroenteritis.  A query of only 'headache' should rank Migraine
        first because the IDF of 'headache' is higher than that of 'nausea'.
        """
        r = Retriever(_ENTRIES)
        results = r.retrieve("headache", top_k=1)
        self.assertEqual(results[0]["condition"], "Migraine")


if __name__ == "__main__":
    unittest.main()
