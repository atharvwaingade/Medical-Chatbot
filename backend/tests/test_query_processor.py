"""
Tests for the Medical Query Processor.

Validates:
- Synonym normalization resolves layperson terms to canonical forms.
- NegEx negation detection correctly identifies negated symptoms.
- Medical stop words are filtered out.
- Multi-word phrase matching works correctly.
- ProcessedQuery contains correct fields.
"""
import unittest

from app.rag.query_processor import QueryProcessor, ProcessedQuery


class SynonymNormalizationTests(unittest.TestCase):
    def test_throwing_up_maps_to_vomiting(self):
        pq = QueryProcessor.process("I have been throwing up all day")
        self.assertIn("vomiting", pq.normalized_terms)

    def test_tummy_ache_maps_to_abdominal_pain(self):
        pq = QueryProcessor.process("tummy ache for two days")
        self.assertIn("abdominal pain", pq.normalized_terms)

    def test_fever_identity_mapping(self):
        pq = QueryProcessor.process("fever and chills")
        self.assertIn("fever", pq.normalized_terms)
        self.assertIn("chills", pq.normalized_terms)

    def test_cant_breathe_normalized(self):
        pq = QueryProcessor.process("I cant breathe properly")
        self.assertIn("difficulty breathing", pq.normalized_terms)

    def test_stuffy_nose_maps_to_nasal_congestion(self):
        pq = QueryProcessor.process("stuffy nose and sneezing")
        self.assertIn("nasal congestion", pq.normalized_terms)

    def test_high_temperature_maps_to_fever(self):
        pq = QueryProcessor.process("high temperature since yesterday")
        self.assertIn("fever", pq.normalized_terms)

    def test_unknown_term_not_in_normalized(self):
        pq = QueryProcessor.process("xyzzy gobbledygook")
        # No synonym for "xyzzy" — normalized_terms might be empty
        self.assertNotIn("xyzzy", pq.normalized_terms)

    def test_symptom_list_also_normalized(self):
        pq = QueryProcessor.process("question", symptoms=["throwing up", "dizzy"])
        self.assertIn("vomiting", pq.normalized_terms)
        self.assertIn("dizziness", pq.normalized_terms)


class NegationDetectionTests(unittest.TestCase):
    def test_no_fever_marks_fever_as_negated(self):
        pq = QueryProcessor.process("no fever present")
        self.assertTrue(
            any("fever" in t for t in pq.negated_terms),
            msg=f"Expected 'fever' in negated_terms, got: {pq.negated_terms}",
        )

    def test_without_marks_next_tokens_negated(self):
        pq = QueryProcessor.process("headache without nausea")
        # "nausea" should be negated
        self.assertTrue(
            any("nausea" in t for t in pq.negated_terms),
            msg=f"Expected 'nausea' negated, got: {pq.negated_terms}",
        )

    def test_affirmed_term_not_in_negated(self):
        pq = QueryProcessor.process("fever and cough but no headache")
        # "fever" and "cough" should NOT be in negated_terms
        self.assertFalse(
            any("fever" in t for t in pq.negated_terms),
            msg=f"'fever' should not be negated",
        )

    def test_negated_term_excluded_from_expanded_query(self):
        pq = QueryProcessor.process("no nausea")
        # "nausea" should not appear as an affirmed medical token
        # (it might still appear in normalized_terms since normalization
        #  happens before negation detection on the expanded text)
        self.assertNotIn("nausea", pq.medical_tokens)

    def test_positive_query_has_no_negated_terms(self):
        pq = QueryProcessor.process("fever cough fatigue body aches")
        self.assertEqual(pq.negated_terms, [])


class StopWordFilterTests(unittest.TestCase):
    def test_stopwords_excluded_from_medical_tokens(self):
        pq = QueryProcessor.process("I have a fever and cough")
        for token in pq.medical_tokens:
            self.assertNotIn(token, {"i", "have", "a", "and"})

    def test_short_tokens_excluded(self):
        pq = QueryProcessor.process("It is a bad cough")
        for token in pq.medical_tokens:
            self.assertGreater(len(token), 2, f"Short token found: {token!r}")


class ProcessedQueryFieldTests(unittest.TestCase):
    def test_original_preserved(self):
        q = "I have fever and cough"
        pq = QueryProcessor.process(q)
        self.assertEqual(pq.original, q)

    def test_expanded_query_non_empty_for_medical_query(self):
        pq = QueryProcessor.process("fever cough headache")
        self.assertTrue(len(pq.expanded_query) > 0)

    def test_medical_tokens_unique(self):
        pq = QueryProcessor.process("fever fever fever cough cough")
        self.assertEqual(len(pq.medical_tokens), len(set(pq.medical_tokens)))

    def test_process_returns_processed_query_type(self):
        pq = QueryProcessor.process("headache nausea")
        self.assertIsInstance(pq, ProcessedQuery)


if __name__ == "__main__":
    unittest.main()
