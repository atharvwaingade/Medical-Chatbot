"""
Tests for the GRADE-Inspired Evidence Tier Grader.

Validates:
- WHO/CDC/NIH sources map to Tier 3 (Clinical Guideline).
- PubMed/NEJM sources map to Tier 4 (Peer-reviewed).
- Cochrane / systematic review sources map to Tier 1.
- Unknown sources default to Tier 5 (Expert consensus).
- Tier descriptions and confidence bonuses are consistent.
"""
import unittest

from app.rag.evidence_grader import (
    grade_source,
    tier_description,
    confidence_bonus,
    TIER_DESCRIPTIONS,
    TIER_CONFIDENCE_BONUS,
)


class EvidenceTierMappingTests(unittest.TestCase):
    def test_cochrane_maps_to_tier_1(self):
        self.assertEqual(grade_source("Cochrane Review — Upper Respiratory Infections"), 1)

    def test_systematic_review_maps_to_tier_1(self):
        self.assertEqual(grade_source("Systematic review of antiviral therapy"), 1)

    def test_who_maps_to_tier_3(self):
        self.assertEqual(grade_source("WHO - Respiratory Tract Infection Guidance"), 3)

    def test_cdc_maps_to_tier_3(self):
        self.assertEqual(grade_source("CDC - Influenza (Flu)"), 3)

    def test_nih_maps_to_tier_3(self):
        self.assertEqual(grade_source("NIH - National Institute of Mental Health"), 3)

    def test_aha_acc_maps_to_tier_3(self):
        self.assertEqual(grade_source("AHA/ACC - Acute Coronary Syndrome Guidelines"), 3)

    def test_pubmed_maps_to_tier_4(self):
        self.assertEqual(grade_source("PubMed - Migraine Pathophysiology Review"), 4)

    def test_nejm_maps_to_tier_4(self):
        self.assertEqual(grade_source("NEJM - COVID-19 Clinical Outcomes"), 4)

    def test_unknown_source_maps_to_tier_5(self):
        self.assertEqual(grade_source("Random blog post about headaches"), 5)

    def test_empty_source_maps_to_tier_5(self):
        self.assertEqual(grade_source(""), 5)

    def test_case_insensitive(self):
        self.assertEqual(grade_source("WHO guidelines"), grade_source("who guidelines"))


class TierDescriptionTests(unittest.TestCase):
    def test_tier_1_description(self):
        desc = tier_description(1)
        self.assertIn("Systematic", desc)

    def test_tier_3_description(self):
        desc = tier_description(3)
        self.assertIn("Guideline", desc)

    def test_tier_5_description(self):
        desc = tier_description(5)
        self.assertIn("consensus", desc.lower())

    def test_unknown_tier_returns_fallback_string(self):
        desc = tier_description(99)
        self.assertIsInstance(desc, str)
        self.assertTrue(len(desc) > 0)


class ConfidenceBonusTests(unittest.TestCase):
    def test_tier_1_has_highest_bonus(self):
        self.assertGreater(confidence_bonus(1), confidence_bonus(3))

    def test_tier_3_bonus_greater_than_tier_5(self):
        self.assertGreater(confidence_bonus(3), confidence_bonus(5))

    def test_tier_5_bonus_is_zero(self):
        self.assertEqual(confidence_bonus(5), 0.0)

    def test_all_bonuses_non_negative(self):
        for tier in TIER_CONFIDENCE_BONUS:
            self.assertGreaterEqual(TIER_CONFIDENCE_BONUS[tier], 0.0)

    def test_all_tiers_have_description(self):
        for tier in [1, 2, 3, 4, 5]:
            self.assertIn(tier, TIER_DESCRIPTIONS)


if __name__ == "__main__":
    unittest.main()
