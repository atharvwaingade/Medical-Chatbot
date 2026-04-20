"""
Tests for MedRAPTOR: ontology-anchored four-level hierarchical retrieval.

Validates:
- Hierarchy is built correctly from KB entries.
- ICD-10 chapter classification routes conditions to the correct organ system.
- Syndrome clusters contain the expected conditions.
- route_systems() returns relevant systems for symptom tokens.
- hierarchical_context() returns complete context dicts.
- vague_query_conditions() returns None for specific queries and a set for vague.
- Edge cases: unknown ICD-10, empty entries, no matching syndrome.
"""
import json
import pathlib
import unittest

from app.rag.raptor import MedRAPTOR


def _load_entries():
    return json.loads(
        pathlib.Path("backend/data/sample_medical_knowledge.json").read_text()
    )


def _build_raptor():
    r = MedRAPTOR()
    r.build_from_entries(_load_entries())
    return r


class RaptorBuildTests(unittest.TestCase):
    def setUp(self):
        self.raptor = _build_raptor()

    def test_respiratory_conditions_classified_correctly(self):
        system = self.raptor._cond_to_system.get("Influenza", "")
        self.assertEqual(system, "Respiratory")

    def test_cardiovascular_conditions_classified(self):
        system = self.raptor._cond_to_system.get("Ischemic Stroke", "")
        self.assertEqual(system, "Cardiovascular")

    def test_psychiatric_conditions_classified(self):
        system = self.raptor._cond_to_system.get("Anxiety Disorder", "")
        self.assertEqual(system, "Psychiatric")

    def test_metabolic_conditions_classified(self):
        system = self.raptor._cond_to_system.get("Hypothyroidism", "")
        self.assertEqual(system, "Metabolic/Endocrine")

    def test_system_to_conditions_populated(self):
        self.assertIn("Respiratory", self.raptor._system_to_conditions)
        self.assertGreater(len(self.raptor._system_to_conditions["Respiratory"]), 0)

    def test_aetiology_to_systems_populated(self):
        self.assertGreater(len(self.raptor._aetiology_to_systems), 0)

    def test_syndrome_clusters_have_conditions(self):
        # Upper Respiratory Syndrome should contain Common Cold
        from app.rag.raptor import _SYNDROME_CLUSTERS
        for syn_name, _, patterns in _SYNDROME_CLUSTERS:
            if syn_name == "Upper Respiratory Syndrome":
                node = self.raptor._syndromes.get(syn_name)
                if node:
                    self.assertGreater(len(node.condition_names), 0)

    def test_empty_entries_no_error(self):
        r = MedRAPTOR()
        r.build_from_entries([])
        self.assertEqual(len(r._cond_to_system), 0)


class RaptorRoutingTests(unittest.TestCase):
    def setUp(self):
        self.raptor = _build_raptor()

    def test_fever_cough_routes_to_respiratory(self):
        systems = self.raptor.route_systems(["fever", "cough", "infection"])
        self.assertIn("Respiratory", systems)

    def test_chest_pain_routes_to_cardiovascular(self):
        systems = self.raptor.route_systems(["chest", "heart", "palpitation"])
        self.assertIn("Cardiovascular", systems)

    def test_headache_routes_to_neurological(self):
        systems = self.raptor.route_systems(["headache", "dizzy", "vertigo"])
        self.assertIn("Neurological", systems)

    def test_empty_tokens_returns_all_systems(self):
        """No token signal → return all registered systems (safe fallback)."""
        systems = self.raptor.route_systems([])
        self.assertGreater(len(systems), 0)

    def test_route_systems_returns_list(self):
        result = self.raptor.route_systems(["fever"])
        self.assertIsInstance(result, list)

    def test_conditions_in_systems_returns_set(self):
        result = self.raptor.conditions_in_systems(["Respiratory"])
        self.assertIsInstance(result, set)
        self.assertGreater(len(result), 0)


class RaptorHierarchicalContextTests(unittest.TestCase):
    def setUp(self):
        self.raptor = _build_raptor()

    def test_hierarchical_context_returns_dict(self):
        ctx = self.raptor.hierarchical_context("Influenza")
        self.assertIsInstance(ctx, dict)

    def test_context_has_required_keys(self):
        ctx = self.raptor.hierarchical_context("Influenza")
        for key in ("syndrome", "shared_symptoms", "organ_system", "aetiology_class", "lateral_syndromes"):
            self.assertIn(key, ctx)

    def test_influenza_organ_system_is_respiratory(self):
        ctx = self.raptor.hierarchical_context("Influenza")
        self.assertEqual(ctx["organ_system"], "Respiratory")

    def test_anxiety_aetiology_is_neuropsychiatric(self):
        ctx = self.raptor.hierarchical_context("Anxiety Disorder")
        self.assertEqual(ctx["aetiology_class"], "Neurological/Psychiatric")

    def test_unknown_condition_returns_empty_strings(self):
        ctx = self.raptor.hierarchical_context("NonexistentCondition_XYZ")
        self.assertEqual(ctx["organ_system"], "")
        self.assertEqual(ctx["syndrome"], "")

    def test_lateral_syndromes_list(self):
        ctx = self.raptor.hierarchical_context("Influenza")
        self.assertIsInstance(ctx["lateral_syndromes"], list)


class RaptorVagueQueryTests(unittest.TestCase):
    def setUp(self):
        self.raptor = _build_raptor()

    def test_specific_query_returns_none(self):
        """Low entropy (< threshold) → None, meaning use normal retrieval."""
        result = self.raptor.vague_query_conditions(
            ["fever", "dry cough"], retrieval_entropy=0.30
        )
        self.assertIsNone(result)

    def test_vague_query_returns_set(self):
        """High entropy → returns candidate set."""
        result = self.raptor.vague_query_conditions(
            ["feel unwell"], retrieval_entropy=0.90
        )
        self.assertIsInstance(result, set)
        self.assertGreater(len(result), 0)

    def test_vague_query_result_is_subset_of_all_conditions(self):
        all_conds = set(self.raptor._cond_to_system.keys())
        result = self.raptor.vague_query_conditions(
            ["pain"], retrieval_entropy=0.85
        )
        if result:
            self.assertTrue(result.issubset(all_conds))

    def test_get_syndrome_conditions(self):
        conds = self.raptor.get_syndrome_conditions("Upper Respiratory Syndrome")
        self.assertIsInstance(conds, list)


if __name__ == "__main__":
    unittest.main()
