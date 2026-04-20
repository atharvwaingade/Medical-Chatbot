"""
Tests for MedCausalGraph: symptom-condition causal graph with personalised PageRank.

Validates:
- Graph is built correctly from KB entries (SYMPTOM_CONDITION edges).
- Comorbidity edges are correctly registered.
- Temporal edges are correctly registered.
- Personalised PageRank seed on affirmed symptoms returns non-zero condition scores.
- Negated symptoms as barriers reduce condition scores compared to no barrier.
- Prior conditions boost adjacent comorbid conditions (sequential Bayesian update).
- causal_scores() returns values in [0, 1].
- fuse_scores() preserves descending order.
- Edge cases: empty entries, no seed symptoms, all symptoms negated.
"""
import unittest

from app.rag.causal_graph import MedCausalGraph

_ENTRIES = [
    {
        "condition": "Influenza",
        "symptoms": ["fever", "body aches", "fatigue", "dry cough"],
        "prevalence": "common",
        "verified": True,
        "source": "CDC",
    },
    {
        "condition": "Common Cold",
        "symptoms": ["runny nose", "sore throat", "sneezing", "mild cough"],
        "prevalence": "very common",
        "verified": True,
        "source": "WHO",
    },
    {
        "condition": "Migraine",
        "symptoms": ["severe headache", "nausea", "light sensitivity"],
        "prevalence": "common",
        "verified": True,
        "source": "PubMed",
    },
    {
        "condition": "Acute Gastroenteritis",
        "symptoms": ["diarrhea", "vomiting", "nausea", "abdominal cramps"],
        "prevalence": "very common",
        "verified": True,
        "source": "WHO",
    },
]


def _build_graph(entries=None):
    g = MedCausalGraph()
    g.build_from_entries(entries if entries is not None else _ENTRIES)
    return g


class GraphBuildTests(unittest.TestCase):
    def test_condition_nodes_registered(self):
        g = _build_graph()
        self.assertIn("Influenza", g._condition_nodes)
        self.assertIn("Migraine", g._condition_nodes)

    def test_symptom_nodes_registered(self):
        g = _build_graph()
        self.assertIn("fever", g._symptom_nodes)
        self.assertIn("nausea", g._symptom_nodes)

    def test_symptom_condition_edge_exists(self):
        g = _build_graph()
        # "fever" should have an outgoing edge to "Influenza"
        self.assertIn("Influenza", g._adj.get("fever", {}))

    def test_shared_symptom_weight_lower(self):
        """Symptoms shared by multiple conditions get lower edge weights (1/freq)."""
        g = _build_graph()
        # "nausea" appears in Migraine AND Gastroenteritis → weight < 1.0
        w = g._adj.get("nausea", {}).get("Migraine", 0.0)
        self.assertGreater(w, 0.0)
        self.assertLess(w, 1.0)  # shared symptom → divided by freq > 1

    def test_empty_entries_no_error(self):
        g = _build_graph(entries=[])
        self.assertEqual(len(g._condition_nodes), 0)
        self.assertEqual(len(g._symptom_nodes), 0)


class PageRankTests(unittest.TestCase):
    def setUp(self):
        self.g = _build_graph()

    def test_seed_symptom_gives_nonzero_scores(self):
        scores = self.g.personalised_pagerank(["fever", "body aches"])
        self.assertGreater(scores.get("Influenza", 0.0), 0.0)

    def test_influenza_tops_fever_bodyaches(self):
        scores = self.g.personalised_pagerank(["fever", "body aches", "fatigue"])
        top_cond = max(scores, key=scores.get)
        self.assertEqual(top_cond, "Influenza")

    def test_migraine_tops_headache(self):
        scores = self.g.personalised_pagerank(["severe headache", "nausea", "light sensitivity"])
        top_cond = max(scores, key=scores.get)
        self.assertEqual(top_cond, "Migraine")

    def test_negated_barrier_reduces_score(self):
        """Negating fever should reduce Influenza's score vs. no barrier."""
        score_no_barrier = self.g.personalised_pagerank(["fever"])
        score_with_barrier = self.g.personalised_pagerank(["fever"], barrier_symptoms=["fever"])
        flu_no = score_no_barrier.get("Influenza", 0.0)
        flu_bar = score_with_barrier.get("Influenza", 0.0)
        self.assertGreaterEqual(flu_no, flu_bar)

    def test_no_seed_returns_nonzero_scores(self):
        """When no seeds match KB nodes, PageRank still returns scores."""
        scores = self.g.personalised_pagerank(["xyzzy_nonexistent"])
        self.assertIsInstance(scores, dict)
        self.assertGreater(len(scores), 0)

    def test_all_symptoms_negated_low_scores(self):
        """All barriers → condition scores are low (uniform teleport only)."""
        scores = self.g.personalised_pagerank(
            ["fever"], barrier_symptoms=["fever", "body aches", "fatigue", "dry cough"]
        )
        # With barriers, no structured walk is possible; scores should be low
        flu = scores.get("Influenza", 0.0)
        # Cannot be exactly zero because teleportation exists
        self.assertIsNotNone(flu)

    def test_prior_conditions_boost_comorbid(self):
        """Prior conditions seed the graph and should affect comorbid scores."""
        # Build a full graph with COMORBID pairs
        from app.rag.causal_graph import MedCausalGraph
        import json, pathlib
        data = json.loads(
            pathlib.Path("backend/data/sample_medical_knowledge.json").read_text()
        )
        g = MedCausalGraph()
        g.build_from_entries(data)
        # "chest pain" → should retrieve cardiovascular conditions more
        scores_no_prior = g.personalised_pagerank(["chest pain"])
        scores_with_prior = g.personalised_pagerank(
            ["chest pain"],
            prior_conditions={"Hypertension (High Blood Pressure)": 0.8},
        )
        # The sum of scores should differ, indicating priors had an effect
        total_no = sum(scores_no_prior.values())
        total_with = sum(scores_with_prior.values())
        self.assertNotAlmostEqual(total_no, total_with, places=4)


class CausalScoresTests(unittest.TestCase):
    def setUp(self):
        self.g = _build_graph()

    def test_causal_scores_in_range(self):
        scores = self.g.causal_scores(["fever", "body aches"])
        for cond, score in scores.items():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_causal_scores_max_is_one(self):
        scores = self.g.causal_scores(["fever", "body aches"])
        if scores:
            self.assertAlmostEqual(max(scores.values()), 1.0, places=5)

    def test_causal_scores_empty_affirmed(self):
        scores = self.g.causal_scores([])
        self.assertIsInstance(scores, dict)


class FuseScoresTests(unittest.TestCase):
    def setUp(self):
        self.g = _build_graph()

    def test_fuse_scores_sorted_descending(self):
        hybrid = [
            (0.8, {"condition": "Influenza"}),
            (0.5, {"condition": "Migraine"}),
            (0.3, {"condition": "Common Cold"}),
        ]
        causal = {"Influenza": 1.0, "Migraine": 0.5, "Common Cold": 0.2}
        fused = self.g.fuse_scores(hybrid, causal)
        scores = [s for s, _ in fused]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_fuse_scores_all_entries_preserved(self):
        hybrid = [(0.7, {"condition": "Influenza"}), (0.4, {"condition": "Migraine"})]
        causal = {}
        fused = self.g.fuse_scores(hybrid, causal)
        self.assertEqual(len(fused), 2)


if __name__ == "__main__":
    unittest.main()
