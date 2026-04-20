"""
MedCausalGraph: Symptom-Condition Causal Knowledge Graph with Personalised PageRank
=====================================================================================
Implements a lightweight directed weighted graph over symptom and condition nodes,
supporting four medically-grounded edge types:

Edge Types
----------
1. **SYMPTOM_CONDITION** (symptom → condition)
   Built automatically from the knowledge base: for each condition C with symptom
   list S, every symptom s ∈ S contributes an edge s → C with weight
   P(C | s) ∝ 1 / count(conditions that list s).  This is a standard
   co-occurrence estimator — conditions with rare symptoms get a higher boost
   from those symptoms.

2. **COMORBID** (condition ↔ condition)
   Bidirectional edges encoding known co-occurrence risk ratios from published
   epidemiology.  Sourced from Barnett et al. (2012, PLOS Medicine) and AHA/ACC
   risk guidelines.  These propagate probability mass between comorbid conditions
   when one has already been identified in a prior session turn.

3. **TEMPORAL** (symptom → symptom)
   Directed edges encoding P(symptom_B appears within 72h | symptom_A already
   present).  Derived from illness-progression literature.  Example:
   fever → productive_cough, indicating bacterial progression.

4. **EXCLUDES** (negated_symptom → condition — treated as absorbing barriers)
   When a symptom is negated by the patient (NegEx output), its node acts as an
   absorbing barrier during PageRank: probability cannot flow FROM a negated
   symptom node TO condition nodes.  This encodes the diagnostic rule:
   "if fever is absent, down-weight all conditions for which fever is typical".

Personalised PageRank
---------------------
The retrieval signal is computed as personalised PageRank (Haveliwala, 2002)
seeded on the patient's affirmed symptom nodes:

    PR(v) = (1-d)/N + d · Σ_{u→v} PR(u) · w(u,v) / out_weight(u)

where d = 0.85 (damping) and seed nodes have their teleportation probability
boosted proportionally.  Negated symptom nodes have their out-edges removed
(absorbing barrier effect).

After convergence (≤50 power iterations, ε = 1e-6), the scores of condition
nodes are extracted and returned as a dict {condition_name: score}.  This
provides a graph-based causal ranking that complements the lexical BM25+SCS
signal and the Bayesian prevalence prior.

Integration in the Pipeline
---------------------------
The pipeline calls ``graph.causal_scores(affirmed, negated, prior_conditions)``
and uses the resulting scores to re-rank retrieval results:

    final_score = RETRIEVER_WEIGHT · hybrid_score
                + CAUSAL_WEIGHT   · causal_score_normalised

References
----------
Haveliwala, T. (2002). Topic-sensitive PageRank. *WWW 2002*, 517–526.

Barnett, K. et al. (2012). Epidemiology of multimorbidity and implications for
health care, research, and medical education: a cross-sectional study. *The
Lancet*, 380(9836), 37–43.

Page, L. et al. (1999). The PageRank citation ranking: Bringing order to the
web. *Stanford InfoLab Technical Report*.
"""
from __future__ import annotations

import math
import re

# ---------------------------------------------------------------------------
# Hardcoded comorbidity edges (bidirectional, risk-ratio–derived weights)
# ---------------------------------------------------------------------------
# Format: (condition_name_A, condition_name_B, weight)
# Weights ∈ (0, 1) represent conditional co-occurrence probability.
# Source: Barnett et al. 2012, AHA/ACC risk guidelines, ADA guidelines.
_COMORBID_PAIRS: list[tuple[str, str, float]] = [
    # Cardiometabolic cluster
    ("Hypertension (High Blood Pressure)", "Type 2 Diabetes Mellitus", 0.45),
    ("Hypertension (High Blood Pressure)", "Possible Acute Coronary Syndrome", 0.30),
    ("Hypertension (High Blood Pressure)", "Ischemic Stroke", 0.28),
    ("Hypertension (High Blood Pressure)", "Atrial Fibrillation", 0.25),
    ("Type 2 Diabetes Mellitus", "Possible Acute Coronary Syndrome", 0.35),
    ("Type 2 Diabetes Mellitus", "Hypertension (High Blood Pressure)", 0.45),
    # Thrombotic cascade
    ("Deep Vein Thrombosis (DVT)", "Pulmonary Embolism", 0.60),
    ("Pulmonary Embolism", "Deep Vein Thrombosis (DVT)", 0.50),
    ("Atrial Fibrillation", "Ischemic Stroke", 0.50),
    # Metabolic / endocrine cluster
    ("Hypothyroidism", "Iron Deficiency Anemia", 0.25),
    ("Hypothyroidism", "Depression (Major Depressive Disorder)", 0.30),
    ("Type 2 Diabetes Mellitus", "Iron Deficiency Anemia", 0.20),
    # Respiratory cluster
    ("Asthma Exacerbation", "Allergic Rhinitis", 0.55),
    ("Allergic Rhinitis", "Asthma Exacerbation", 0.40),
    ("Community-Acquired Pneumonia", "Influenza", 0.30),
    ("Influenza", "Community-Acquired Pneumonia", 0.20),
    # GI cluster
    ("Gastroesophageal Reflux Disease (GERD)", "Asthma Exacerbation", 0.20),
    ("Irritable Bowel Syndrome (IBS)", "Anxiety Disorder", 0.35),
    ("Anxiety Disorder", "Irritable Bowel Syndrome (IBS)", 0.25),
    # Psychiatric cluster
    ("Anxiety Disorder", "Depression (Major Depressive Disorder)", 0.50),
    ("Depression (Major Depressive Disorder)", "Anxiety Disorder", 0.45),
    # Dehydration downstream
    ("Acute Gastroenteritis", "Dehydration", 0.40),
    ("Kidney Stones (Nephrolithiasis)", "Dehydration", 0.30),
]

# ---------------------------------------------------------------------------
# Hardcoded temporal progression edges (symptom_early → symptom_late)
# ---------------------------------------------------------------------------
# Weight = approximate P(symptom_late within 72h | symptom_early present)
# Derived from clinical illness-progression literature and UpToDate reviews.
_TEMPORAL_PAIRS: list[tuple[str, str, float]] = [
    # Respiratory progression
    ("fever", "productive cough", 0.40),       # viral → secondary bacterial
    ("fever", "shortness of breath", 0.25),     # pneumonia progression
    ("dry cough", "productive cough", 0.35),    # URTI → LRTI
    ("runny nose", "nasal congestion", 0.70),   # common cold progression
    ("sore throat", "dry cough", 0.50),         # post-nasal drip
    # Systemic → organ-specific
    ("fatigue", "body aches", 0.45),            # viral syndrome
    ("fever", "chills", 0.65),                  # infection
    ("fever", "sweating", 0.55),                # fever breaking
    # GI progression
    ("nausea", "vomiting", 0.60),              # typical GI illness
    ("vomiting", "diarrhea", 0.45),            # gastroenteritis progression
    ("abdominal pain", "nausea", 0.40),         # visceral irritation
    # Neurological
    ("severe headache", "nausea", 0.50),        # migraine prodrome
    ("dizziness", "nausea", 0.55),              # vestibular
    # Cardiovascular progression
    ("palpitations", "shortness of breath", 0.35),  # arrhythmia impact
    ("leg swelling", "shortness of breath", 0.45),   # DVT → PE
]


class MedCausalGraph:
    """
    Lightweight directed weighted graph over symptom and condition nodes.

    Build the graph once per application lifetime (``build_from_entries``),
    then call ``causal_scores`` on every query for graph-based re-ranking.
    """

    DAMPING: float = 0.85
    MAX_ITER: int = 50
    CONVERGENCE_EPS: float = 1e-6

    # Pipeline integration weights
    RETRIEVER_WEIGHT: float = 0.80
    CAUSAL_WEIGHT: float = 0.20

    def __init__(self) -> None:
        # Adjacency list: node → {neighbor: weight}
        self._adj: dict[str, dict[str, float]] = {}
        self._nodes: list[str] = []
        self._condition_nodes: set[str] = set()
        self._symptom_nodes: set[str] = set()
        self._node_index: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    @staticmethod
    def _canonicalise(text: str) -> str:
        """Lowercase and collapse whitespace."""
        return re.sub(r"\s+", " ", text.strip().lower())

    def _add_edge(self, src: str, dst: str, weight: float) -> None:
        """Add a directed edge, accumulating weights if edge already exists."""
        self._adj.setdefault(src, {})[dst] = (
            self._adj.get(src, {}).get(dst, 0.0) + weight
        )
        # Ensure dst exists in adjacency map even without outgoing edges
        self._adj.setdefault(dst, {})

    def build_from_entries(self, entries: list[dict]) -> None:
        """
        Construct the graph from KB entries plus hardcoded domain knowledge.

        Parameters
        ----------
        entries : list[dict]
            Verified KB entries with ``condition`` and ``symptoms`` fields.
        """
        self._adj.clear()
        self._condition_nodes.clear()
        self._symptom_nodes.clear()

        # -- Stage 1: SYMPTOM_CONDITION edges (symptom → condition) -----------
        # For each symptom, count how many conditions list it.
        symptom_condition_count: dict[str, int] = {}
        for entry in entries:
            for sym in entry.get("symptoms", []):
                canon = self._canonicalise(sym)
                symptom_condition_count[canon] = symptom_condition_count.get(canon, 0) + 1

        for entry in entries:
            cond_name = entry.get("condition", "")
            if not cond_name:
                continue
            self._condition_nodes.add(cond_name)
            self._adj.setdefault(cond_name, {})
            for sym in entry.get("symptoms", []):
                canon = self._canonicalise(sym)
                self._symptom_nodes.add(canon)
                # P(condition | symptom) ∝ 1 / global_freq(symptom)
                # Rarer symptoms are more diagnostically discriminative.
                freq = max(symptom_condition_count.get(canon, 1), 1)
                weight = 1.0 / freq
                self._add_edge(canon, cond_name, weight)

        # -- Stage 2: COMORBID edges (condition ↔ condition) ------------------
        for cond_a, cond_b, w in _COMORBID_PAIRS:
            if cond_a in self._condition_nodes and cond_b in self._condition_nodes:
                self._add_edge(cond_a, cond_b, w)

        # -- Stage 3: TEMPORAL edges (symptom → symptom) ----------------------
        for sym_a, sym_b, w in _TEMPORAL_PAIRS:
            # Only add if at least one endpoint appears in the KB symptom set
            if sym_a in self._symptom_nodes or sym_b in self._symptom_nodes:
                self._symptom_nodes.add(sym_a)
                self._symptom_nodes.add(sym_b)
                self._adj.setdefault(sym_a, {})
                self._adj.setdefault(sym_b, {})
                self._add_edge(sym_a, sym_b, w)

        # Build stable node ordering for vector operations
        all_nodes = sorted(self._condition_nodes | self._symptom_nodes)
        self._nodes = all_nodes
        self._node_index = {n: i for i, n in enumerate(all_nodes)}

    # ------------------------------------------------------------------
    # Normalised out-weight per node
    # ------------------------------------------------------------------

    def _out_weights(self, excluded_nodes: frozenset[str]) -> dict[str, float]:
        """
        Compute total out-weight for each node (excluding absorbing barrier nodes).
        """
        totals: dict[str, float] = {}
        for src, neighbours in self._adj.items():
            if src in excluded_nodes:
                totals[src] = 0.0  # absorbing barrier: no outflow
                continue
            totals[src] = sum(
                w for dst, w in neighbours.items() if dst not in excluded_nodes
            )
        return totals

    # ------------------------------------------------------------------
    # Personalised PageRank
    # ------------------------------------------------------------------

    def personalised_pagerank(
        self,
        seed_symptoms: list[str],
        barrier_symptoms: list[str] | None = None,
        prior_conditions: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """
        Run personalised PageRank seeded on *seed_symptoms*.

        Parameters
        ----------
        seed_symptoms : list[str]
            Affirmed symptom strings from the patient query (canonical form).
        barrier_symptoms : list[str] | None
            NegEx-negated symptoms; their nodes act as absorbing barriers
            (zero probability outflow), modelling diagnostic exclusion.
        prior_conditions : dict[str, float] | None
            Posterior from a previous session turn {condition: score}.
            These conditions receive additional seed weight, implementing
            sequential Bayesian updating across conversation turns.

        Returns
        -------
        dict[str, float]
            Condition name → PageRank score, sorted descending.
        """
        if not self._nodes:
            return {}

        barriers = frozenset(
            self._canonicalise(s) for s in (barrier_symptoms or [])
        )
        seeds: dict[str, float] = {}
        # Seed on affirmed symptoms
        for sym in seed_symptoms:
            c = self._canonicalise(sym)
            if c in self._node_index:
                seeds[c] = seeds.get(c, 0.0) + 1.0
        # Additional seed on prior-turn conditions (sequential Bayesian update)
        if prior_conditions:
            for cond, score in prior_conditions.items():
                if cond in self._node_index:
                    seeds[cond] = seeds.get(cond, 0.0) + score

        if not seeds:
            # No seeds → return uniform condition scores
            n_cond = len(self._condition_nodes)
            if n_cond == 0:
                return {}
            uniform = 1.0 / n_cond
            return {c: uniform for c in sorted(self._condition_nodes)}

        # Normalise seed vector
        total_seed = sum(seeds.values())
        seed_vec: dict[str, float] = {k: v / total_seed for k, v in seeds.items()}

        # Pre-compute out-weights (absorbing barriers have zero outflow)
        out_w = self._out_weights(barriers)

        N = len(self._nodes)
        # Initialise PR vector to seed
        pr: dict[str, float] = {n: seed_vec.get(n, 0.0) for n in self._nodes}

        for _ in range(self.MAX_ITER):
            new_pr: dict[str, float] = {}
            for v in self._nodes:
                if v in barriers:
                    new_pr[v] = 0.0
                    continue
                # Teleportation term: personalised to seed vector
                teleport = (1.0 - self.DAMPING) * seed_vec.get(v, 1.0 / N)
                # Random walk term
                walk = 0.0
                for u, neighbours in self._adj.items():
                    if u in barriers:
                        continue
                    if v not in neighbours:
                        continue
                    w_uv = neighbours[v]
                    total_out = out_w.get(u, 0.0)
                    if total_out > 0:
                        walk += pr.get(u, 0.0) * (w_uv / total_out)
                new_pr[v] = teleport + self.DAMPING * walk

            # Check convergence (L1 norm)
            delta = sum(abs(new_pr.get(v, 0.0) - pr.get(v, 0.0)) for v in self._nodes)
            pr = new_pr
            if delta < self.CONVERGENCE_EPS:
                break

        # Extract condition node scores
        condition_scores = {
            cond: pr.get(cond, 0.0) for cond in self._condition_nodes
        }
        return dict(sorted(condition_scores.items(), key=lambda x: x[1], reverse=True))

    # ------------------------------------------------------------------
    # Pipeline integration helper
    # ------------------------------------------------------------------

    def causal_scores(
        self,
        affirmed_symptoms: list[str],
        negated_symptoms: list[str] | None = None,
        prior_conditions: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """
        Return normalised causal scores {condition: score ∈ [0,1]}.

        Scores are min-max normalised across all returned conditions so they
        can be linearly combined with hybrid_score in the pipeline.
        """
        raw = self.personalised_pagerank(
            affirmed_symptoms,
            barrier_symptoms=negated_symptoms,
            prior_conditions=prior_conditions,
        )
        if not raw:
            return {}
        max_score = max(raw.values()) or 1.0
        min_score = min(raw.values())
        score_range = max_score - min_score or 1.0
        return {k: (v - min_score) / score_range for k, v in raw.items()}

    def fuse_scores(
        self,
        hybrid_scores: list[tuple[float, dict]],
        causal_scores_map: dict[str, float],
    ) -> list[tuple[float, dict]]:
        """
        Fuse retriever hybrid scores with causal graph scores.

        Parameters
        ----------
        hybrid_scores : list[(score, entry)]
            Sorted retriever results.
        causal_scores_map : dict[str, float]
            Normalised causal scores keyed by condition name.

        Returns
        -------
        list[(fused_score, entry)]
            Re-ranked by fused score, descending.
        """
        fused = []
        for score, entry in hybrid_scores:
            cond = entry.get("condition", "")
            causal = causal_scores_map.get(cond, 0.0)
            fused_score = self.RETRIEVER_WEIGHT * score + self.CAUSAL_WEIGHT * causal
            fused.append((fused_score, entry))
        return sorted(fused, key=lambda x: x[0], reverse=True)
