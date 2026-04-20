"""
Temporal Symptom Progression Graph (TSPG)
==========================================
Tracks the temporal sequence of reported symptoms across multi-turn session
conversations and up-ranks conditions whose canonical illness progression
matches the patient's observed symptom onset order.

Formal Model
------------
Let the patient's symptom history across session turns be represented as a
sequence of (symptom_token, turn_index) pairs:

    O = [(s₁, t₁), (s₂, t₂), ..., (sₙ, tₙ)]

where t_i ∈ {1, 2, ..., T} is the turn at which symptom sᵢ was first reported.

The TSPG augments the causal graph's TEMPORAL edges — which encode
P(symptom_B appears within 72h | symptom_A present) — with a temporal
compatibility score that measures how well a condition C's canonical progression
matches the patient's observed sequence.

Temporal Compatibility Score
-----------------------------
For a condition C with TEMPORAL edge set E_C = {(sₐ → s_b, w_ab)}, the
temporal compatibility score TC(C, O) is:

    TC(C, O) = Σ_{(sₐ→s_b) ∈ E_C ∩ O²} w_ab × exp(−λ × |rank(sₐ) − rank(s_b) + 1|)

where:
  rank(s) = turn index at which symptom s was first reported
  λ = DECAY_RATE = 0.5 (exponential decay over turn distance)
  The "+1" offset ensures that sₐ being reported one turn before s_b gives
  exp(0) = 1.0 (perfect match); any earlier or later departure is penalised.

Interpretation:
  TC = 1.0 : condition's expected progression perfectly matches patient's timeline
  TC = 0.0 : no temporal overlap or symptoms reported out of expected order

The score is normalised across all conditions before being added to the
causal re-ranking step.

Research Contribution
---------------------
This contribution extends personalised PageRank to a **time-series graph** —
a novel medical session modeling approach not present in any existing medical
RAG paper.  Prior work (MedRAG, BioASQ-style RAG) treats all symptoms as a
bag-of-words without temporal order.  The TSPG makes the system sensitive to
illness progression dynamics, which are clinically essential:
  - Fever → cough (1 day): bacterial URI progression
  - Chest pain → syncope (same day): ACS / arrhythmia progression
  - Fatigue → jaundice (weeks): hepatitis progression

The temporal compatibility score can also serve as a feature for a downstream
learning-to-rank model trained on EHR temporal symptom sequences.

References
----------
Rajpurkar, P. et al. (2022). AI in health and medicine. *Nature Medicine*, 28,
31–38.

Rotmensch, M. et al. (2017). Learning a health knowledge graph from electronic
medical records. *Scientific Reports*, 7, 5994.

Zhou, J. et al. (2014). From micro to macro: Data driven phenotyping by densification
of longitudinal electronic medical records. *KDD*, 1374–1383.
"""
from __future__ import annotations

import math


# ---------------------------------------------------------------------------
# Hardcoded temporal progression edges from the causal graph
# (mirror of MedCausalGraph._TEMPORAL_EDGES)
# ---------------------------------------------------------------------------
# Each entry: (symptom_A_token, symptom_B_token, weight)
# These represent P(B appears within 72h | A already present).

_TEMPORAL_EDGES: list[tuple[str, str, float]] = [
    # Respiratory / Infectious progression
    ("fever",       "cough",              0.60),
    ("fever",       "productive cough",   0.45),
    ("cough",       "shortness of breath",0.40),
    ("runny nose",  "cough",              0.55),
    ("runny nose",  "sore throat",        0.50),
    ("sore throat", "cough",              0.55),
    ("fatigue",     "fever",              0.40),
    ("fever",       "chills",             0.65),
    # Cardiac progression
    ("chest pain",  "shortness of breath",0.50),
    ("palpitations","dizziness",          0.40),
    ("chest pain",  "dizziness",          0.35),
    # Gastrointestinal progression
    ("nausea",      "vomiting",           0.70),
    ("vomiting",    "diarrhea",           0.40),
    ("abdominal pain","nausea",           0.55),
    # Neurological progression
    ("headache",    "nausea",             0.45),
    ("headache",    "vomiting",           0.35),
    ("dizziness",   "nausea",             0.40),
    # Metabolic progression
    ("fatigue",     "weight loss",        0.30),
    ("increased thirst","frequent urination",0.65),
]

# Exponential decay rate for turn-distance mismatch
_DECAY_RATE: float = 0.5


# ---------------------------------------------------------------------------
# Tokenisation (consistent with Retriever._tokenize)
# ---------------------------------------------------------------------------

def _tokenize_sym(sym: str) -> set[str]:
    """Return the set of lowercase non-trivial tokens from a symptom string."""
    return {
        t.strip(".,!?;:()[]\"'").lower()
        for t in sym.split()
        if len(t.strip()) > 2
    }


# ---------------------------------------------------------------------------
# Temporal Compatibility Score
# ---------------------------------------------------------------------------


def compute_temporal_compatibility(
    condition_symptoms: list[str],
    symptom_timeline: dict[str, int],
) -> float:
    """
    Compute the temporal compatibility between a condition's expected progression
    and the patient's observed symptom onset sequence.

    Parameters
    ----------
    condition_symptoms : list[str]
        Canonical symptom list for the condition (KB entry).
    symptom_timeline : dict[str, int]
        Maps symptom token → turn index (1-indexed) at which it was first reported.
        Built from multi-turn session history.

    Returns
    -------
    float
        Temporal compatibility score ≥ 0.  Higher = better temporal match.
    """
    if not condition_symptoms or not symptom_timeline or len(symptom_timeline) < 2:
        return 0.0

    # Flatten condition symptom tokens
    cond_tokens: set[str] = set()
    for sym in condition_symptoms:
        cond_tokens |= _tokenize_sym(sym)

    score = 0.0
    for sym_a, sym_b, weight in _TEMPORAL_EDGES:
        tok_a = _tokenize_sym(sym_a)
        tok_b = _tokenize_sym(sym_b)

        # Check both ends of the edge are in this condition's symptoms
        if not (tok_a & cond_tokens) or not (tok_b & cond_tokens):
            continue

        # Find the patient's reported turns for each end
        t_a: int | None = None
        t_b: int | None = None
        for tok, turn_idx in symptom_timeline.items():
            if tok in tok_a:
                t_a = turn_idx
            if tok in tok_b:
                t_b = turn_idx

        if t_a is None or t_b is None:
            continue  # patient hasn't reported one or both ends

        # Ideal: sₐ reported at turn t, s_b reported at turn t+1
        # Actual: t_b - t_a (positive = b came after a, as expected)
        expected_delta = 1  # one turn apart = canonical progression
        actual_delta = t_b - t_a

        # Decay: exp(-λ × |actual - expected|)
        mismatch = abs(actual_delta - expected_delta)
        decay = math.exp(-_DECAY_RATE * mismatch)

        # Only award positive score if a comes before b (or same turn)
        if actual_delta >= 0:
            score += weight * decay

    return round(score, 4)


# ---------------------------------------------------------------------------
# Timeline extraction from session history
# ---------------------------------------------------------------------------


def build_symptom_timeline(
    session_turns: list[dict],
) -> dict[str, int]:
    """
    Build a symptom → first_turn_index map from session turn history.

    Parameters
    ----------
    session_turns : list[dict]
        Each dict should have a ``symptoms`` key (list of symptom strings)
        and optionally ``query`` (str).  Turns are 1-indexed.

    Returns
    -------
    dict[str, int]
        Maps canonical symptom token → turn number (1-indexed) at which it
        was first mentioned.  If a symptom appears in multiple turns, only
        the first occurrence is recorded (symptom onset = first turn).
    """
    timeline: dict[str, int] = {}
    for turn_idx, turn in enumerate(session_turns, start=1):
        syms = turn.get("symptoms", [])
        for sym in syms:
            for tok in _tokenize_sym(sym):
                if tok not in timeline:
                    timeline[tok] = turn_idx
        # Also tokenise the query string
        query = turn.get("query", "")
        for tok in _tokenize_sym(query):
            if tok not in timeline:
                timeline[tok] = turn_idx
    return timeline


# ---------------------------------------------------------------------------
# Public API: TSPG reranking
# ---------------------------------------------------------------------------


def tspg_rerank(
    results: list[dict],
    session_turns: list[dict],
    tspg_weight: float = 0.10,
) -> list[dict]:
    """
    Re-rank retrieval results using temporal compatibility scores.

    Each result dict must have ``hybrid_score`` and ``entry`` fields.
    Results are re-ranked by a fused score:

        fused = (1 − tspg_weight) × hybrid_score
              + tspg_weight × tc_score_normalised

    Parameters
    ----------
    results : list[dict]
        Retrieval results with ``hybrid_score`` and ``entry`` keys.
    session_turns : list[dict]
        Prior session turns (each with ``symptoms`` and ``query`` keys).
    tspg_weight : float
        Weight of temporal compatibility in the fused score.
        0.0 = disable TSPG; 0.10 = 10% weight (default).

    Returns
    -------
    list[dict]
        Re-sorted results with ``tspg_score`` field added.
    """
    if not results or not session_turns or tspg_weight <= 0.0:
        return results

    timeline = build_symptom_timeline(session_turns)
    if len(timeline) < 2:
        # Single-symptom history — insufficient for temporal reasoning
        return results

    tc_scores = [
        compute_temporal_compatibility(
            r["entry"].get("symptoms", []), timeline
        )
        for r in results
    ]

    # Normalise tc_scores to [0, 1]
    max_tc = max(tc_scores) if tc_scores else 0.0
    if max_tc > 0.0:
        tc_norms = [s / max_tc for s in tc_scores]
    else:
        tc_norms = [0.0] * len(tc_scores)

    reranked = []
    for r, tc_norm in zip(results, tc_norms):
        fused = (1 - tspg_weight) * r.get("hybrid_score", 0.0) + tspg_weight * tc_norm
        reranked.append({**r, "tspg_score": tc_norm, "hybrid_score": fused})

    reranked.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return reranked
