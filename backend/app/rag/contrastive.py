"""
Contrastive Differential Diagnosis Analysis
============================================
For each adjacent pair of top-ranked conditions, computes a symptom contrast
matrix that identifies the distinguishing features between them.

Formal Definition
-----------------
For conditions A and B with canonical symptom token sets S_A and S_B:

    for_A       = S_A \\ S_B   (symptoms present in A but absent from B → rule in A)
    for_B       = S_B \\ S_A   (symptoms present in B but absent from A → rule in B)
    shared      = S_A ∩ S_B    (shared symptoms — non-discriminating)

These contrasts are derived exclusively from the knowledge base — no LLM
generation, no statistical learning — making the contrasts *hallucination-free
by construction*.

Clinical Significance
---------------------
The contrast answers the clinical question:

    "Given that both condition A and condition B fit the patient's symptoms,
     what would distinguish them?"

This is the central skill of differential diagnosis training in medical education
(Coderre et al., 2003).  Classical clinical reasoning frameworks (illness script
theory, Schmidt & Rikers, 2007) require explicitly comparing competing hypotheses
against each other, not just against the symptoms in isolation.

The contrast is also intersected with the patient's *affirmed tokens* to surface
which of their reported symptoms are actually discriminating:

    patient_discriminating_A = for_A ∩ affirmed_tokens
    patient_discriminating_B = for_B ∩ affirmed_tokens

If patient_discriminating_A is non-empty but patient_discriminating_B is empty,
this provides strong Bayesian evidence for A over B — quantifiable as:

    P(A | patient_symptoms) / P(B | patient_symptoms)
     ≥ P(patient_discriminating_A | A) / P(patient_discriminating_A | B)

A contrast matrix with zero patient-discriminating symptoms in both directions
triggers the active inquiry module: the patient's current symptoms do not
distinguish the two leading hypotheses, and asking about a discriminating
symptom would resolve the ambiguity.

References
----------
Coderre, S., Mandin, H., Harasym, P., & Fick, G. (2003). Diagnostic reasoning
strategies and diagnostic success. *Medical Education*, 37(8), 695–703.

Schmidt, H.G., & Rikers, R.M. (2007). How expertise develops in medicine:
Knowledge encapsulation and illness script formation. *Medical Education*, 41(12).

Elstein, A., & Schwarz, A. (2002). Clinical problem solving and diagnostic
decision making. *British Medical Journal*, 324(7339), 729–732.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from app.rag.retriever import RetrievalResult


# ---------------------------------------------------------------------------
# Token normalisation
# ---------------------------------------------------------------------------


def _symptom_token_set(symptoms: list[str]) -> set[str]:
    """Flatten a symptom list into a set of lowercase canonical tokens."""
    tokens: set[str] = set()
    for sym in symptoms:
        for t in sym.lower().split():
            t_clean = t.strip(".,!?;:()[]\"'")
            if len(t_clean) > 2:
                tokens.add(t_clean)
    return tokens


# ---------------------------------------------------------------------------
# Bayes Factor computation
# ---------------------------------------------------------------------------
# Epsilon for Laplace smoothing to prevent division-by-zero and log(0)
_BF_EPSILON: float = 1e-3


def _symptom_token_freq(entry: dict) -> dict[str, float]:
    """
    Compute token frequency map for a KB entry's symptom list.

    Token frequency = count(token in symptom list) / len(symptom tokens).
    This is used as P(token | condition) for the Bayes factor.
    """
    tokens = _symptom_token_set(entry.get("symptoms", []))
    if not tokens:
        return {}
    freq = {t: 1.0 / len(tokens) for t in tokens}
    return freq


def compute_bayes_factor(
    entry_a: dict,
    entry_b: dict,
    discriminating_tokens: set[str],
) -> float:
    """
    Compute the Bayes Factor B₁₂ = P(disc_tokens | A) / P(disc_tokens | B).

    Under the Naive Bayes independence assumption (Sackett et al., 2000):

        B₁₂ = ∏_{t ∈ discriminating_tokens} [P(t | A) / P(t | B)]

    where P(t | condition) = (token_freq_in_symptoms + ε) / (1 + ε).

    Interpretation (Jeffreys, 1961):
        B₁₂ > 10    : strong evidence for A over B
        3 < B₁₂ ≤ 10: moderate evidence
        1 < B₁₂ ≤ 3 : anecdotal evidence
        B₁₂ = 1.0   : equal support (no discrimination)
        B₁₂ < 1.0   : evidence favours B over A

    Parameters
    ----------
    entry_a, entry_b : dict
        KB entries for conditions A and B.
    discriminating_tokens : set[str]
        Symptom tokens that discriminate A from B (for_a ∩ patient_affirmed).

    Returns
    -------
    float
        Log₁₀ Bayes factor.  Positive = favours A, negative = favours B.
        Returns 0.0 if no discriminating tokens.
    """
    if not discriminating_tokens:
        return 0.0

    freq_a = _symptom_token_freq(entry_a)
    freq_b = _symptom_token_freq(entry_b)

    log_bf = 0.0
    for token in discriminating_tokens:
        p_a = freq_a.get(token, _BF_EPSILON)
        p_b = freq_b.get(token, _BF_EPSILON)
        # Laplace smoothing: ensure probabilities are non-zero
        p_a = max(p_a, _BF_EPSILON)
        p_b = max(p_b, _BF_EPSILON)
        log_bf += math.log10(p_a / p_b)

    return round(log_bf, 3)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_pairwise_contrasts(
    results: list["RetrievalResult"],
    affirmed_tokens: set[str],
    top_n_pairs: int = 2,
) -> list[dict]:
    """
    Compute symptom contrasts for the top adjacent condition pairs.

    Parameters
    ----------
    results : list[RetrievalResult]
        Ranked retrieval results (index 0 = highest ranked).
    affirmed_tokens : set[str]
        Non-negated medical tokens from the patient's query.
    top_n_pairs : int
        Number of adjacent pairs to analyse (default: top-1 vs top-2,
        and top-2 vs top-3 when available).

    Returns
    -------
    list[dict], one entry per pair, each containing:
        ``condition_a``              — name of higher-ranked condition
        ``condition_b``              — name of lower-ranked condition
        ``for_a``                    — symptom tokens that rule in A over B
        ``for_b``                    — symptom tokens that rule in B over A
        ``shared``                   — symptom tokens shared by both
        ``patient_discriminating_a`` — for_a ∩ patient affirmed tokens
        ``patient_discriminating_b`` — for_b ∩ patient affirmed tokens
        ``ambiguous``                — True if patient has NO discriminating
                                       symptoms for either condition (→ ask
                                       the active inquiry question)
    """
    if len(results) < 2:
        return []

    n_pairs = min(top_n_pairs, len(results) - 1)
    contrasts: list[dict] = []

    for i in range(n_pairs):
        a = results[i]
        b = results[i + 1]

        syms_a = _symptom_token_set(a.entry.get("symptoms", []))
        syms_b = _symptom_token_set(b.entry.get("symptoms", []))

        for_a = syms_a - syms_b
        for_b = syms_b - syms_a
        shared = syms_a & syms_b

        pat_disc_a = for_a & affirmed_tokens
        pat_disc_b = for_b & affirmed_tokens

        # Ambiguous: neither discriminating direction is confirmed by patient
        ambiguous = (not pat_disc_a) and (not pat_disc_b) and (bool(for_a) or bool(for_b))

        # Bayes factor: how strongly do patient's discriminating symptoms
        # favour condition A over condition B (log₁₀ scale).
        bf_a_over_b = compute_bayes_factor(a.entry, b.entry, pat_disc_a) if pat_disc_a else 0.0
        bf_b_over_a = compute_bayes_factor(b.entry, a.entry, pat_disc_b) if pat_disc_b else 0.0

        contrasts.append(
            {
                "condition_a": a.entry.get("condition", ""),
                "condition_b": b.entry.get("condition", ""),
                "for_a": sorted(for_a),
                "for_b": sorted(for_b),
                "shared": sorted(shared),
                "patient_discriminating_a": sorted(pat_disc_a),
                "patient_discriminating_b": sorted(pat_disc_b),
                "ambiguous": ambiguous,
                # Bayes factor (log₁₀): > 0 favours A over B, < 0 favours B over A
                "log10_bayes_factor_a_over_b": bf_a_over_b,
                "log10_bayes_factor_b_over_a": bf_b_over_a,
            }
        )

    return contrasts


def format_contrasts_for_prompt(contrasts: list[dict]) -> str:
    """
    Render pairwise contrasts as structured text for injection into the
    LLM CoT prompt (Step 9 of the MedCoT-DDx template).

    Includes Bayes factors to provide quantitative strength of evidence.
    """
    if not contrasts:
        return "  No pairwise contrast available."

    def _bf_label(log10_bf: float) -> str:
        """Convert log₁₀ Bayes factor to Jeffreys interpretation string."""
        if log10_bf >= 1.0:
            return f"BF={10**log10_bf:.1f} (strong)"
        if log10_bf >= 0.48:
            return f"BF={10**log10_bf:.1f} (moderate)"
        if log10_bf > 0.0:
            return f"BF={10**log10_bf:.1f} (anecdotal)"
        return ""

    lines: list[str] = []
    for c in contrasts:
        a, b = c["condition_a"], c["condition_b"]
        lines.append(f"  {a} vs {b}:")

        if c["for_a"]:
            lines.append(
                f"    → Rules in {a}: {', '.join(sorted(c['for_a'])[:6])}"
            )
        if c["for_b"]:
            lines.append(
                f"    → Rules in {b}: {', '.join(sorted(c['for_b'])[:6])}"
            )
        if c["patient_discriminating_a"]:
            bf_str = _bf_label(c.get("log10_bayes_factor_a_over_b", 0.0))
            bf_suffix = f"  [{bf_str}]" if bf_str else ""
            lines.append(
                f"    ✓ Patient has (supports {a}): "
                f"{', '.join(c['patient_discriminating_a'])}{bf_suffix}"
            )
        if c["patient_discriminating_b"]:
            bf_str = _bf_label(c.get("log10_bayes_factor_b_over_a", 0.0))
            bf_suffix = f"  [{bf_str}]" if bf_str else ""
            lines.append(
                f"    ✓ Patient has (supports {b}): "
                f"{', '.join(c['patient_discriminating_b'])}{bf_suffix}"
            )
        if c.get("ambiguous"):
            lines.append(
                f"    ⚠ Ambiguous: patient's current symptoms do not distinguish "
                f"{a} from {b} — consider asking a discriminating question."
            )

    return "\n".join(lines)
