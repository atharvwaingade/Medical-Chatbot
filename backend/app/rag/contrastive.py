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
            }
        )

    return contrasts


def format_contrasts_for_prompt(contrasts: list[dict]) -> str:
    """
    Render pairwise contrasts as structured text for injection into the
    LLM CoT prompt (Step 9 of the MedCoT-DDx template).
    """
    if not contrasts:
        return "  No pairwise contrast available."

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
            lines.append(
                f"    ✓ Patient has (supports {a}): "
                f"{', '.join(c['patient_discriminating_a'])}"
            )
        if c["patient_discriminating_b"]:
            lines.append(
                f"    ✓ Patient has (supports {b}): "
                f"{', '.join(c['patient_discriminating_b'])}"
            )
        if c.get("ambiguous"):
            lines.append(
                f"    ⚠ Ambiguous: patient's current symptoms do not distinguish "
                f"{a} from {b} — consider asking a discriminating question."
            )

    return "\n".join(lines)
