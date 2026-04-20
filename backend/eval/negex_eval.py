#!/usr/bin/env python3
"""
NegEx Implementation Evaluation
=================================
Evaluates the NegEx negation detector in ``query_processor.py`` against 30
manually annotated clinical sentences.  Reports precision, recall, and F1,
and compares against the baseline reported by Chapman et al. (2001).

Methodology
-----------
Chapman et al. (2001) evaluated NegEx on clinical discharge summaries and
reported an overall F1 of 0.84 (P=0.88, R=0.82) for detecting negated
medical findings.

This evaluation uses a curated set of 30 sentences covering:
  - Simple negation ("no fever")
  - Multi-word triggers ("does not have", "denies")
  - Window-boundary negation (negation > 4 tokens away)
  - Affirmative sentences (true positives for affirmed symptoms)
  - Edge cases (double negation, "neither...nor")

References
----------
Chapman, W.W. et al. (2001). A simple algorithm for identifying negated
findings and diseases in discharge summaries. *Journal of Biomedical
Informatics*, 34(5), 301–310.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag.query_processor import QueryProcessor


# ---------------------------------------------------------------------------
# 30 Annotated Test Sentences
# ---------------------------------------------------------------------------
# Format: (sentence, symptom_to_test, expected_negated: bool)
# expected_negated=True means the symptom IS negated in the sentence.
# expected_negated=False means the symptom is AFFIRMED.

@dataclass
class NegExSample:
    sentence: str
    symptom: str           # canonical symptom token to check
    expected_negated: bool  # True if symptom should be in negated_terms


ANNOTATED_SAMPLES: list[NegExSample] = [
    # ── Simple single-token negation triggers ────────────────────────────
    NegExSample("I have no fever.",                          "fever",       True),
    NegExSample("She has no chest pain.",                    "chest",       True),
    NegExSample("Patient denies shortness of breath.",       "shortness",   True),
    NegExSample("He denies any cough.",                      "cough",       True),
    NegExSample("Without dizziness or nausea.",              "dizziness",   True),
    NegExSample("Without dizziness or nausea.",              "nausea",      True),
    NegExSample("Absent reflexes.",                          "reflexes",    True),
    NegExSample("There is absence of jaundice.",             "jaundice",    True),
    NegExSample("Never had headaches before.",               "headaches",   True),
    NegExSample("None of the above symptoms are present.",   "symptoms",    True),

    # ── Multi-token negation triggers ────────────────────────────────────
    NegExSample("She does not have a rash.",                 "rash",        True),
    NegExSample("Patient cannot tolerate exercise.",         "exercise",    True),
    NegExSample("He does not have a productive cough.",      "productive",  True),
    NegExSample("Negative for fever and chills.",            "fever",       True),
    NegExSample("Negative for fever and chills.",            "chills",      True),
    NegExSample("Rules out pulmonary embolism.",             "pulmonary",   True),
    NegExSample("No sign of bleeding.",                      "bleeding",    True),
    NegExSample("No evidence of pneumonia.",                 "pneumonia",   True),
    NegExSample("Neither swelling nor tenderness.",          "swelling",    True),
    NegExSample("Neither swelling nor tenderness.",          "tenderness",  True),

    # ── Affirmative sentences (should NOT be negated) ────────────────────
    NegExSample("Patient has fever and chills.",             "fever",       False),
    NegExSample("Chest pain radiating to the left arm.",     "chest",       False),
    NegExSample("Experiencing shortness of breath.",         "shortness",   False),
    NegExSample("Persistent cough for 2 weeks.",             "cough",       False),
    NegExSample("Patient reports dizziness on standing.",    "dizziness",   False),
    NegExSample("Nausea and vomiting present.",              "nausea",      False),

    # ── Edge cases ────────────────────────────────────────────────────────
    # Window boundary: negation trigger > 4 tokens from symptom
    NegExSample("No known allergies but patient does have severe headache.", "headache", False),
    # Negation in subordinate clause
    NegExSample("The patient, who has no prior cardiac history, now presents with chest pain.", "chest", False),
    # Double context (affirmed after negation — challenging)
    NegExSample("No fever today but patient had fever last week.", "fever", True),
    # Unrelated negation
    NegExSample("Not sure about the diagnosis, but fever is present.", "fever", False),
]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_negex(samples: list[NegExSample]) -> dict:
    """
    Run the NegEx implementation on all samples and compute P/R/F1.

    Returns
    -------
    dict with keys: tp, fp, tn, fn, precision, recall, f1
    """
    tp = fp = tn = fn = 0

    predictions: list[dict] = []
    for sample in samples:
        pq = QueryProcessor.process(sample.sentence, [])
        negated_tokens = {t.lower().strip(".,!?;:()[]\"'") for t in pq.negated_terms}
        predicted_negated = sample.symptom.lower() in negated_tokens

        if sample.expected_negated and predicted_negated:
            tp += 1
        elif sample.expected_negated and not predicted_negated:
            fn += 1
        elif not sample.expected_negated and predicted_negated:
            fp += 1
        else:
            tn += 1

        predictions.append({
            "sentence": sample.sentence[:60],
            "symptom": sample.symptom,
            "expected_negated": sample.expected_negated,
            "predicted_negated": predicted_negated,
            "correct": sample.expected_negated == predicted_negated,
        })

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(samples) if samples else 0.0

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "n_samples": len(samples),
        "predictions": predictions,
    }


def print_comparison(our_metrics: dict) -> None:
    """Print comparison table against Chapman et al. (2001) baseline."""
    print()
    print("=" * 50)
    print("NegEx Evaluation — Comparison with Chapman et al. (2001)")
    print("=" * 50)
    print(f"{'Metric':<12} {'Our NegEx':>12} {'Chapman 2001':>14}")
    print("-" * 50)
    # Chapman et al. reported (on discharge summaries):
    chapman = {"precision": 0.88, "recall": 0.82, "f1": 0.84}
    for metric in ("precision", "recall", "f1"):
        ours = our_metrics[metric]
        ref = chapman[metric]
        delta = ours - ref
        print(f"{metric:<12} {ours:>12.4f} {ref:>14.4f}  ({delta:+.4f})")
    print("=" * 50)
    print(f"Accuracy on 30 sentences: {our_metrics['accuracy']:.4f}")
    print(f"TP={our_metrics['tp']}, FP={our_metrics['fp']}, "
          f"TN={our_metrics['tn']}, FN={our_metrics['fn']}")
    print()


def print_failures(predictions: list[dict]) -> None:
    """Print sentences where the prediction was wrong."""
    failures = [p for p in predictions if not p["correct"]]
    if not failures:
        print("All 30 sentences correctly classified!")
        return
    print(f"\nMisclassified sentences ({len(failures)}/{len(predictions)}):")
    for p in failures:
        status = "FP" if p["predicted_negated"] and not p["expected_negated"] else "FN"
        print(f"  [{status}] '{p['sentence'][:55]}...' — symptom='{p['symptom']}'")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NegEx implementation evaluation")
    parser.add_argument("--verbose", action="store_true", help="Print all predictions")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    metrics = evaluate_negex(ANNOTATED_SAMPLES)

    if args.json:
        import json
        print(json.dumps({k: v for k, v in metrics.items() if k != "predictions"}, indent=2))
    else:
        print_comparison(metrics)
        print_failures(metrics["predictions"])

        if args.verbose:
            print("\nFull predictions:")
            for p in metrics["predictions"]:
                ok = "✓" if p["correct"] else "✗"
                print(f"  {ok} '{p['sentence'][:55]}' | {p['symptom']} | "
                      f"exp={p['expected_negated']} pred={p['predicted_negated']}")
