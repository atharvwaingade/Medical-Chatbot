#!/usr/bin/env python3
"""
Kendall's τ Coherence Critique Threshold Calibration
======================================================
Calibrates the COHERENCE_TAU_MIN threshold used by Critique 3 (Bayesian
Coherence) in the Self-RAG loop.  Sweeps τ ∈ [0.2, 0.8] on the KB and
reports:
  - Override rate: fraction of queries where the LLM ordering is overridden.
  - F1 vs. ground truth (KB ground-truth ordering = retriever Bayesian rank).
  - The optimal τ that maximises F1.

Motivation
----------
The default τ = 0.50 is a reasonable choice, but it should be justified
empirically.  This script shows the override rate and F1 as a function of τ
and identifies the optimal threshold, turning an arbitrary constant into a
principled, data-driven hyperparameter.

Protocol
--------
Since we don't have LLM outputs at evaluation time, we simulate LLM orderings
by adding Gaussian noise (σ = NOISE_STD) to the Bayesian scores to model the
LLM's imperfect but roughly-correct ranking.  This simulates the realistic
scenario where the LLM mostly agrees with Bayesian order but occasionally
transposes conditions.

Ground Truth: Bayesian ranking (Retriever hybrid scores, descending).
LLM Simulation: Bayesian scores + N(0, σ²), re-sorted.
"""
from __future__ import annotations

import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag.dataset import MedicalDataset
from app.rag.retriever import Retriever
from app.rag.query_processor import QueryProcessor
from app.rag.pipeline import _kendall_tau

# Tau sweep range
TAU_VALUES: list[float] = [round(0.2 + i * 0.05, 2) for i in range(13)]  # 0.20, 0.25, ..., 0.80

# LLM noise simulation parameters
NOISE_STD: float = 0.15     # std of Gaussian noise added to scores
N_SIMULATIONS: int = 5      # simulated LLM orderings per query (for variance)
RANDOM_SEED: int = 42

# Queries to use for calibration
CALIBRATION_QUERIES: list[tuple[str, list[str]]] = [
    ("chest pain shortness of breath sweating", ["chest pain", "shortness of breath"]),
    ("fever cough body aches", ["fever", "cough", "body aches"]),
    ("burning urination frequent urination", ["burning urination", "frequent urination"]),
    ("severe headache nausea photophobia", ["severe headache", "nausea"]),
    ("fatigue weight gain cold intolerance dry skin", ["fatigue", "cold intolerance"]),
    ("increased thirst frequent urination blurred vision", ["increased thirst", "frequent urination"]),
    ("persistent sadness loss of interest insomnia", ["persistent sadness", "loss of interest"]),
    ("shortness of breath swollen legs", ["shortness of breath", "leg swelling"]),
    ("sudden weakness facial droop slurred speech", ["sudden weakness", "facial droop"]),
    ("runny nose sore throat nasal congestion", ["runny nose", "sore throat"]),
    ("palpitations heat intolerance weight loss tremor", ["palpitations", "heat intolerance"]),
    ("abdominal pain nausea vomiting diarrhea", ["abdominal pain", "nausea", "vomiting"]),
    ("productive cough fever chills shortness of breath", ["productive cough", "fever"]),
    ("headache stiff neck high fever", ["headache", "stiff neck", "fever"]),
    ("unexplained weight loss cough fatigue", ["unexplained weight loss", "cough"]),
]


def _simulate_llm_order(
    true_order: list[str],
    scores: list[float],
    noise_std: float,
    rng: random.Random,
) -> list[str]:
    """
    Simulate LLM ranking by adding Gaussian noise to Bayesian scores.

    Models realistic LLM imperfection: usually agrees with Bayesian order
    but occasionally transposes closely-scored conditions.
    """
    if len(true_order) < 2:
        return true_order[:]
    noisy = [(s + rng.gauss(0, noise_std), c) for s, c in zip(scores, true_order)]
    noisy.sort(reverse=True)
    return [c for _, c in noisy]


def calibrate_tau(
    entries: list[dict],
    tau_values: list[float] = TAU_VALUES,
    top_k: int = 5,
) -> list[dict]:
    """
    Sweep τ thresholds and compute override rate + F1.

    Returns
    -------
    list[dict], one per τ value, with keys:
        tau, override_rate, precision, recall, f1, n_samples
    """
    rng = random.Random(RANDOM_SEED)
    retriever = Retriever(entries)

    # Collect all simulation instances
    all_cases: list[dict] = []
    for query, symptoms in CALIBRATION_QUERIES:
        pq = QueryProcessor.process(query, symptoms)
        results = retriever.retrieve_results(pq.expanded_query, pq.medical_tokens, top_k)
        if len(results) < 2:
            continue

        true_order = [r.entry.get("condition", "") for r in results]
        scores = [r.hybrid_score for r in results]

        for _ in range(N_SIMULATIONS):
            llm_order = _simulate_llm_order(true_order, scores, NOISE_STD, rng)
            tau = _kendall_tau(llm_order, true_order)
            all_cases.append({
                "true_order": true_order,
                "llm_order": llm_order,
                "tau": tau,
                # A "correct" LLM order means top-1 matches true top-1
                "llm_top1_correct": (llm_order[0] == true_order[0]) if llm_order else False,
            })

    if not all_cases:
        return []

    tau_results = []
    for threshold in tau_values:
        overrides = 0
        tp = fp = tn = fn = 0

        for case in all_cases:
            should_override = case["tau"] < threshold
            overrides += int(should_override)

            # After override: use true_order[0] as top prediction
            # Without override: use llm_order[0]
            final_top1_correct = case["true_order"][0] == case["true_order"][0] if should_override \
                else case["llm_top1_correct"]

            # Ground truth: is overriding the right decision?
            llm_wrong = not case["llm_top1_correct"]
            if should_override and llm_wrong:
                tp += 1  # correctly overrode wrong LLM
            elif should_override and not llm_wrong:
                fp += 1  # unnecessarily overrode correct LLM
            elif not should_override and llm_wrong:
                fn += 1  # missed override when LLM was wrong
            else:
                tn += 1  # correctly didn't override correct LLM

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        override_rate = overrides / len(all_cases)

        tau_results.append({
            "tau": threshold,
            "override_rate": round(override_rate, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "n_samples": len(all_cases),
        })

    return tau_results


def _print_calibration(results: list[dict]) -> None:
    """Print τ calibration curve."""
    best = max(results, key=lambda r: r["f1"])

    print()
    print("=" * 65)
    print("Kendall τ Threshold Calibration for Critique 3")
    print("=" * 65)
    print(f"{'τ':>5}  {'Override Rate':>14}  {'Precision':>10}  {'Recall':>7}  {'F1':>6}")
    print("-" * 65)
    for r in results:
        marker = " ◄ OPTIMAL" if r["tau"] == best["tau"] else ""
        print(
            f"{r['tau']:>5.2f}  {r['override_rate']:>14.4f}  "
            f"{r['precision']:>10.4f}  {r['recall']:>7.4f}  {r['f1']:>6.4f}{marker}"
        )
    print("=" * 65)
    print(f"\nOptimal τ = {best['tau']:.2f}  (F1 = {best['f1']:.4f})")
    print(f"Default τ = 0.50  (F1 = {next((r['f1'] for r in results if abs(r['tau']-0.50)<0.01), 'N/A')})")
    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Kendall τ threshold calibration")
    parser.add_argument("--kb", default="data/sample_medical_knowledge.json")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    dataset = MedicalDataset(args.kb)
    if not dataset.entries:
        print(f"ERROR: No entries loaded from {args.kb}", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(dataset.entries)} KB entries.")

    results = calibrate_tau(dataset.entries)

    if args.json:
        import json
        print(json.dumps(results, indent=2))
    else:
        _print_calibration(results)
