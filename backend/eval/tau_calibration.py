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

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  EVALUATION VALIDITY WARNING (M1 fix)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  SIMULATION MODE (default, --mode simulate):
    The default mode models the LLM as "retriever + Gaussian noise".
    This is NOT a valid calibration for publication because:
      1. It is circular: the "LLM" is defined as mostly agreeing with
         the retriever, so τ will always be close to 1.0.
      2. The Gaussian noise model has no empirical grounding — it does
         not reflect actual GPT-4 / LLaMA disagreement patterns.
      3. The F1 curve will show an optimal τ near the noise boundary,
         which depends only on the arbitrary NOISE_STD parameter.
    SIMULATION results MUST NOT be reported as τ calibration in a paper.

  REAL LLM MODE (--mode real --llm-outputs-file <CSV>):
    For valid calibration, collect actual LLM outputs on the calibration
    queries, compute real Kendall's τ between the LLM and retriever
    rankings, and find the threshold that maximises F1 vs. gold labels.

    To produce the required CSV file, run your LLM on each calibration
    query with the prompt:
        "Rank the following conditions from most to least likely given
         these symptoms: {conditions}. Output a comma-separated ranked
         list with the most likely condition first."
    Then record whether the LLM top-1 matches ground truth (is_correct=1/0).

    CSV format:
        query_id,tau,llm_top1_correct
        q1,0.71,1
        q2,0.43,0
        q3,0.88,1

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Motivation
----------
The default τ = 0.50 is a reasonable choice, but it should be justified
empirically.  This script shows the override rate and F1 as a function of τ
and identifies the optimal threshold, turning an arbitrary constant into a
principled, data-driven hyperparameter.

Protocol (Simulation Mode — Development Only)
---------------------------------------------
Since we don't have LLM outputs at evaluation time, we simulate LLM orderings
by adding Gaussian noise (σ = NOISE_STD) to the Bayesian scores to model the
LLM's imperfect but roughly-correct ranking.  This simulates the realistic
scenario where the LLM mostly agrees with Bayesian order but occasionally
transposes conditions.

This is explicitly a development aid, not a publishable calibration.
Ground Truth: Bayesian ranking (Retriever hybrid scores, descending).
LLM Simulation: Bayesian scores + N(0, σ²), re-sorted.
"""
from __future__ import annotations

import csv
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

    NOTE: This is a development-only simulation, NOT a substitute for
    real LLM outputs.  See module docstring for guidance on collecting
    real LLM outputs.
    """
    if len(true_order) < 2:
        return true_order[:]
    noisy = [(s + rng.gauss(0, noise_std), c) for s, c in zip(scores, true_order)]
    noisy.sort(reverse=True)
    return [c for _, c in noisy]


def load_real_llm_outputs(path: str) -> list[dict]:
    """
    Load real LLM outputs from a CSV file for valid calibration.

    CSV format:
        query_id,tau,llm_top1_correct
        q1,0.71,1
        q2,0.43,0

    Parameters
    ----------
    path : str
        Path to the CSV file produced by running the LLM on calibration queries.

    Returns
    -------
    list[dict] with keys: query_id, tau, llm_top1_correct (bool)
    """
    outputs = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            outputs.append({
                "query_id": row["query_id"],
                "tau": float(row["tau"]),
                "llm_top1_correct": bool(int(row.get("llm_top1_correct", 0))),
            })
    return outputs


def calibrate_tau_real(
    llm_outputs: list[dict],
    tau_values: list[float] = TAU_VALUES,
) -> list[dict]:
    """
    Calibrate τ threshold from real LLM outputs.

    Uses actual Kendall's τ values measured between real LLM rankings
    and Bayesian retriever rankings.  This is the valid calibration
    method for publication.

    Parameters
    ----------
    llm_outputs : list[dict]
        Loaded from CSV via load_real_llm_outputs().
    tau_values : list[float]
        Threshold values to sweep.

    Returns
    -------
    list[dict], one per τ value.
    """
    tau_results = []
    for threshold in tau_values:
        tp = fp = tn = fn = 0
        overrides = 0

        for case in llm_outputs:
            actual_tau = case["tau"]
            llm_correct = case["llm_top1_correct"]
            should_override = actual_tau < threshold
            overrides += int(should_override)

            llm_wrong = not llm_correct
            if should_override and llm_wrong:
                tp += 1
            elif should_override and not llm_wrong:
                fp += 1
            elif not should_override and llm_wrong:
                fn += 1
            else:
                tn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        tau_results.append({
            "tau": threshold,
            "override_rate": round(overrides / len(llm_outputs), 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "n_samples": len(llm_outputs),
            "_mode": "real",
        })

    return tau_results


def calibrate_tau_simulated(
    entries: list[dict],
    tau_values: list[float] = TAU_VALUES,
    top_k: int = 5,
) -> list[dict]:
    """
    Calibrate τ threshold using simulated LLM outputs (development only).

    ⚠ WARNING: This produces results based on a Gaussian noise model of
    the LLM.  These results are not valid for publication.  Use
    calibrate_tau_real() with actual LLM outputs for valid calibration.

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
            "_mode": "simulated",
        })

    return tau_results


# Keep old function name for backward compatibility
def calibrate_tau(
    entries: list[dict],
    tau_values: list[float] = TAU_VALUES,
    top_k: int = 5,
) -> list[dict]:
    """Backward-compatible alias for calibrate_tau_simulated."""
    return calibrate_tau_simulated(entries, tau_values, top_k)


def _print_calibration(results: list[dict]) -> None:
    """Print τ calibration curve."""
    if not results:
        print("No results to display.")
        return
    best = max(results, key=lambda r: r["f1"])
    mode = results[0].get("_mode", "unknown")

    if mode == "simulated":
        print()
        print("⚠ " * 30)
        print("SIMULATION MODE — These results are NOT valid for publication.")
        print(f"LLM modelled as Gaussian noise (σ={NOISE_STD}) on retriever scores.")
        print("Use --mode real --llm-outputs-file <CSV> for valid calibration.")
        print("⚠ " * 30)

    print()
    print("=" * 65)
    print(f"Kendall τ Threshold Calibration for Critique 3  [{mode.upper()} MODE]")
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

    if mode == "simulated":
        print()
        print("IMPORTANT: To produce valid calibration for publication:")
        print("  1. Run your LLM on the CALIBRATION_QUERIES above.")
        print("  2. Record the LLM's condition ranking for each query.")
        print("  3. Compute Kendall's τ between LLM and retriever rankings.")
        print("  4. Determine whether each LLM top-1 matches ground truth.")
        print("  5. Save to CSV: query_id,tau,llm_top1_correct")
        print("  6. Re-run: python eval/tau_calibration.py --mode real --llm-outputs-file <CSV>")
    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Kendall τ threshold calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--kb", default="data/sample_medical_knowledge.json")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--mode",
        choices=["simulate", "real"],
        default="simulate",
        help=(
            "Calibration mode. 'simulate' uses Gaussian noise (dev only, not publishable). "
            "'real' requires --llm-outputs-file with actual LLM rankings."
        ),
    )
    parser.add_argument(
        "--llm-outputs-file",
        default=None,
        help=(
            "CSV file with real LLM outputs (required for --mode real). "
            "Columns: query_id,tau,llm_top1_correct"
        ),
    )
    args = parser.parse_args()

    if args.mode == "real":
        if not args.llm_outputs_file:
            parser.error(
                "--llm-outputs-file is required for --mode real. "
                "See module docstring for CSV format and how to collect LLM outputs."
            )
        print(f"Loading real LLM outputs from {args.llm_outputs_file} ...")
        llm_outputs = load_real_llm_outputs(args.llm_outputs_file)
        print(f"  Loaded {len(llm_outputs)} LLM output records.")
        results = calibrate_tau_real(llm_outputs)
    else:
        print()
        print("Running in SIMULATION MODE (development only — not publishable).")
        print("For valid calibration, use --mode real --llm-outputs-file <CSV>.")
        print()
        dataset = MedicalDataset(args.kb)
        if not dataset.entries:
            print(f"ERROR: No entries loaded from {args.kb}", file=sys.stderr)
            sys.exit(1)
        print(f"Loaded {len(dataset.entries)} KB entries.")
        results = calibrate_tau_simulated(dataset.entries)

    if args.json:
        import json
        print(json.dumps(results, indent=2))
    else:
        _print_calibration(results)
