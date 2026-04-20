#!/usr/bin/env python3
"""
Conformal Prediction Calibration Experiment
============================================
Implements a leave-one-out (LOO) calibration sweep over α ∈ {0.05, 0.10, 0.20}
to empirically verify the coverage guarantee of the conformal predictor.

Theory
------
The conformal prediction guarantee (Vovk, 2005; Angelopoulos & Bates, 2023):

    P(y ∈ Ĉ(x)) ≥ 1 − α

must hold for any α and any data distribution.  This script measures the
**empirical coverage** and **average prediction set size** at each α level
using leave-one-out cross-validation on the KB.

Protocol (LOO Conformal Calibration)
--------------------------------------
For each KB entry i:
1. Use all entries EXCEPT i as the calibration set.
2. Compute non-conformity scores A(x_j, y_j) = 1 − softmax_score(y_j | x_j)
   for all j ≠ i.
3. Find quantile threshold q̂ at level ⌈(n+1)(1−α)⌉/n.
4. Check whether y_i ∈ Ĉ_α(x_i) — i.e., A(x_i, y_i) ≤ q̂.

Metrics Reported
----------------
  coverage(α)    : empirical P(y ∈ Ĉ(x)) across all LOO folds.
  avg_set_size(α): average |Ĉ_α(x)| — smaller = more specific.
  coverage_gap(α): empirical_coverage − (1 − α) [should be ≥ 0, ≤ 0.05].

References
----------
Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a
Random World*.  Springer.

Angelopoulos, A.N., & Bates, S. (2023). Conformal prediction: A gentle
introduction. *Foundations and Trends in Machine Learning*, 16(4), 494–591.

Lei, J., & Wasserman, L. (2014). Distribution-free prediction bands for
non-parametric regression. *JRSS-B*, 76(1), 71–96.
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag.dataset import MedicalDataset
from app.rag.retriever import Retriever
from app.rag.query_processor import QueryProcessor

# Alpha levels to sweep
ALPHA_LEVELS: list[float] = [0.05, 0.10, 0.20]

# Number of top-k results per query
TOP_K: int = 5


def _nonconformity_score(
    retriever: Retriever,
    condition_name: str,
    query_tokens: list[str],
    top_k: int,
) -> float:
    """
    Compute non-conformity score A(x, y) = 1 − softmax_score(y | x).

    The softmax_score is the normalised hybrid score of the correct condition
    among the top-k retrieved results.  If y is not in the top-k, A = 1.0.
    """
    results = retriever.retrieve_results(" ".join(query_tokens), query_tokens, top_k)
    total_score = sum(r.hybrid_score for r in results)
    if total_score <= 0.0:
        return 1.0
    for r in results:
        if r.entry.get("condition", "") == condition_name:
            return 1.0 - r.hybrid_score / total_score
    return 1.0  # correct condition not retrieved at all


def _empirical_quantile(scores: list[float], alpha: float) -> float:
    """
    Compute the empirical (1−α) quantile of the calibration scores.

    Uses the standard conformal quantile formula:
        q̂ = ⌈(n+1)(1−α)⌉ / n -th order statistic.
    """
    n = len(scores)
    if n == 0:
        return 1.0
    sorted_scores = sorted(scores)
    level = math.ceil((n + 1) * (1 - alpha)) / n
    idx = min(int(math.floor(level * n)) - 1, n - 1)
    idx = max(idx, 0)
    return sorted_scores[idx]


def run_loo_calibration(
    entries: list[dict],
    alpha_levels: list[float] = ALPHA_LEVELS,
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Run the LOO conformal calibration sweep.

    Uses a single full retriever for consistent non-conformity scoring
    across both calibration and test splits (inductive conformal approach).
    For fold i, the calibration scores are A(x_j, y_j) for all j ≠ i,
    computed on the same retriever as the test score A(x_i, y_i).

    Parameters
    ----------
    entries : list[dict]
        KB entries.
    alpha_levels : list[float]
        Error rates to evaluate.
    top_k : int
        Retrieval depth for scoring.

    Returns
    -------
    list[dict]
        One result dict per alpha level.
    """
    n = len(entries)
    if n < 4:
        print("WARNING: KB too small for meaningful LOO calibration (n < 4)")
        return []

    # Build a single full retriever (inductive conformal uses one fixed model)
    full_retriever = Retriever(entries)

    # Pre-compute non-conformity scores for all entries using the full retriever
    # A(x_j, y_j) = 1 - normalised_score(y_j | x_j)
    all_nc_scores: list[float] = []
    all_valid: list[bool] = []

    for j, entry in enumerate(entries):
        syms = entry.get("symptoms", [])
        cond = entry.get("condition", "")
        if not syms or not cond:
            all_nc_scores.append(1.0)
            all_valid.append(False)
            continue
        pq = QueryProcessor.process(" ".join(syms[:3]), syms[:3])
        tokens = pq.medical_tokens or pq.affirmed_terms or syms[:3]
        if not tokens:
            all_nc_scores.append(1.0)
            all_valid.append(False)
            continue
        nc = _nonconformity_score(full_retriever, cond, tokens, max(top_k, n))
        all_nc_scores.append(nc)
        all_valid.append(True)

    results = []
    for alpha in alpha_levels:
        covered = 0
        total_set_size = 0.0
        valid_folds = 0

        for i in range(n):
            if not all_valid[i]:
                continue

            # Calibration scores: all j ≠ i that are valid
            cal_scores = [
                all_nc_scores[j] for j in range(n) if j != i and all_valid[j]
            ]
            if not cal_scores:
                continue

            # Quantile threshold from calibration set
            q_hat = _empirical_quantile(cal_scores, alpha)

            # Prediction set: all conditions with A ≤ q̂
            pred_set = [
                entries[j].get("condition", "")
                for j in range(n)
                if all_nc_scores[j] <= q_hat and all_valid[j]
            ]

            # Coverage: is the correct condition in the prediction set?
            test_condition = entries[i].get("condition", "")
            in_set = any(
                test_condition.lower() in c.lower() or c.lower() in test_condition.lower()
                for c in pred_set
            )
            covered += int(in_set)
            total_set_size += len(pred_set)
            valid_folds += 1

        if valid_folds == 0:
            continue

        emp_coverage = covered / valid_folds
        avg_set_size = total_set_size / valid_folds
        promised_coverage = 1.0 - alpha
        gap = emp_coverage - promised_coverage

        results.append({
            "alpha": alpha,
            "promised_coverage": round(promised_coverage, 4),
            "empirical_coverage": round(emp_coverage, 4),
            "coverage_gap": round(gap, 4),
            "avg_set_size": round(avg_set_size, 3),
            "n_folds": valid_folds,
            "guarantee_satisfied": emp_coverage >= promised_coverage,
        })

    return results


def _print_calibration_table(results: list[dict]) -> None:
    """Print calibration results as a formatted table."""
    print()
    print("=" * 65)
    print("Conformal Predictor Calibration (LOO Cross-Validation)")
    print("=" * 65)
    print(f"{'α':>6}  {'Promised':>9}  {'Empirical':>10}  {'Gap':>6}  {'Avg |Ĉ|':>8}  {'OK?':>5}")
    print("-" * 65)
    for r in results:
        ok = "✓" if r["guarantee_satisfied"] else "✗"
        print(
            f"{r['alpha']:>6.2f}  "
            f"{r['promised_coverage']:>9.4f}  "
            f"{r['empirical_coverage']:>10.4f}  "
            f"{r['coverage_gap']:>+6.4f}  "
            f"{r['avg_set_size']:>8.3f}  "
            f"{ok:>5}"
        )
    print("=" * 65)
    all_ok = all(r["guarantee_satisfied"] for r in results)
    status = "✓ Coverage guarantee satisfied at all α levels." if all_ok else "✗ Coverage guarantee violated at some α levels."
    print(f"\n{status}")
    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Conformal prediction calibration sweep")
    parser.add_argument("--kb", default="data/sample_medical_knowledge.json")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    dataset = MedicalDataset(args.kb)
    if not dataset.entries:
        print(f"ERROR: No entries loaded from {args.kb}", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(dataset.entries)} KB entries.")
    print("Running LOO conformal calibration sweep...")

    results = run_loo_calibration(dataset.entries, ALPHA_LEVELS, args.top_k)

    if args.json:
        import json
        print(json.dumps(results, indent=2))
    else:
        _print_calibration_table(results)
