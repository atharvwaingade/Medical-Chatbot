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
**empirical coverage** and **average prediction set size** at each α level.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  EVALUATION VALIDITY WARNING (M2 fix)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  TRAINING-SET MODE (default, --mode training):
    The default LOO protocol uses the same KB for both retrieval and
    calibration.  This violates the EXCHANGEABILITY ASSUMPTION of
    conformal prediction because:
      1. The retriever is fitted on the calibration entries.
      2. Non-conformity scores computed on training data are
         optimistically biased (the model has "seen" those entries).
      3. The reported coverage guarantee is formally invalid.
    This mode is provided for development/debugging ONLY.

  SPLIT-CONFORMAL MODE (--mode split --calibration-set <JSON>):
    For valid conformal prediction, you MUST provide a HELD-OUT
    calibration set that was NEVER used during retrieval model
    construction:
      1. Randomly hold out 20% of KB entries during system construction.
      2. Build the retriever on the remaining 80% (the "proper training set").
      3. Compute non-conformity scores on the 20% held-out set.
      4. Use those scores to find the quantile threshold q̂.
      5. Report empirical coverage on a separate TEST set.

    Steps to produce the calibration set JSON:
        python eval/conformal_coverage.py --split-kb \\
            --kb data/sample_medical_knowledge.json \\
            --cal-fraction 0.2 \\
            --cal-out data/cal_set.json \\
            --train-out data/train_set.json

    Then run split-conformal calibration:
        python eval/conformal_coverage.py --mode split \\
            --kb data/train_set.json \\
            --calibration-set data/cal_set.json

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Protocol (Split-Conformal Calibration — Valid for Publication)
--------------------------------------------------------------
1. Split KB into train (80%) and calibration (20%) sets.
2. Build retriever on train set only.
3. For each calibration entry i:
   a. Compute A(x_i, y_i) = 1 − softmax_score(y_i | x_i)  on train retriever.
4. For target α:
   a. Find q̂ = ⌈(n_cal + 1)(1 − α)⌉ / n_cal -th order stat of A values.
   b. Ĉ_α(x) = {y : A(x, y) ≤ q̂}
5. Report empirical coverage on SEPARATE TEST SET (not train, not cal).

Protocol (LOO Inductive Conformal — Training Set Only, Dev Only)
--------------------------------------------------------------
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

import json
import math
import os
import random
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


def run_split_conformal(
    train_entries: list[dict],
    cal_entries: list[dict],
    test_entries: list[dict],
    alpha_levels: list[float] = ALPHA_LEVELS,
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Run VALID split-conformal calibration on a proper held-out set.

    This is the correct method for computing a coverage guarantee.
    The retriever is built on train_entries ONLY.  Calibration scores are
    computed on cal_entries (never seen during training).  Coverage is
    measured on test_entries.

    Parameters
    ----------
    train_entries : list[dict]
        KB entries used to build the retriever.
    cal_entries : list[dict]
        Held-out entries for calibration (never used in retriever training).
    test_entries : list[dict]
        Independent test entries for coverage measurement.
    alpha_levels : list[float]
        Error rates to evaluate.
    top_k : int
        Retrieval depth for scoring.

    Returns
    -------
    list[dict], one per alpha level, with coverage guarantee validity flag.
    """
    if not cal_entries:
        print("ERROR: No calibration entries provided for split-conformal.", file=sys.stderr)
        return []
    if not test_entries:
        print("WARNING: No test entries — reporting calibration-set coverage only.",
              file=sys.stderr)
        test_entries = cal_entries  # fallback, but coverage estimates will be biased

    # Build retriever on TRAINING set only
    retriever = Retriever(train_entries)

    # Compute non-conformity scores on calibration set
    cal_nc: list[float] = []
    for entry in cal_entries:
        syms = entry.get("symptoms", [])
        cond = entry.get("condition", "")
        if not syms or not cond:
            continue
        pq = QueryProcessor.process(" ".join(syms[:3]), syms[:3])
        tokens = pq.medical_tokens or pq.affirmed_terms or syms[:3]
        if not tokens:
            continue
        nc = _nonconformity_score(retriever, cond, tokens, max(top_k, len(train_entries)))
        cal_nc.append(nc)

    if not cal_nc:
        print("WARNING: Could not compute any calibration non-conformity scores.",
              file=sys.stderr)
        return []

    results = []
    for alpha in alpha_levels:
        q_hat = _empirical_quantile(cal_nc, alpha)

        # Measure coverage on TEST set
        covered = 0
        total_set_size = 0.0
        valid_test = 0

        for entry in test_entries:
            syms = entry.get("symptoms", [])
            cond = entry.get("condition", "")
            if not syms or not cond:
                continue
            pq = QueryProcessor.process(" ".join(syms[:3]), syms[:3])
            tokens = pq.medical_tokens or pq.affirmed_terms or syms[:3]
            if not tokens:
                continue

            test_nc = _nonconformity_score(
                retriever, cond, tokens, max(top_k, len(train_entries))
            )
            in_set = test_nc <= q_hat
            covered += int(in_set)

            # Count set size
            set_size = sum(
                1 for e in train_entries
                if _nonconformity_score(retriever, e.get("condition", ""), tokens, top_k) <= q_hat
            )
            total_set_size += set_size
            valid_test += 1

        if valid_test == 0:
            continue

        emp_coverage = covered / valid_test
        avg_set_size = total_set_size / valid_test
        promised_coverage = 1.0 - alpha

        results.append({
            "alpha": alpha,
            "promised_coverage": round(promised_coverage, 4),
            "empirical_coverage": round(emp_coverage, 4),
            "coverage_gap": round(emp_coverage - promised_coverage, 4),
            "avg_set_size": round(avg_set_size, 3),
            "q_hat": round(q_hat, 4),
            "n_cal": len(cal_nc),
            "n_test": valid_test,
            "guarantee_satisfied": emp_coverage >= promised_coverage,
            "_mode": "split-conformal",
        })

    return results


def run_loo_calibration(
    entries: list[dict],
    alpha_levels: list[float] = ALPHA_LEVELS,
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Run the LOO conformal calibration sweep.

    ⚠ WARNING: This method calibrates on the SAME distribution used for
    retrieval.  The coverage guarantee is only valid under the assumption
    of exchangeability, which is violated when the retriever is fitted on
    the same data.  Use run_split_conformal() for valid results.

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
            "_mode": "loo-training-only",
        })

    return results


def split_kb(
    entries: list[dict],
    cal_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Split KB entries into train, calibration, and test sets.

    For split-conformal, use:
      train (60%) → build retriever
      cal   (20%) → compute calibration non-conformity scores
      test  (20%) → measure empirical coverage

    Returns
    -------
    (train_entries, cal_entries, test_entries)
    """
    rng = random.Random(seed)
    shuffled = entries[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_cal = max(1, int(n * cal_fraction))
    n_test = max(1, int(n * cal_fraction))
    n_train = n - n_cal - n_test
    if n_train < 2:
        # Not enough entries for 3-way split; use 2-way (train+cal, no test)
        n_train = max(1, n - n_cal)
        n_test = 0
    train = shuffled[:n_train]
    cal = shuffled[n_train:n_train + n_cal]
    test = shuffled[n_train + n_cal:]
    return train, cal, test


def _print_calibration_table(results: list[dict]) -> None:
    """Print calibration results as a formatted table."""
    if not results:
        print("No results to display.")
        return

    mode = results[0].get("_mode", "unknown")
    print()
    print("=" * 72)
    print(f"Conformal Predictor Calibration  [{mode.upper()}]")
    print("=" * 72)

    if mode == "loo-training-only":
        print("⚠  WARNING: LOO mode — calibrated on training distribution (invalid guarantee)")
        print("   For valid results, use: --mode split --calibration-set <held-out JSON>")
    elif mode == "split-conformal":
        print("✓  SPLIT-CONFORMAL MODE — calibrated on held-out set (valid guarantee)")

    print(f"{'α':>6}  {'Promised':>9}  {'Empirical':>10}  {'Gap':>6}  {'Avg |Ĉ|':>8}  {'OK?':>5}")
    print("-" * 72)
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
    print("=" * 72)
    all_ok = all(r["guarantee_satisfied"] for r in results)
    status = "✓ Coverage guarantee satisfied at all α levels." if all_ok \
        else "✗ Coverage guarantee violated at some α levels."
    print(f"\n{status}")

    if mode == "loo-training-only":
        print()
        print("To obtain a VALID coverage guarantee:")
        print("  1. Split KB: python eval/conformal_coverage.py --split-kb \\")
        print("       --kb data/sample_medical_knowledge.json \\")
        print("       --cal-out data/cal_set.json --train-out data/train_set.json")
        print("  2. Run: python eval/conformal_coverage.py --mode split \\")
        print("       --kb data/train_set.json \\")
        print("       --calibration-set data/cal_set.json")
    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Conformal prediction calibration sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--kb", default="data/sample_medical_knowledge.json",
                        help="KB JSON file (training set for split mode, full set for LOO)")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--mode",
        choices=["loo", "split"],
        default="loo",
        help=(
            "Calibration mode. "
            "'loo' = LOO on training set (invalid guarantee, dev only). "
            "'split' = split-conformal with held-out calibration set (valid for publication)."
        ),
    )
    parser.add_argument(
        "--calibration-set",
        default=None,
        help="Path to held-out calibration JSON (required for --mode split).",
    )
    parser.add_argument(
        "--test-set",
        default=None,
        help="Path to test JSON for coverage measurement (optional for --mode split).",
    )
    parser.add_argument(
        "--split-kb",
        action="store_true",
        help="Split KB into train/cal/test and save to separate files.",
    )
    parser.add_argument(
        "--cal-fraction",
        type=float,
        default=0.2,
        help="Fraction of KB to use as calibration set (default: 0.2).",
    )
    parser.add_argument("--cal-out", default="data/cal_set.json",
                        help="Output path for calibration set JSON.")
    parser.add_argument("--train-out", default="data/train_set.json",
                        help="Output path for training set JSON.")
    parser.add_argument("--test-out", default="data/test_set.json",
                        help="Output path for test set JSON.")
    args = parser.parse_args()

    dataset = MedicalDataset(args.kb)
    if not dataset.entries:
        print(f"ERROR: No entries loaded from {args.kb}", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(dataset.entries)} KB entries.")

    # --split-kb: create train/cal/test splits
    if args.split_kb:
        train, cal, test = split_kb(dataset.entries, args.cal_fraction)
        for path, data, label in [
            (args.train_out, train, "train"),
            (args.cal_out, cal, "calibration"),
            (args.test_out, test, "test"),
        ]:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            print(f"Saved {len(data)} {label} entries to {path}")
        sys.exit(0)

    if args.mode == "split":
        if not args.calibration_set:
            parser.error(
                "--calibration-set is required for --mode split. "
                "Run with --split-kb first to create a held-out calibration set."
            )
        with open(args.calibration_set, encoding="utf-8") as fh:
            cal_entries = json.load(fh)
        print(f"Loaded {len(cal_entries)} calibration entries from {args.calibration_set}.")

        test_entries: list[dict] = []
        if args.test_set:
            with open(args.test_set, encoding="utf-8") as fh:
                test_entries = json.load(fh)
            print(f"Loaded {len(test_entries)} test entries from {args.test_set}.")
        else:
            print("WARNING: No --test-set provided. Using calibration set for coverage "
                  "measurement (biased estimate).", file=sys.stderr)
            test_entries = cal_entries

        print("Running split-conformal calibration ...")
        results = run_split_conformal(
            dataset.entries, cal_entries, test_entries, ALPHA_LEVELS, args.top_k
        )
    else:
        print()
        print("Running in LOO mode (training-data calibration — invalid guarantee).")
        print("For valid calibration, use: --mode split --calibration-set <held-out JSON>")
        print()
        results = run_loo_calibration(dataset.entries, ALPHA_LEVELS, args.top_k)

    if args.json:
        import json as _json
        print(_json.dumps(results, indent=2))
    else:
        _print_calibration_table(results)
