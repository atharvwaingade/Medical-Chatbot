#!/usr/bin/env python3
"""
GPT-4 / GPT-4o-mini Zero-Shot Baseline for MedQA (USMLE-style)
================================================================
Runs two OpenAI model baselines on the MedQA test split (Jin et al. 2021)
and saves per-question outputs to results/gpt4_baseline_outputs.csv.

The output CSV is consumed by GPT4BaselineInterface in benchmark_runner.py.
That interface expects columns:
    question_id, predicted_condition, confidence_score, is_correct

This script writes those columns (using the predicted letter as
predicted_condition) PLUS the extra columns defined in this baseline:
    predicted_letter, correct_letter

So the CSV is a strict superset of what GPT4BaselineInterface needs.

Models evaluated
----------------
1. gpt-4-0125-preview  — full GPT-4 Turbo (high accuracy, higher cost)
2. gpt-4o-mini-2024-07-18 — cost-efficient miniature GPT-4o

Sampling
--------
By default a stratified random sample of 200 questions is drawn from the
test split (seed=42) to minimise API cost.  Pass --full to evaluate all
questions.

Usage
-----
    cd backend

    # Sample run (200 questions, both models):
    python eval/gpt4_baseline.py \\
        --dataset-file path/to/medqa/test.jsonl

    # Full test split:
    python eval/gpt4_baseline.py \\
        --dataset-file path/to/medqa/test.jsonl \\
        --full

    # Custom output path:
    python eval/gpt4_baseline.py \\
        --dataset-file path/to/medqa/test.jsonl \\
        --output results/my_gpt4_outputs.csv

Environment
-----------
    Set OPENAI_API_KEY in the environment or in a .env file (see .env.example).

References
----------
Jin, D. et al. (2021). What disease does this patient have? Applied Sciences.
OpenAI. (2024). GPT-4 Technical Report. https://openai.com/research/gpt-4
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a medical expert. Answer the following USMLE-style question "
    "by selecting the single best answer. Respond with ONLY the letter "
    "of the correct answer (A, B, C, or D). No explanation."
)

_USER_PROMPT_TEMPLATE = (
    "{question_stem}\n"
    "A. {option_A}\n"
    "B. {option_B}\n"
    "C. {option_C}\n"
    "D. {option_D}"
)

_MODELS = [
    "gpt-4-0125-preview",
    "gpt-4o-mini-2024-07-18",
]

_SAMPLE_SIZE = 200
_SEED = 42

_CSV_FIELDNAMES = [
    "question_id",
    "model",
    "predicted_letter",
    "correct_letter",
    "is_correct",
    "confidence_score",
    # Also included for compatibility with GPT4BaselineInterface:
    "predicted_condition",
]


# ---------------------------------------------------------------------------
# MedQA JSONL loader (mirrors MedQALoader in benchmark_runner.py)
# ---------------------------------------------------------------------------


def load_medqa(path: str) -> list[dict]:
    """
    Load the MedQA JSONL test split.

    Expected format per line:
        {
          "question": str,
          "options": {"A": str, "B": str, "C": str, "D": str},
          "answer": str,
          "answer_idx": str,
          "metamap_phrases": [str, ...]
        }

    Returns a list of normalised dicts with keys:
        id, stem, options_dict, correct_letter, correct_text
    """
    questions: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"WARNING: skipping malformed JSON at line {line_num}: {exc}",
                      file=sys.stderr)
                continue

            options: dict[str, str] = obj.get("options", {})
            # Ensure we have exactly four options A-D; skip if not
            if not all(k in options for k in ("A", "B", "C", "D")):
                continue

            answer_idx: str = (obj.get("answer_idx") or obj.get("answer", "A")).strip().upper()
            if answer_idx not in options:
                # Fall back: try to find the letter by matching answer text
                answer_text = obj.get("answer", "")
                answer_idx = next(
                    (k for k, v in options.items() if v == answer_text),
                    "A",
                )

            questions.append({
                "id": obj.get("id", f"medqa-{line_num}"),
                "stem": obj.get("question", ""),
                "options_dict": options,
                "correct_letter": answer_idx,
                "correct_text": options[answer_idx],
            })

    return questions


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------


def stratified_sample(questions: list[dict], n: int, seed: int = _SEED) -> list[dict]:
    """
    Draw a stratified random sample of *n* questions.

    Stratification is on the correct answer letter (A/B/C/D) so each option
    is represented proportionally — this avoids accidentally over/under-sampling
    a particular answer key which can bias Acc@1 estimates.
    """
    if len(questions) <= n:
        return questions

    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = {}
    for q in questions:
        letter = q["correct_letter"]
        buckets.setdefault(letter, []).append(q)

    # Shuffle each bucket deterministically
    for letter in buckets:
        rng.shuffle(buckets[letter])

    # Draw proportionally from each bucket
    total = len(questions)
    sampled: list[dict] = []
    letters = sorted(buckets)
    allocated = 0
    for i, letter in enumerate(letters):
        if i < len(letters) - 1:
            quota = round(n * len(buckets[letter]) / total)
        else:
            quota = n - allocated  # remainder goes to last bucket
        sampled.extend(buckets[letter][:quota])
        allocated += quota

    # Top up or trim if rounding pushed us off target
    if len(sampled) < n:
        remaining = [q for q in questions if q not in sampled]
        rng.shuffle(remaining)
        sampled.extend(remaining[: n - len(sampled)])
    sampled = sampled[:n]

    rng.shuffle(sampled)
    return sampled


# ---------------------------------------------------------------------------
# OpenAI call
# ---------------------------------------------------------------------------


def call_openai(
    client,
    model: str,
    question: dict,
    max_retries: int = 5,
    retry_delay: float = 10.0,
) -> tuple[str, float]:
    """
    Call the OpenAI ChatCompletion API for a single MedQA question.

    Returns
    -------
    predicted_letter : str
        Extracted letter (A/B/C/D) or "?" if parsing fails.
    confidence_score : float
        1.0 for greedy decoding (temperature=0); no token-level probs requested.
    """
    options = question["options_dict"]
    user_msg = _USER_PROMPT_TEMPLATE.format(
        question_stem=question["stem"],
        option_A=options["A"],
        option_B=options["B"],
        option_C=options["C"],
        option_D=options["D"],
    )

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                seed=42,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=4,  # We only need a single letter
            )
            raw = response.choices[0].message.content.strip().upper()
            # Extract first A/B/C/D character found
            predicted = next((ch for ch in raw if ch in ("A", "B", "C", "D")), "?")
            return predicted, 1.0

        except Exception as exc:  # noqa: BLE001
            err = str(exc)
            if attempt < max_retries:
                wait = retry_delay * attempt
                print(
                    f"  API error on attempt {attempt}/{max_retries}: {err}. "
                    f"Retrying in {wait:.0f}s ...",
                    file=sys.stderr,
                )
                time.sleep(wait)
            else:
                print(f"  API error after {max_retries} attempts: {err}", file=sys.stderr)
                return "?", 1.0


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def run_baseline(
    questions: list[dict],
    model: str,
    client,
) -> list[dict]:
    """
    Evaluate *model* on *questions*.

    Returns a list of result dicts (one per question).
    """
    results: list[dict] = []
    n = len(questions)
    correct = 0

    for i, q in enumerate(questions, 1):
        predicted_letter, conf = call_openai(client, model, q)
        is_correct = int(predicted_letter == q["correct_letter"])
        correct += is_correct

        results.append({
            "question_id": q["id"],
            "model": model,
            "predicted_letter": predicted_letter,
            "correct_letter": q["correct_letter"],
            "is_correct": is_correct,
            "confidence_score": conf,
            # Alias for GPT4BaselineInterface compatibility
            "predicted_condition": predicted_letter,
        })

        if i % 10 == 0 or i == n:
            running_acc = correct / i
            print(f"  [{model}] {i:>4}/{n}  running Acc@1 = {running_acc:.3f}")

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPT-4 / GPT-4o-mini zero-shot baseline for MedQA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset-file",
        required=True,
        help="Path to MedQA JSONL test split.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results",
            "gpt4_baseline_outputs.csv",
        ),
        help="Output CSV path (default: results/gpt4_baseline_outputs.csv).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run on the entire test split instead of a 200-question sample.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=_MODELS,
        metavar="MODEL",
        help=f"OpenAI model names to evaluate (default: {_MODELS}).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=_SAMPLE_SIZE,
        help=f"Number of questions to sample (default: {_SAMPLE_SIZE}). "
             "Ignored when --full is set.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # OpenAI client setup
    # ------------------------------------------------------------------
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "ERROR: OPENAI_API_KEY environment variable is not set.\n"
            "Set it in your shell or in a .env file (see .env.example).",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        from openai import OpenAI  # noqa: PLC0415
    except ImportError:
        print(
            "ERROR: openai package is not installed.\n"
            "Install it with: pip install openai>=1.0",
            file=sys.stderr,
        )
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # ------------------------------------------------------------------
    # Load and (optionally) sample questions
    # ------------------------------------------------------------------
    print(f"Loading MedQA test split from {args.dataset_file} ...")
    all_questions = load_medqa(args.dataset_file)
    print(f"  Loaded {len(all_questions)} questions.")

    if args.full:
        questions = all_questions
        print(f"  --full flag set: evaluating all {len(questions)} questions.")
    else:
        questions = stratified_sample(all_questions, args.sample_size)
        print(
            f"  Using stratified random sample of {len(questions)} questions "
            f"(seed={_SEED})."
        )

    # ------------------------------------------------------------------
    # Run models
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    all_results: list[dict] = []
    summary: list[dict] = []  # per-model aggregate stats

    for model in args.models:
        print(f"\nEvaluating model: {model}")
        model_results = run_baseline(questions, model, client)
        all_results.extend(model_results)

        n = len(model_results)
        acc = sum(r["is_correct"] for r in model_results) / n if n else 0.0
        summary.append({"model": model, "acc": acc, "n": n})

    # ------------------------------------------------------------------
    # Write CSV
    # ------------------------------------------------------------------
    with open(args.output, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nOutputs written to {args.output}")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    col_w = max(len(r["model"]) for r in summary)
    header = f"{'Model':<{col_w}} | {'Acc@1':>6} | {'N':>6}"
    print("\n" + header)
    print("-" * len(header))
    for row in summary:
        print(f"{row['model']:<{col_w}} | {row['acc']:>6.3f} | {row['n']:>6}")


if __name__ == "__main__":
    main()
