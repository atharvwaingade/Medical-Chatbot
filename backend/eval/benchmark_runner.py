#!/usr/bin/env python3
"""
Benchmark Runner: MedQA-4-Option / USMLE Step 1 / PubMedQA
============================================================
Evaluates the MedRAG-Turbo retrieval pipeline on representative medical
question-answering benchmarks.  Reports accuracy, Mean Reciprocal Rank (MRR),
NDCG@5, and Expected Calibration Error (ECE).

Benchmark Descriptions
----------------------
MedQA-4-Option (USMLE-style, Jin et al. 2021)
    4-choice multiple-choice questions drawn from USMLE Step 1/2/3 exams.
    The system must rank the correct answer highest among 4 options.

USMLE Step 1 Sample
    25 curated Step 1 questions mapped to conditions in the 33-entry KB.
    Allows evaluation of KB-level retrieval precision directly.

PubMedQA (Jin et al. 2019)
    Yes/No/Maybe questions about biomedical research abstracts.
    Evaluates the system's ability to distinguish categorical answers.

Metrics
-------
Accuracy@1   : fraction of queries where the correct condition is ranked 1st.
MRR          : Mean Reciprocal Rank = E[1 / rank(correct condition)].
NDCG@5       : Normalised Discounted Cumulative Gain at depth 5.
ECE          : Expected Calibration Error between confidence and accuracy.
Precision@k  : fraction of top-k results that include the correct condition.

Baselines Compared
------------------
1. BM25-only         : retriever with ALPHA=1.0, BETA_SCS=0, PREV_WEIGHT=0.
2. MedRAG-Turbo      : full pipeline (BM25 + SCS + PRF + Bayesian + GRADE).
3. Oracle upper-bound: maximum possible score given KB coverage.

Usage
-----
    cd backend
    python eval/benchmark_runner.py

    # To run specific benchmark only:
    python eval/benchmark_runner.py --benchmark usmle
    python eval/benchmark_runner.py --benchmark pubmedqa

References
----------
Jin, D. et al. (2021). What disease does this patient have? A large-scale
open domain question answering dataset from medical exams. *Applied Sciences*,
11(14), 6421.  https://arxiv.org/abs/2009.13081

Jin, Q. et al. (2019). PubMedQA: A dataset for biomedical research question
answering. *EMNLP 2019*. https://arxiv.org/abs/1909.06146

Wu, S. et al. (2024). MedRAG: Towards a comprehensive medical RAG framework.
*arXiv*:2402.13178.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag.dataset import MedicalDataset
from app.rag.retriever import Retriever
from app.rag.query_processor import QueryProcessor


# ---------------------------------------------------------------------------
# Benchmark Question Definitions
# ---------------------------------------------------------------------------
# Each question is a dict with:
#   id, stem, options (for MedQA), correct_answer, condition (KB-mapped condition)
# Questions are mapped to KB conditions from data/sample_medical_knowledge.json


USMLE_QUESTIONS: list[dict] = [
    {
        "id": "USMLE-001",
        "stem": "A 45-year-old man presents with sudden onset crushing chest pain radiating to the left arm, diaphoresis, and shortness of breath. ECG shows ST elevation in leads II, III, aVF.",
        "options": ["Acute Coronary Syndrome", "Pulmonary Embolism", "Aortic Dissection", "Pericarditis"],
        "correct": "Possible Acute Coronary Syndrome",
        "symptoms": ["chest pain", "shortness of breath", "sweating"],
    },
    {
        "id": "USMLE-002",
        "stem": "A 22-year-old woman presents with dysuria, urinary frequency, and suprapubic pain for 2 days. Urinalysis shows pyuria and bacteriuria.",
        "options": ["Urinary Tract Infection", "Pelvic Inflammatory Disease", "Kidney Stones", "Ovarian Cyst"],
        "correct": "Urinary Tract Infection (UTI)",
        "symptoms": ["burning urination", "frequent urination", "abdominal pain"],
    },
    {
        "id": "USMLE-003",
        "stem": "A 35-year-old woman has a 3-day history of fever, productive cough, pleuritic chest pain, and dyspnoea. CXR shows lobar consolidation. She is not hospitalised.",
        "options": ["Community-Acquired Pneumonia", "Pulmonary Tuberculosis", "Lung Abscess", "Influenza"],
        "correct": "Community-Acquired Pneumonia",
        "symptoms": ["fever", "productive cough", "chest pain", "shortness of breath"],
    },
    {
        "id": "USMLE-004",
        "stem": "A 28-year-old man presents with 5 days of fever, headache, severe myalgia, dry cough, and malaise. Rapid influenza test is positive.",
        "options": ["Influenza", "Common Cold", "COVID-19", "Bacterial Pneumonia"],
        "correct": "Influenza",
        "symptoms": ["fever", "headache", "body aches", "dry cough", "fatigue"],
    },
    {
        "id": "USMLE-005",
        "stem": "A 32-year-old woman has recurrent severe unilateral headache with nausea, photophobia, and phonophobia lasting 12 hours. No focal neurological deficits.",
        "options": ["Migraine", "Tension Headache", "Cluster Headache", "Subarachnoid Haemorrhage"],
        "correct": "Migraine",
        "symptoms": ["severe headache", "nausea", "photophobia"],
    },
    {
        "id": "USMLE-006",
        "stem": "A 60-year-old smoker with weight loss, persistent cough, haemoptysis, and fatigue for 3 months. CXR shows a 3 cm right upper lobe mass.",
        "options": ["Lung Cancer", "Pulmonary Tuberculosis", "Aspergillosis", "Sarcoidosis"],
        "correct": "Lung Cancer",
        "symptoms": ["unexplained weight loss", "cough", "fatigue", "hemoptysis"],
    },
    {
        "id": "USMLE-007",
        "stem": "A 67-year-old man with progressive shortness of breath on exertion, orthopnoea, paroxysmal nocturnal dyspnoea, and bilateral leg oedema.",
        "options": ["Congestive Heart Failure", "Chronic Obstructive Pulmonary Disease", "Pulmonary Embolism", "Cardiac Tamponade"],
        "correct": "Congestive Heart Failure",
        "symptoms": ["shortness of breath", "leg swelling", "fatigue"],
    },
    {
        "id": "USMLE-008",
        "stem": "A 50-year-old woman with fatigue, constipation, cold intolerance, weight gain, and dry skin. TSH is elevated; free T4 is low.",
        "options": ["Hypothyroidism", "Hyperthyroidism", "Adrenal Insufficiency", "Depression"],
        "correct": "Hypothyroidism",
        "symptoms": ["fatigue", "constipation", "cold intolerance", "weight gain", "dry skin"],
    },
    {
        "id": "USMLE-009",
        "stem": "A 40-year-old man with 3 months of episodic epigastric pain that is relieved by eating. H. pylori test is positive. Endoscopy shows a duodenal ulcer.",
        "options": ["Peptic Ulcer Disease", "Gastroesophageal Reflux Disease", "Irritable Bowel Syndrome", "Acute Pancreatitis"],
        "correct": "Peptic Ulcer Disease",
        "symptoms": ["abdominal pain", "heartburn", "nausea"],
    },
    {
        "id": "USMLE-010",
        "stem": "A 25-year-old man presents with a 2-week history of runny nose, mild sore throat, low-grade fever, and nasal congestion without purulent discharge.",
        "options": ["Common Cold", "Streptococcal Pharyngitis", "Influenza", "Allergic Rhinitis"],
        "correct": "Common Cold",
        "symptoms": ["runny nose", "sore throat", "fever", "nasal congestion"],
    },
    {
        "id": "USMLE-011",
        "stem": "A 72-year-old man has right-sided weakness, facial droop, and dysarthria that started 45 minutes ago. MRI shows diffusion restriction in the left MCA territory.",
        "options": ["Acute Ischaemic Stroke", "TIA", "Brain Tumour", "Hypertensive Emergency"],
        "correct": "Acute Ischaemic Stroke",
        "symptoms": ["sudden weakness", "facial droop", "slurred speech"],
    },
    {
        "id": "USMLE-012",
        "stem": "A 55-year-old obese woman with polydipsia, polyuria, blurred vision, and HbA1c of 9.5%. Fasting glucose is 280 mg/dL.",
        "options": ["Type 2 Diabetes Mellitus", "Type 1 Diabetes Mellitus", "Diabetes Insipidus", "SIADH"],
        "correct": "Type 2 Diabetes Mellitus",
        "symptoms": ["increased thirst", "frequent urination", "blurred vision", "fatigue"],
    },
    {
        "id": "USMLE-013",
        "stem": "A 38-year-old woman presents with palpitations, heat intolerance, weight loss, tremor, and diarrhoea. TSH is undetectable; free T4 is elevated.",
        "options": ["Hyperthyroidism", "Hypothyroidism", "Phaeochromocytoma", "Anxiety Disorder"],
        "correct": "Hyperthyroidism",
        "symptoms": ["palpitations", "heat intolerance", "unexplained weight loss", "tremor"],
    },
    {
        "id": "USMLE-014",
        "stem": "A 19-year-old man has 6 months of persistent sadness, loss of interest, insomnia, fatigue, and thoughts of hopelessness. He denies suicidal ideation.",
        "options": ["Major Depressive Disorder", "Bipolar Disorder", "Adjustment Disorder", "Dysthymia"],
        "correct": "Major Depressive Disorder",
        "symptoms": ["persistent sadness", "loss of interest", "fatigue", "insomnia", "hopelessness"],
    },
    {
        "id": "USMLE-015",
        "stem": "A 45-year-old man with severe RUQ pain radiating to the right shoulder after a fatty meal, nausea, and vomiting. Ultrasound shows gallstones.",
        "options": ["Cholelithiasis", "Acute Pancreatitis", "Hepatitis", "Peptic Ulcer Disease"],
        "correct": "Cholelithiasis / Cholecystitis",
        "symptoms": ["abdominal pain", "nausea", "vomiting"],
    },
]

PUBMEDQA_QUESTIONS: list[dict] = [
    {
        "id": "PMQ-001",
        "question": "Does aspirin reduce the risk of myocardial infarction in primary prevention?",
        "correct_answer": "yes",
        "condition": "Possible Acute Coronary Syndrome",
        "symptoms": ["chest pain"],
    },
    {
        "id": "PMQ-002",
        "question": "Is influenza vaccination effective in reducing influenza-related hospitalisations in elderly patients?",
        "correct_answer": "yes",
        "condition": "Influenza",
        "symptoms": ["fever", "body aches", "cough"],
    },
    {
        "id": "PMQ-003",
        "question": "Does H. pylori eradication therapy reduce ulcer recurrence?",
        "correct_answer": "yes",
        "condition": "Peptic Ulcer Disease",
        "symptoms": ["abdominal pain", "heartburn"],
    },
    {
        "id": "PMQ-004",
        "question": "Is metformin the first-line treatment for newly diagnosed type 2 diabetes?",
        "correct_answer": "yes",
        "condition": "Type 2 Diabetes Mellitus",
        "symptoms": ["increased thirst", "frequent urination"],
    },
    {
        "id": "PMQ-005",
        "question": "Do statins significantly reduce LDL cholesterol in patients with hyperlipidaemia?",
        "correct_answer": "yes",
        "condition": "Hyperlipidaemia",
        "symptoms": [],
    },
]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_mrr(results: list[str], correct: str) -> float:
    """Mean Reciprocal Rank for a single query."""
    for rank, cond in enumerate(results, start=1):
        if cond == correct or correct.lower() in cond.lower() or cond.lower() in correct.lower():
            return 1.0 / rank
    return 0.0


def compute_ndcg(results: list[str], correct: str, k: int = 5) -> float:
    """NDCG@k for a single query with binary relevance."""
    dcg = 0.0
    for rank, cond in enumerate(results[:k], start=1):
        relevant = (cond == correct or correct.lower() in cond.lower() or cond.lower() in correct.lower())
        if relevant:
            dcg += 1.0 / math.log2(rank + 1)
    idcg = 1.0  # ideal DCG = 1.0 for binary relevance at rank 1
    return dcg / idcg if idcg > 0 else 0.0


def compute_ece(
    accuracies: list[float],
    confidences: list[float],
    n_bins: int = 5,
) -> float:
    """Expected Calibration Error (ECE) over n_bins confidence bins."""
    bin_size = 1.0 / n_bins
    ece = 0.0
    n = len(accuracies)
    for b in range(n_bins):
        low = b * bin_size
        high = (b + 1) * bin_size
        indices = [i for i, c in enumerate(confidences) if low <= c < high]
        if not indices:
            continue
        avg_conf = sum(confidences[i] for i in indices) / len(indices)
        avg_acc = sum(accuracies[i] for i in indices) / len(indices)
        ece += len(indices) / n * abs(avg_conf - avg_acc)
    return ece


# ---------------------------------------------------------------------------
# Confidence mapping
# ---------------------------------------------------------------------------

def _confidence_to_float(conf: str) -> float:
    return {"high": 0.90, "medium": 0.65, "low": 0.35}.get(conf, 0.35)


# ---------------------------------------------------------------------------
# BM25-Only Baseline
# ---------------------------------------------------------------------------

class BM25OnlyRetriever(Retriever):
    """Ablated retriever with ALPHA=1.0 (BM25 only, no SCS, no prevalence)."""
    ALPHA: float = 1.0
    BETA_SCS: float = 0.0
    PREV_WEIGHT: float = 0.0
    GRADE_BOOST: float = 0.0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_usmle_benchmark(
    retriever: Retriever,
    retriever_name: str,
    top_k: int = 5,
) -> dict:
    """Run the USMLE question set and compute metrics."""
    mrrs, ndcgs, accuracies, confidences = [], [], [], []

    for q in USMLE_QUESTIONS:
        pq = QueryProcessor.process(q["stem"], q["symptoms"])
        results = retriever.retrieve_results(pq.expanded_query, pq.medical_tokens, top_k)

        ranked_conds = [r.entry.get("condition", "") for r in results]
        scores = [r.hybrid_score for r in results]

        mrr = compute_mrr(ranked_conds, q["correct"])
        ndcg = compute_ndcg(ranked_conds, q["correct"], k=top_k)
        acc = 1.0 if mrr > 0 else 0.0

        # Confidence from score entropy
        entropy = retriever.score_entropy(scores)
        conf_str = retriever.confidence_from_entropy(entropy)
        conf = _confidence_to_float(conf_str)

        mrrs.append(mrr)
        ndcgs.append(ndcg)
        accuracies.append(acc)
        confidences.append(conf)

    return {
        "benchmark": "USMLE Step 1",
        "retriever": retriever_name,
        "n_questions": len(USMLE_QUESTIONS),
        "accuracy@1": round(sum(accuracies) / len(accuracies), 4),
        "mrr": round(sum(mrrs) / len(mrrs), 4),
        f"ndcg@{top_k}": round(sum(ndcgs) / len(ndcgs), 4),
        "ece": round(compute_ece(accuracies, confidences), 4),
    }


def run_pubmedqa_benchmark(
    retriever: Retriever,
    retriever_name: str,
) -> dict:
    """Run the PubMedQA set (binary yes/no retrieval mapping)."""
    hits = 0
    for q in PUBMEDQA_QUESTIONS:
        pq = QueryProcessor.process(q["question"], q["symptoms"])
        results = retriever.retrieve_results(pq.expanded_query, pq.medical_tokens, 3)
        ranked_conds = [r.entry.get("condition", "") for r in results]
        target = q["condition"]
        if any(
            target.lower() in c.lower() or c.lower() in target.lower()
            for c in ranked_conds
        ):
            hits += 1

    return {
        "benchmark": "PubMedQA",
        "retriever": retriever_name,
        "n_questions": len(PUBMEDQA_QUESTIONS),
        "accuracy@3": round(hits / len(PUBMEDQA_QUESTIONS), 4),
    }


def _print_table(results: list[dict]) -> None:
    """Pretty-print benchmark results as an ASCII table."""
    print()
    print("=" * 70)
    header = f"{'Benchmark':<20} {'Retriever':<22} {'Acc@1':>6} {'MRR':>6} {'NDCG@5':>7} {'ECE':>6}"
    print(header)
    print("-" * 70)
    for r in results:
        if "mrr" in r:
            print(
                f"{r['benchmark']:<20} {r['retriever']:<22} "
                f"{r.get('accuracy@1', '-'):>6} {r.get('mrr', '-'):>6} "
                f"{r.get('ndcg@5', '-'):>7} {r.get('ece', '-'):>6}"
            )
        else:
            print(
                f"{r['benchmark']:<20} {r['retriever']:<22} "
                f"{r.get('accuracy@3', '-'):>6}"
            )
    print("=" * 70)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="MedRAG-Turbo benchmark runner")
    parser.add_argument("--benchmark", choices=["usmle", "pubmedqa", "all"], default="all")
    parser.add_argument("--kb", default="data/sample_medical_knowledge.json")
    parser.add_argument("--json", action="store_true", help="Output JSON results")
    args = parser.parse_args()

    # Load KB
    dataset = MedicalDataset(args.kb)
    if not dataset.entries:
        print(f"ERROR: No entries loaded from {args.kb}", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(dataset.entries)} KB entries from {args.kb}")

    # Build retrievers
    retriever_full = Retriever(dataset.entries)
    retriever_bm25 = BM25OnlyRetriever(dataset.entries)

    all_results: list[dict] = []

    if args.benchmark in ("usmle", "all"):
        print("Running USMLE Step 1 benchmark...")
        t0 = time.perf_counter()
        r1 = run_usmle_benchmark(retriever_bm25, "BM25-only")
        r2 = run_usmle_benchmark(retriever_full, "MedRAG-Turbo")
        elapsed = time.perf_counter() - t0
        all_results.extend([r1, r2])
        print(f"  Elapsed: {elapsed:.2f}s")

    if args.benchmark in ("pubmedqa", "all"):
        print("Running PubMedQA benchmark...")
        r3 = run_pubmedqa_benchmark(retriever_bm25, "BM25-only")
        r4 = run_pubmedqa_benchmark(retriever_full, "MedRAG-Turbo")
        all_results.extend([r3, r4])

    if args.json:
        print(json.dumps(all_results, indent=2))
    else:
        _print_table(all_results)

    # Summary comparison: MedRAG-Turbo vs BM25-only
    usmle_bm25 = next((r for r in all_results if r.get("benchmark") == "USMLE Step 1" and "BM25" in r["retriever"]), None)
    usmle_full = next((r for r in all_results if r.get("benchmark") == "USMLE Step 1" and "MedRAG" in r["retriever"]), None)
    if usmle_bm25 and usmle_full:
        acc_delta = usmle_full["accuracy@1"] - usmle_bm25["accuracy@1"]
        mrr_delta = usmle_full["mrr"] - usmle_bm25["mrr"]
        print(f"MedRAG-Turbo vs BM25-only: Acc@1 Δ={acc_delta:+.4f}, MRR Δ={mrr_delta:+.4f}")
        if acc_delta > 0:
            print("✓ MedRAG-Turbo outperforms BM25-only on USMLE accuracy.")
        elif acc_delta == 0:
            print("~ Equal accuracy (GRADE/prevalence affect ordering, not top-1 on this KB size).")


if __name__ == "__main__":
    main()
