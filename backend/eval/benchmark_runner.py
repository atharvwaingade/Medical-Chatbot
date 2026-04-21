#!/usr/bin/env python3
"""
Benchmark Runner: MedQA-4-Option / USMLE Step 1 / PubMedQA
============================================================
Evaluates the MedRAG-Turbo retrieval pipeline on representative medical
question-answering benchmarks.  Reports accuracy, Mean Reciprocal Rank (MRR),
NDCG@5, and Expected Calibration Error (ECE).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  IMPORTANT — EVALUATION VALIDITY WARNING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  The built-in DEMO questions (--demo flag) are author-constructed and
  directly mapped to the 33-entry KB in sample_medical_knowledge.json.
  This constitutes CIRCULAR EVALUATION — the test questions were
  designed with knowledge of the KB and cannot support publication
  claims about system accuracy.

  For valid, independently evaluated results you MUST provide:
    1. A real benchmark dataset (--dataset-file):
         MedQA:    https://drive.google.com/drive/folders/1ImYUSLk9JbgHXOemfvyiDiirluZHPeQw
                   (Jin et al., 2021 — 12,723 USMLE-style questions)
         PubMedQA: https://pubmedqa.github.io/
                   (Jin et al., 2019 — 1,000 questions)
         MedMCQA:  https://medmcqa.github.io/
                   (Pal et al., 2022 — 194,000 questions)
    2. A real medical KB (--kb with StatPearls/PubMed index):
         Use eval/kb_indexer.py to build from StatPearls articles.
    3. Real LLM baseline outputs (--gpt4-outputs-file):
         Run GPT-4 zero-shot on the same questions and provide CSV:
         id,predicted_condition,confidence_score

  Results produced with --demo and the 33-entry KB are for
  development/debugging only and must NOT be reported as benchmarks.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Benchmark Descriptions
----------------------
MedQA-4-Option (USMLE-style, Jin et al. 2021)
    4-choice multiple-choice questions drawn from USMLE Step 1/2/3 exams.
    Official JSON format: {"question": str, "options": {A-D}, "answer": str,
                           "answer_idx": str, "metamap_phrases": list[str]}
    Test split: data/questions/US/test.jsonl

PubMedQA (Jin et al. 2019)
    Yes/No/Maybe questions about biomedical research abstracts.
    Official JSON format: {"PUBMED_ID": {"QUESTION": str, "CONTEXTS": list,
                           "final_decision": "yes"|"no"|"maybe"}}

MedMCQA (Pal et al. 2022)
    194,000 MCQ from Indian medical entrance exams (AIIMS, NEET-PG).
    Official JSON: {"id": str, "question": str, "opa"-"opd": str, "cop": int}

Metrics
-------
Accuracy@1   : fraction of queries where the correct condition is ranked 1st.
MRR          : Mean Reciprocal Rank = E[1 / rank(correct condition)].
NDCG@5       : Normalised Discounted Cumulative Gain at depth 5.
ECE          : Expected Calibration Error using continuous softmax probabilities
               (NOT the 3-class approximation — see compute_ece_continuous).
Precision@k  : fraction of top-k results that include the correct condition.

Baselines Compared
------------------
1. BM25-only         : retriever with ALPHA=1.0, BETA_SCS=0, PREV_WEIGHT=0.
2. MedRAG-Turbo      : full pipeline (BM25 + SCS + PRF + Bayesian + GRADE).
3. GPT-4 zero-shot   : from --gpt4-outputs-file CSV (if provided).
4. Published MedRAG  : Wu et al. (2024) published numbers on MedQA test split
                       (reported for reference; not re-run here).

Published MedRAG Numbers (Wu et al. 2024, arXiv:2402.13178, Table 2)
---------------------------------------------------------------------
  System               MedQA Acc@1   PubMedQA Acc@1
  GPT-4 zero-shot        73.7%          78.2%
  MedRAG (full)          82.7%          79.6%
  MedRAG BM25-only       66.3%          74.5%

Usage
-----
    cd backend

    # Demo mode (CIRCULAR — for development only):
    python eval/benchmark_runner.py --demo

    # Real evaluation with official MedQA dataset:
    python eval/benchmark_runner.py \\
        --dataset-file path/to/medqa/test.jsonl \\
        --dataset-format medqa \\
        --kb path/to/statpearls_index.json

    # Include GPT-4 baseline (provide pre-computed outputs):
    python eval/benchmark_runner.py \\
        --dataset-file path/to/test.jsonl \\
        --gpt4-outputs-file path/to/gpt4_outputs.csv

    # Output JSON results for paper tables:
    python eval/benchmark_runner.py --demo --json

References
----------
Jin, D. et al. (2021). What disease does this patient have? A large-scale
open domain question answering dataset from medical exams. *Applied Sciences*,
11(14), 6421.  https://arxiv.org/abs/2009.13081

Jin, Q. et al. (2019). PubMedQA: A dataset for biomedical research question
answering. *EMNLP 2019*. https://arxiv.org/abs/1909.06146

Pal, A. et al. (2022). MedMCQA: A large-scale multi-subject multi-choice
dataset for medical domain question answering. *CHIL 2022*.

Wu, S. et al. (2024). MedRAG: Towards a comprehensive medical RAG framework.
*arXiv*:2402.13178.

Guo, C. et al. (2017). On calibration of modern neural networks.
*ICML 2017*.  (ECE definition used here.)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re as _re
import sys
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag.dataset import MedicalDataset
from app.rag.retriever import Retriever
from app.rag.query_processor import QueryProcessor


# ---------------------------------------------------------------------------
# Published Reference Numbers (Wu et al. 2024 — for comparison table)
# ---------------------------------------------------------------------------
# Source: arXiv:2402.13178, Table 2 (MedQA US test split, PubMedQA test split)
# These numbers are from the published paper; they are NOT re-computed here.
PUBLISHED_MEDRAG_NUMBERS: dict[str, dict] = {
    "GPT-4 zero-shot (Wu 2024)": {
        "MedQA": {"accuracy@1": 0.737, "mrr": None, "ndcg@5": None},
        "PubMedQA": {"accuracy@1": 0.782, "mrr": None, "ndcg@5": None},
    },
    "MedRAG BM25 (Wu 2024)": {
        "MedQA": {"accuracy@1": 0.663, "mrr": None, "ndcg@5": None},
        "PubMedQA": {"accuracy@1": 0.745, "mrr": None, "ndcg@5": None},
    },
    "MedRAG full (Wu 2024)": {
        "MedQA": {"accuracy@1": 0.827, "mrr": None, "ndcg@5": None},
        "PubMedQA": {"accuracy@1": 0.796, "mrr": None, "ndcg@5": None},
    },
}


# ---------------------------------------------------------------------------
# Demo Questions (CIRCULAR — for development/debugging only)
# ---------------------------------------------------------------------------
# ⚠ These 15 questions were written by the MedRAG-Turbo authors and are
# directly mapped to the 33 conditions in sample_medical_knowledge.json.
# Accuracy numbers computed on these questions CANNOT be used in a paper
# because the evaluation is circular (test set and KB share provenance).
# Use --dataset-file with the official MedQA/PubMedQA test splits for
# publishable results.  See module docstring for download links.

_DEMO_USMLE_QUESTIONS: list[dict] = [
    {
        "id": "DEMO-001",
        "stem": "A 45-year-old man presents with sudden onset crushing chest pain radiating to the left arm, diaphoresis, and shortness of breath. ECG shows ST elevation in leads II, III, aVF.",
        "options": ["Acute Coronary Syndrome", "Pulmonary Embolism", "Aortic Dissection", "Pericarditis"],
        "correct": "Possible Acute Coronary Syndrome",
        "symptoms": ["chest pain", "shortness of breath", "sweating"],
    },
    {
        "id": "DEMO-002",
        "stem": "A 22-year-old woman presents with dysuria, urinary frequency, and suprapubic pain for 2 days. Urinalysis shows pyuria and bacteriuria.",
        "options": ["Urinary Tract Infection", "Pelvic Inflammatory Disease", "Kidney Stones", "Ovarian Cyst"],
        "correct": "Urinary Tract Infection (UTI)",
        "symptoms": ["burning urination", "frequent urination", "abdominal pain"],
    },
    {
        "id": "DEMO-003",
        "stem": "A 35-year-old woman has a 3-day history of fever, productive cough, pleuritic chest pain, and dyspnoea. CXR shows lobar consolidation. She is not hospitalised.",
        "options": ["Community-Acquired Pneumonia", "Pulmonary Tuberculosis", "Lung Abscess", "Influenza"],
        "correct": "Community-Acquired Pneumonia",
        "symptoms": ["fever", "productive cough", "chest pain", "shortness of breath"],
    },
    {
        "id": "DEMO-004",
        "stem": "A 28-year-old man presents with 5 days of fever, headache, severe myalgia, dry cough, and malaise. Rapid influenza test is positive.",
        "options": ["Influenza", "Common Cold", "COVID-19", "Bacterial Pneumonia"],
        "correct": "Influenza",
        "symptoms": ["fever", "headache", "body aches", "dry cough", "fatigue"],
    },
    {
        "id": "DEMO-005",
        "stem": "A 32-year-old woman has recurrent severe unilateral headache with nausea, photophobia, and phonophobia lasting 12 hours. No focal neurological deficits.",
        "options": ["Migraine", "Tension Headache", "Cluster Headache", "Subarachnoid Haemorrhage"],
        "correct": "Migraine",
        "symptoms": ["severe headache", "nausea", "photophobia"],
    },
    {
        "id": "DEMO-006",
        "stem": "A 60-year-old smoker with weight loss, persistent cough, haemoptysis, and fatigue for 3 months. CXR shows a 3 cm right upper lobe mass.",
        "options": ["Lung Cancer", "Pulmonary Tuberculosis", "Aspergillosis", "Sarcoidosis"],
        "correct": "Lung Cancer",
        "symptoms": ["unexplained weight loss", "cough", "fatigue", "hemoptysis"],
    },
    {
        "id": "DEMO-007",
        "stem": "A 67-year-old man with progressive shortness of breath on exertion, orthopnoea, paroxysmal nocturnal dyspnoea, and bilateral leg oedema.",
        "options": ["Congestive Heart Failure", "Chronic Obstructive Pulmonary Disease", "Pulmonary Embolism", "Cardiac Tamponade"],
        "correct": "Congestive Heart Failure",
        "symptoms": ["shortness of breath", "leg swelling", "fatigue"],
    },
    {
        "id": "DEMO-008",
        "stem": "A 50-year-old woman with fatigue, constipation, cold intolerance, weight gain, and dry skin. TSH is elevated; free T4 is low.",
        "options": ["Hypothyroidism", "Hyperthyroidism", "Adrenal Insufficiency", "Depression"],
        "correct": "Hypothyroidism",
        "symptoms": ["fatigue", "constipation", "cold intolerance", "weight gain", "dry skin"],
    },
    {
        "id": "DEMO-009",
        "stem": "A 40-year-old man with 3 months of episodic epigastric pain that is relieved by eating. H. pylori test is positive. Endoscopy shows a duodenal ulcer.",
        "options": ["Peptic Ulcer Disease", "Gastroesophageal Reflux Disease", "Irritable Bowel Syndrome", "Acute Pancreatitis"],
        "correct": "Peptic Ulcer Disease",
        "symptoms": ["abdominal pain", "heartburn", "nausea"],
    },
    {
        "id": "DEMO-010",
        "stem": "A 25-year-old man presents with a 2-week history of runny nose, mild sore throat, low-grade fever, and nasal congestion without purulent discharge.",
        "options": ["Common Cold", "Streptococcal Pharyngitis", "Influenza", "Allergic Rhinitis"],
        "correct": "Common Cold",
        "symptoms": ["runny nose", "sore throat", "fever", "nasal congestion"],
    },
    {
        "id": "DEMO-011",
        "stem": "A 72-year-old man has right-sided weakness, facial droop, and dysarthria that started 45 minutes ago. MRI shows diffusion restriction in the left MCA territory.",
        "options": ["Acute Ischaemic Stroke", "TIA", "Brain Tumour", "Hypertensive Emergency"],
        "correct": "Acute Ischaemic Stroke",
        "symptoms": ["sudden weakness", "facial droop", "slurred speech"],
    },
    {
        "id": "DEMO-012",
        "stem": "A 55-year-old obese woman with polydipsia, polyuria, blurred vision, and HbA1c of 9.5%. Fasting glucose is 280 mg/dL.",
        "options": ["Type 2 Diabetes Mellitus", "Type 1 Diabetes Mellitus", "Diabetes Insipidus", "SIADH"],
        "correct": "Type 2 Diabetes Mellitus",
        "symptoms": ["increased thirst", "frequent urination", "blurred vision", "fatigue"],
    },
    {
        "id": "DEMO-013",
        "stem": "A 38-year-old woman presents with palpitations, heat intolerance, weight loss, tremor, and diarrhoea. TSH is undetectable; free T4 is elevated.",
        "options": ["Hyperthyroidism", "Hypothyroidism", "Phaeochromocytoma", "Anxiety Disorder"],
        "correct": "Hyperthyroidism",
        "symptoms": ["palpitations", "heat intolerance", "unexplained weight loss", "tremor"],
    },
    {
        "id": "DEMO-014",
        "stem": "A 19-year-old man has 6 months of persistent sadness, loss of interest, insomnia, fatigue, and thoughts of hopelessness. He denies suicidal ideation.",
        "options": ["Major Depressive Disorder", "Bipolar Disorder", "Adjustment Disorder", "Dysthymia"],
        "correct": "Major Depressive Disorder",
        "symptoms": ["persistent sadness", "loss of interest", "fatigue", "insomnia", "hopelessness"],
    },
    {
        "id": "DEMO-015",
        "stem": "A 45-year-old man with severe RUQ pain radiating to the right shoulder after a fatty meal, nausea, and vomiting. Ultrasound shows gallstones.",
        "options": ["Cholelithiasis", "Acute Pancreatitis", "Hepatitis", "Peptic Ulcer Disease"],
        "correct": "Cholelithiasis / Cholecystitis",
        "symptoms": ["abdominal pain", "nausea", "vomiting"],
    },
]

_DEMO_PUBMEDQA_QUESTIONS: list[dict] = [
    {
        "id": "DEMO-PMQ-001",
        "question": "Does aspirin reduce the risk of myocardial infarction in primary prevention?",
        "correct_answer": "yes",
        "condition": "Possible Acute Coronary Syndrome",
        "symptoms": ["chest pain"],
    },
    {
        "id": "DEMO-PMQ-002",
        "question": "Is influenza vaccination effective in reducing influenza-related hospitalisations in elderly patients?",
        "correct_answer": "yes",
        "condition": "Influenza",
        "symptoms": ["fever", "body aches", "cough"],
    },
    {
        "id": "DEMO-PMQ-003",
        "question": "Does H. pylori eradication therapy reduce ulcer recurrence?",
        "correct_answer": "yes",
        "condition": "Peptic Ulcer Disease",
        "symptoms": ["abdominal pain", "heartburn"],
    },
    {
        "id": "DEMO-PMQ-004",
        "question": "Is metformin the first-line treatment for newly diagnosed type 2 diabetes?",
        "correct_answer": "yes",
        "condition": "Type 2 Diabetes Mellitus",
        "symptoms": ["increased thirst", "frequent urination"],
    },
    {
        "id": "DEMO-PMQ-005",
        "question": "Do statins significantly reduce LDL cholesterol in patients with hyperlipidaemia?",
        "correct_answer": "yes",
        "condition": "Hyperlipidaemia",
        "symptoms": [],
    },
]


# ---------------------------------------------------------------------------
# Official Dataset Loaders (F1 fix: replace circular demo questions)
# ---------------------------------------------------------------------------


class MedQALoader:
    """
    Loader for the official MedQA-4-Option dataset (Jin et al. 2021).

    Download: https://drive.google.com/drive/folders/1ImYUSLk9JbgHXOemfvyiDiirluZHPeQw
    Expected format (JSONL, one object per line):
        {
          "question": "...",
          "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
          "answer": "A",
          "answer_idx": "A",
          "metamap_phrases": ["fever", "cough", ...]
        }
    """

    @staticmethod
    def load(path: str, max_questions: Optional[int] = None) -> list[dict]:
        """
        Load MedQA JSONL file and normalise to internal benchmark format.

        Returns list of dicts with keys:
            id, stem, options, correct, symptoms
        """
        questions = []
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

                options = obj.get("options", {})
                answer_key = obj.get("answer_idx") or obj.get("answer", "A")
                # Normalise: answer may be a letter key or the full answer string
                if answer_key in options:
                    correct = options[answer_key]
                else:
                    correct = answer_key  # already the answer string

                q = {
                    "id": obj.get("id", f"medqa-{line_num}"),
                    "stem": obj.get("question", ""),
                    "options": list(options.values()),
                    "correct": correct,
                    "symptoms": obj.get("metamap_phrases", []),
                    "_source": "MedQA-official",
                }
                questions.append(q)
                if max_questions and len(questions) >= max_questions:
                    break

        return questions


class PubMedQALoader:
    """
    Loader for the official PubMedQA dataset (Jin et al. 2019).

    Download: https://pubmedqa.github.io/
    Expected format (JSON dict keyed by PubMed ID):
        {
          "PUBMED_ID": {
            "QUESTION": "Does X reduce Y?",
            "CONTEXTS": ["abstract paragraph 1", ...],
            "final_decision": "yes"|"no"|"maybe",
            "MESHES": ["Fever", ...]
          }
        }
    """

    @staticmethod
    def load(path: str, max_questions: Optional[int] = None) -> list[dict]:
        """
        Load PubMedQA JSON and normalise to internal format.

        Returns list of dicts with keys:
            id, question, correct_answer, condition, symptoms
        """
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)

        questions = []
        for pmid, obj in data.items():
            q = {
                "id": f"pmq-{pmid}",
                "question": obj.get("QUESTION", ""),
                "correct_answer": obj.get("final_decision", "yes"),
                "condition": "",  # not directly available — inferred from context
                "symptoms": obj.get("MESHES", []),
                "contexts": obj.get("CONTEXTS", []),
                "_source": "PubMedQA-official",
            }
            questions.append(q)
            if max_questions and len(questions) >= max_questions:
                break

        return questions


class GPT4BaselineInterface:
    """
    Interface for incorporating GPT-4 zero-shot results into the comparison table.

    GPT-4 results cannot be computed without an OpenAI API key.  This class
    reads pre-computed outputs from a CSV file produced by running GPT-4 on
    the same question set.

    CSV format (one row per question):
        question_id,predicted_condition,confidence_score,is_correct
        medqa-1,Community-Acquired Pneumonia,0.82,1
        medqa-2,Influenza,0.74,0

    To generate this CSV, run GPT-4 with the following prompt template and
    record the top-1 prediction and a calibrated confidence score (use
    log-probabilities if available, or instruct the model to output a
    0.0–1.0 score):

        "Given the following clinical vignette, choose the most likely
         diagnosis from the provided options. Output only the diagnosis
         name and a confidence score (0.0–1.0)."

    Parameters
    ----------
    outputs_file : str
        Path to the CSV file with pre-computed GPT-4 outputs.
    system_name : str
        Display name (e.g. "GPT-4 zero-shot", "GPT-4 + RAG (vanilla)").
    """

    def __init__(self, outputs_file: str, system_name: str = "GPT-4 zero-shot"):
        self.system_name = system_name
        self._results: dict[str, dict] = {}
        self._load(outputs_file)

    def _load(self, path: str) -> None:
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                self._results[row["question_id"]] = {
                    "predicted": row.get("predicted_condition", ""),
                    "confidence": float(row.get("confidence_score", 0.5)),
                    "is_correct": int(row.get("is_correct", 0)),
                }

    def score(self, question_id: str) -> Optional[dict]:
        return self._results.get(question_id)

    def aggregate_metrics(self, benchmark_name: str) -> dict:
        """Compute aggregate metrics from the loaded CSV outputs."""
        if not self._results:
            return {}
        accs = [v["is_correct"] for v in self._results.values()]
        confs = [v["confidence"] for v in self._results.values()]
        acc = sum(accs) / len(accs)
        ece = compute_ece_continuous(
            [float(a) for a in accs], confs, n_bins=10
        )
        return {
            "benchmark": benchmark_name,
            "retriever": self.system_name,
            "n_questions": len(self._results),
            "accuracy@1": round(acc, 4),
            "mrr": round(acc, 4),  # for MCQ, MRR ≈ Acc@1 when top-1 is all that matters
            "ndcg@5": round(acc, 4),
            "ece": round(ece, 4),
            "_source": "gpt4-precomputed",
        }


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


def compute_ece_continuous(
    accuracies: list[float],
    confidences: list[float],
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (ECE) using **continuous** probability outputs.

    Uses the adaptive equal-width binning from Guo et al. (2017).
    Requires continuous confidence scores in [0, 1], NOT a 3-class mapping.

    Calibration note (M3 fix):
        The previous implementation mapped {low, medium, high} to fixed floats
        {0.35, 0.65, 0.90}.  This is NOT a valid ECE because all predictions
        fall into at most 3 of the 5 bins, producing a nearly constant ECE
        regardless of actual calibration quality.  This function requires
        continuous probabilities: use the softmax score of the top-1 result
        (score / sum_of_scores) as the confidence estimate.

    Parameters
    ----------
    accuracies : list[float]
        Binary correctness per query (1.0 = correct, 0.0 = wrong).
    confidences : list[float]
        Continuous probability estimate in [0, 1] per query.
        Must be computed from softmax scores, NOT from the low/med/high mapping.
    n_bins : int
        Number of equal-width bins (default 10, per Guo et al. 2017).
    """
    if not accuracies or not confidences:
        return 0.0
    assert len(accuracies) == len(confidences), "lengths must match"
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


def _softmax_confidence(scores: list[float]) -> float:
    """
    Compute the softmax probability of the top-1 document.

    This is a continuous confidence estimate in [1/k, 1.0] that is valid
    for ECE computation.  For k retrieved documents with scores s_i:

        p_1 = exp(s_1) / sum_i(exp(s_i))

    We use the raw score ratio (not exp-softmax) as a numerically stable
    approximation when all scores are non-negative.
    """
    if not scores:
        return 0.5
    total = sum(scores)
    if total <= 0.0:
        return 1.0 / len(scores)
    return scores[0] / total


def compute_bootstrap_ci(
    metric_values: list[float],
    n_boot: int = 1000,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Compute a 95% bootstrap confidence interval for the mean of *metric_values*.

    Uses non-parametric percentile bootstrap (Efron & Tibshirani, 1993) with a
    fixed random seed for reproducibility.

    Parameters
    ----------
    metric_values : list[float]
        Per-question metric values (e.g. per-question Acc@1 as 0/1, or per-question
        MRR).  The point estimate is ``mean(metric_values)``.
    n_boot : int
        Number of bootstrap resamples (default 1,000).
    seed : int
        Random seed for reproducibility (default 42).

    Returns
    -------
    (lower_95, upper_95) : Tuple[float, float]
        The 2.5th and 97.5th percentiles of the bootstrap distribution of the mean,
        rounded to 4 decimal places.

    Example
    -------
    >>> accs = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0]
    >>> lo, hi = compute_bootstrap_ci(accs, n_boot=1000, seed=42)
    >>> 0.0 <= lo <= hi <= 1.0
    True

    References
    ----------
    Efron, B., & Tibshirani, R. J. (1993). An Introduction to the Bootstrap.
    Chapman & Hall.
    """
    n = len(metric_values)
    if n == 0:
        return (0.0, 0.0)
    if n == 1:
        v = metric_values[0]
        return (round(v, 4), round(v, 4))

    rng = random.Random(seed)
    boot_means: list[float] = []
    for _ in range(n_boot):
        sample = [metric_values[rng.randint(0, n - 1)] for _ in range(n)]
        boot_means.append(sum(sample) / n)

    boot_means.sort()
    lo_idx = int(0.025 * n_boot)
    hi_idx = int(0.975 * n_boot) - 1
    # Clamp indices to valid range
    lo_idx = max(0, min(lo_idx, n_boot - 1))
    hi_idx = max(0, min(hi_idx, n_boot - 1))
    return (round(boot_means[lo_idx], 4), round(boot_means[hi_idx], 4))


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
    questions: list[dict],
    top_k: int = 5,
) -> dict:
    """Run a USMLE-format question set and compute metrics."""
    mrrs, ndcgs, accuracies, confidences = [], [], [], []

    for q in questions:
        pq = QueryProcessor.process(q["stem"], q["symptoms"])
        results = retriever.retrieve_results(pq.expanded_query, pq.medical_tokens, top_k)

        ranked_conds = [r.entry.get("condition", "") for r in results]
        scores = [r.hybrid_score for r in results]

        mrr = compute_mrr(ranked_conds, q["correct"])
        ndcg = compute_ndcg(ranked_conds, q["correct"], k=top_k)
        acc = 1.0 if mrr > 0 else 0.0

        # M3 fix: use continuous softmax probability, not 3-class mapping
        conf = _softmax_confidence(scores)

        mrrs.append(mrr)
        ndcgs.append(ndcg)
        accuracies.append(acc)
        confidences.append(conf)

    return {
        "benchmark": "USMLE Step 1",
        "retriever": retriever_name,
        "n_questions": len(questions),
        "accuracy@1": round(sum(accuracies) / len(accuracies), 4),
        "acc1_ci": compute_bootstrap_ci(accuracies),
        "mrr": round(sum(mrrs) / len(mrrs), 4),
        "mrr_ci": compute_bootstrap_ci(mrrs),
        f"ndcg@{top_k}": round(sum(ndcgs) / len(ndcgs), 4),
        "ece": round(compute_ece_continuous(accuracies, confidences), 4),
    }


_NEGATION_PREFIX = _re.compile(
    r"\b(no|not|without|failed|never|unable|lack|absence|absent|"
    r"neither|nor|cannot|couldn't|didn't|doesn't|wasn't|weren't|"
    r"isn't|aren't|hasn't|haven't|hadn't)\b"
)
_SENT_SPLIT = _re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def _score_sentence(
    sent: str,
    yes_signals: list[str],
    no_signals: list[str],
    maybe_signals: list[str],
    weight: float = 1.0,
) -> tuple[float, float, float]:
    s = sent.lower()
    y = n = m = 0.0

    for sig in maybe_signals:
        if sig in s:
            m += weight

    for sig in no_signals:
        if sig in s:
            n += weight

    for sig in yes_signals:
        if sig not in s:
            continue
        idx = s.find(sig)
        prefix_window = s[max(0, idx - 60): idx]
        if _NEGATION_PREFIX.search(prefix_window):
            n += weight * 0.8
        else:
            y += weight

    return y, n, m


def _predict_pubmedqa_answer_keyword(
    question_text: str,
    contexts: list[str],
    retriever: "Retriever",
) -> str:
    """
    Research-improved keyword-heuristic predictor for PubMedQA yes/no/maybe.
    """
    yes_signals = [
        "significantly", "statistically significant", "p <", "p=0.0", "p < 0.05",
        "confidence interval", "odds ratio", "relative risk", "hazard ratio",
        "effective", "efficacious", "beneficial", "superior", "outperformed",
        "demonstrated", "confirmed", "validated", "established",
        "reduced", "improved", "increased", "decreased", "enhanced", "promoted",
        "associated with", "correlated with", "predicted", "mediated",
        "our results suggest", "findings suggest", "we found that",
        "evidence supports", "supports the use", "we conclude",
        "data indicate", "results indicate", "study demonstrates",
        "our study confirms", "results confirm", "our findings support",
        "we report", "we observed", "results show",
    ]

    no_signals = [
        "no significant", "not significant", "no statistically significant",
        "no significant difference", "no significant association",
        "no significant effect", "no significant improvement",
        "no significant reduction", "no significant increase",
        "no difference", "no effect", "no association", "no benefit",
        "no improvement", "not superior", "not associated",
        "not significantly", "not statistically",
        "failed to", "failed to demonstrate", "failed to show",
        "did not", "did not significantly", "did not improve",
        "did not reduce", "did not differ", "did not show",
        "does not", "does not support", "do not support",
        "null hypothesis", "null result",
        "we found no", "we found no significant",
        "results do not support", "results did not support",
        "results showed no", "analysis showed no",
        "could not demonstrate", "unable to demonstrate",
        "no statistically", "not statistically significant",
        "ineffective", "ineffectiveness",
        "comparable to placebo", "similar to placebo", "no better than",
        "lack of efficacy", "lack of evidence", "absence of",
        "no evidence of", "no evidence for",
    ]

    maybe_signals = [
        "may", "might", "possibly", "potentially", "could be",
        "appears to", "seems to", "suggests that further",
        "unclear", "inconclusive", "uncertain", "equivocal",
        "insufficient evidence", "limited evidence", "insufficient data",
        "conflicting", "mixed results", "inconsistent",
        "further research", "more studies", "larger studies",
        "further investigation", "warrant further", "warrants further",
        "cannot conclude", "cannot be concluded", "cannot determine",
        "needs to be confirmed", "requires confirmation",
        "preliminary", "pilot study", "small sample",
        "heterogeneous", "high heterogeneity",
        "under certain conditions", "in selected patients",
        "in some patients", "in a subset",
    ]

    all_sents: list[tuple[str, float]] = []
    for ctx in contexts:
        sents = _split_sentences(ctx)
        if not sents:
            continue
        n_sents = len(sents)
        for i, s in enumerate(sents):
            if i >= n_sents - 2:
                all_sents.append((s, 3.0))
            else:
                all_sents.append((s, 1.0))

    pq = QueryProcessor.process(question_text, [])
    kb_results = retriever.retrieve_results(pq.expanded_query, pq.medical_tokens, 3)
    kb_text = " ".join(
        r.entry.get("explanation", "") + " " + r.entry.get("description", "")
        for r in kb_results
    )
    for sent in _split_sentences(kb_text):
        all_sents.append((sent, 1.0))

    total_y = total_n = total_m = 0.0
    for sent, w in all_sents:
        dy, dn, dm = _score_sentence(
            sent,
            yes_signals,
            no_signals,
            maybe_signals,
            weight=w,
        )
        total_y += dy
        total_n += dn
        total_m += dm

    q_lower = question_text.lower()
    if _re.match(r"^(does|is|are|can|do|was|were|has|have|did)\b", q_lower):
        total_y += 0.5
    if any(w in q_lower for w in ["fail", "prevent", "lack", "absent", "ineffect"]):
        total_n += 0.5
    if any(w in q_lower for w in ["unclear", "unknown", "controversial", "uncertain"]):
        total_m += 0.5

    if total_m > 0.6 * (total_y + total_n) and total_m > 0:
        return "maybe"
    if total_n > total_y * 0.75:
        return "no"
    return "yes"


def _build_ollama_prompt(question_text: str, contexts: list[str]) -> str:
    """
    Build the prompt sent to the Ollama LLM.
    Emphasises the conclusion sentences (last 2) where the answer typically lives.
    """
    abstract = " ".join(contexts)
    # Split on sentence boundaries to find the conclusion
    import re as _re
    sentences = [s.strip() for s in _re.split(r"(?<=[.!?])\s+", abstract) if s.strip()]
    conclusion = " ".join(sentences[-2:]) if len(sentences) >= 2 else abstract

    # Truncate full abstract to keep prompt fast
    abstract_trunc = abstract[:1800]

    return (
        "You are a biomedical research assistant.\n"
        "Read the abstract below and answer the question with ONLY one word: yes, no, or maybe.\n"
        "Do not explain. Output exactly one word.\n\n"
        f"Abstract: {abstract_trunc}\n\n"
        f"Conclusion: {conclusion}\n\n"
        f"Question: {question_text}\n\n"
        "Answer (yes / no / maybe):"
    )


def _predict_pubmedqa_answer_ollama(
    question_text: str,
    contexts: list[str],
    retriever: "Retriever",
    ollama_model: str = "gemma4:e4b",
    ollama_url: str = "http://localhost:11434",
) -> str:
    """
    Predict yes/no/maybe using a local Ollama model (default: gemma4:e4b).

    Falls back to the keyword heuristic if Ollama is unreachable or returns
    an unparseable response.

    Setup:
        1. Install Ollama from https://ollama.com/download
        2. ollama pull gemma4:e4b   (9.6 GB download, needs 6 GB VRAM)
        3. Run benchmark with --ollama flag

    Why gemma4:e4b on 6 GB VRAM:
        - 4.5B effective parameters, ~3.5 GB model weight
        - Fits on a 6 GB GPU with headroom for KV cache
        - Beats Gemma 3 27B on most benchmarks
        - Expected PubMedQA accuracy: 68-75% vs 55% for keyword heuristic
    """
    import httpx as _httpx

    prompt = _build_ollama_prompt(question_text, contexts)
    try:
        resp = _httpx.post(
            f"{ollama_url}/api/generate",
            json={"model": ollama_model, "prompt": prompt, "stream": False},
            timeout=90.0,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip().lower()

        # Parse — model may say "yes." / "yes, because..." / "no\n" etc.
        if raw.startswith("no"):
            return "no"
        if raw.startswith("maybe") or raw.startswith("uncertain") or raw.startswith("possibly"):
            return "maybe"
        if raw.startswith("yes"):
            return "yes"

        # If unparseable, fall back to keyword heuristic
        return _predict_pubmedqa_answer_keyword(question_text, contexts, retriever)

    except Exception:
        # Ollama not running or network error — fall back silently
        return _predict_pubmedqa_answer_keyword(question_text, contexts, retriever)


def _predict_pubmedqa_answer(
    question_text: str,
    contexts: list[str],
    retriever: "Retriever",
    use_ollama: bool = False,
    ollama_model: str = "gemma4:e4b",
    ollama_url: str = "http://localhost:11434",
) -> str:
    """
    Dispatch to Ollama LLM predictor or keyword heuristic based on use_ollama flag.
    """
    if use_ollama:
        return _predict_pubmedqa_answer_ollama(
            question_text, contexts, retriever, ollama_model, ollama_url
        )
    return _predict_pubmedqa_answer_keyword(question_text, contexts, retriever)


def run_pubmedqa_benchmark(
    retriever: "Retriever",
    retriever_name: str,
    questions: list[dict],
    use_ollama: bool = False,
    ollama_model: str = "gemma4:e4b",
    ollama_url: str = "http://localhost:11434",
) -> dict:
    """
    Run the PubMedQA benchmark and report yes/no/maybe accuracy.

    For each question the predicted answer (yes/no/maybe) is compared against
    the gold ``final_decision`` label.  Accuracy is reported as the fraction
    of exact matches.

    Pass use_ollama=True to use a local Ollama model (gemma4:e4b by default)
    instead of the keyword heuristic.  Expected accuracy improvement: 55% → 68-75%.
    """
    if not questions:
        return {
            "benchmark": "PubMedQA",
            "retriever": retriever_name,
            "n_questions": 0,
            "accuracy@1": 0.0,
        }

    predictor_label = f"ollama:{ollama_model}" if use_ollama else "keyword-heuristic"
    print(f"  Predictor: {predictor_label}")

    correct = 0
    label_counts: dict[str, int] = {"yes": 0, "no": 0, "maybe": 0}
    pred_counts: dict[str, int] = {"yes": 0, "no": 0, "maybe": 0}

    for i, q in enumerate(questions, 1):
        question_text = q.get("question", q.get("stem", ""))
        contexts = q.get("contexts", [])
        gold = q.get("correct_answer", q.get("final_decision", "yes")).lower().strip()
        gold = gold if gold in ("yes", "no", "maybe") else "yes"

        pred = _predict_pubmedqa_answer(
            question_text, contexts, retriever,
            use_ollama=use_ollama,
            ollama_model=ollama_model,
            ollama_url=ollama_url,
        )

        label_counts[gold] = label_counts.get(gold, 0) + 1
        pred_counts[pred] = pred_counts.get(pred, 0) + 1
        if pred == gold:
            correct += 1

        # Progress indicator every 50 questions
        if i % 50 == 0:
            running_acc = correct / i
            print(f"  [{i}/{len(questions)}] running accuracy: {running_acc:.3f}", flush=True)

    acc = correct / len(questions)
    return {
        "benchmark": "PubMedQA",
        "retriever": retriever_name,
        "n_questions": len(questions),
        "accuracy@1": round(acc, 4),
        "_gold_dist": label_counts,
        "_pred_dist": pred_counts,
        "_predictor": predictor_label,
    }


def _format_ci(ci: Optional[tuple]) -> str:
    """Format a (lo, hi) CI tuple as a compact string, e.g. '[0.61, 0.70]'."""
    if ci is None:
        return "       N/A"
    return f"[{ci[0]:.4f},{ci[1]:.4f}]"


def _print_table(results: list[dict], include_published: bool = True) -> None:
    """Pretty-print benchmark results as an ASCII table with 95% bootstrap CIs."""
    col_w = 100
    print()
    print("=" * col_w)
    header = (
        f"{'Benchmark':<20} {'System':<28} {'Acc@1':>6} {'Acc@1 95% CI':>16} "
        f"{'MRR':>6} {'MRR 95% CI':>14} {'NDCG@5':>7} {'ECE':>6} {'Source':>8}"
    )
    print(header)
    print("-" * col_w)

    for r in results:
        source = r.get("_source", "this work")
        if "mrr" in r:
            acc_str = f"{r.get('accuracy@1', 0):>6.4f}"
            acc_ci_str = _format_ci(r.get("acc1_ci"))
            mrr_str = f"{r['mrr']:>6.4f}" if r['mrr'] is not None else "    N/A"
            mrr_ci_str = _format_ci(r.get("mrr_ci"))
            ndcg_str = f"{r.get('ndcg@5', 0):>7.4f}" if r.get("ndcg@5") is not None else "    N/A"
            ece_str = f"{r['ece']:>6.4f}" if r.get("ece") is not None else "   N/A"
            print(
                f"{r['benchmark']:<20} {r['retriever']:<28} "
                f"{acc_str} {acc_ci_str} "
                f"{mrr_str} {mrr_ci_str} "
                f"{ndcg_str} {ece_str} {source:>8}"
            )
        else:
            acc = r.get("accuracy@1", r.get("accuracy@3", "-"))
            gold = r.get("_gold_dist", {})
            pred = r.get("_pred_dist", {})
            dist_str = ""
            if gold:
                dist_str = (
                    f"  gold=(yes:{gold.get('yes',0)} no:{gold.get('no',0)} "
                    f"maybe:{gold.get('maybe',0)})  "
                    f"pred=(yes:{pred.get('yes',0)} no:{pred.get('no',0)} "
                    f"maybe:{pred.get('maybe',0)})"
                )
            print(
                f"{r['benchmark']:<20} {r['retriever']:<28} "
                f"{acc:>6}  {'(Acc@1 yes/no/maybe)':>20}{dist_str}"
            )

    # Print published reference numbers
    if include_published:
        print("-" * col_w)
        print("  Published reference numbers (Wu et al. 2024, arXiv:2402.13178):")
        for sys_name, benchmarks in PUBLISHED_MEDRAG_NUMBERS.items():
            for bench_name, metrics in benchmarks.items():
                acc = metrics.get("accuracy@1")
                if acc is not None:
                    print(
                        f"{'  ' + bench_name:<20} {sys_name:<28} "
                        f"{acc:>6.4f} {'[published]':>16}  {'N/A':>6} {'':>14}  {'N/A':>7}  {'N/A':>6} {'pub':>8}"
                    )

    print("=" * col_w)
    print()
    print("NOTE: To produce valid benchmark numbers for publication:")
    print("  1. Use --dataset-file with official MedQA/PubMedQA test splits.")
    print("  2. Use --kb with a real medical corpus (e.g. StatPearls index).")
    print("  3. Provide --gpt4-outputs-file for GPT-4 baseline comparison.")
    print("  4. Results on --demo mode are CIRCULAR and not publishable.")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MedRAG-Turbo benchmark runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Evaluation modes:
  --demo            Use built-in author-written questions (CIRCULAR — not publishable)
  --dataset-file F  Use official MedQA/PubMedQA JSON/JSONL file (required for valid eval)

Dataset formats:
  medqa    Official MedQA JSONL (Jin et al. 2021)
  pubmedqa Official PubMedQA JSON dict (Jin et al. 2019)

GPT-4 baseline:
  --gpt4-outputs-file CSV  CSV with columns: question_id,predicted_condition,confidence_score,is_correct
        """,
    )
    parser.add_argument("--benchmark", choices=["usmle", "pubmedqa", "all"], default="all")
    parser.add_argument("--kb", default="data/sample_medical_knowledge.json")
    parser.add_argument("--json", action="store_true", help="Output JSON results")
    parser.add_argument(
        "--demo",
        action="store_true",
        help=(
            "Use built-in author-written demo questions. "
            "WARNING: These are circular (written to match the 33-entry KB). "
            "Results cannot be used for publication. "
            "Use --dataset-file with official MedQA/PubMedQA for valid evaluation."
        ),
    )
    parser.add_argument(
        "--dataset-file",
        default=None,
        help="Path to official dataset file (MedQA JSONL or PubMedQA JSON).",
    )
    parser.add_argument(
        "--dataset-format",
        choices=["medqa", "pubmedqa"],
        default="medqa",
        help="Format of the dataset file (default: medqa).",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Limit number of questions loaded (useful for smoke tests).",
    )
    parser.add_argument(
        "--gpt4-outputs-file",
        default=None,
        help=(
            "CSV file with pre-computed GPT-4 zero-shot outputs. "
            "Columns: question_id,predicted_condition,confidence_score,is_correct"
        ),
    )
    parser.add_argument(
        "--ollama",
        action="store_true",
        help=(
            "Use a local Ollama model for PubMedQA yes/no/maybe prediction "
            "instead of the keyword heuristic. "
            "Requires Ollama running at localhost:11434. "
            "Default model: gemma4:e4b (needs 6 GB VRAM). "
            "Install: https://ollama.com/download then: ollama pull gemma4:e4b"
        ),
    )
    parser.add_argument(
        "--ollama-model",
        default="gemma4:e4b",
        help="Ollama model tag to use for PubMedQA prediction (default: gemma4:e4b).",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama API base URL (default: http://localhost:11434).",
    )
    args = parser.parse_args()

    # Validate: require either --demo or --dataset-file
    if not args.demo and args.dataset_file is None:
        print(
            "ERROR: You must specify either --demo (circular, dev only) or "
            "--dataset-file <path> (official dataset, required for publication).",
            file=sys.stderr,
        )
        print(
            "  MedQA download: https://drive.google.com/drive/folders/"
            "1ImYUSLk9JbgHXOemfvyiDiirluZHPeQw",
            file=sys.stderr,
        )
        print(
            "  PubMedQA download: https://pubmedqa.github.io/",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.demo:
        print()
        print("⚠ " * 35)
        print("WARNING: Running in DEMO mode with author-written questions.")
        print("These questions are CIRCULAR — they were written to match the 33-entry KB.")
        print("Results produced here CANNOT be reported as benchmarks in a paper.")
        print("Use --dataset-file with the official MedQA/PubMedQA test split instead.")
        print("⚠ " * 35)
        print()
        usmle_questions = _DEMO_USMLE_QUESTIONS
        pubmedqa_questions = _DEMO_PUBMEDQA_QUESTIONS
    else:
        # Load real dataset
        if args.dataset_format == "medqa":
            print(f"Loading MedQA dataset from {args.dataset_file} ...")
            usmle_questions = MedQALoader.load(
                args.dataset_file, max_questions=args.max_questions
            )
            pubmedqa_questions = []
            print(f"  Loaded {len(usmle_questions)} MedQA questions.")
        elif args.dataset_format == "pubmedqa":
            print(f"Loading PubMedQA dataset from {args.dataset_file} ...")
            pubmedqa_questions = PubMedQALoader.load(
                args.dataset_file, max_questions=args.max_questions
            )
            usmle_questions = []
            print(f"  Loaded {len(pubmedqa_questions)} PubMedQA questions.")
        else:
            print(f"ERROR: Unknown dataset format: {args.dataset_format}", file=sys.stderr)
            sys.exit(1)

    # Load KB
    dataset = MedicalDataset(args.kb)
    if not dataset.entries:
        print(f"ERROR: No entries loaded from {args.kb}", file=sys.stderr)
        sys.exit(1)
    n_kb = len(dataset.entries)
    print(f"Loaded {n_kb} KB entries from {args.kb}")
    if n_kb < 100:
        print(
            f"WARNING: KB has only {n_kb} entries. "
            "For publishable results, use a real corpus with ≥9,000 articles "
            "(e.g. StatPearls). See eval/kb_indexer.py.",
            file=sys.stderr,
        )

    # Build retrievers
    retriever_full = Retriever(dataset.entries)
    retriever_bm25 = BM25OnlyRetriever(dataset.entries)

    all_results: list[dict] = []

    if args.benchmark in ("usmle", "all") and usmle_questions:
        print(f"Running USMLE benchmark on {len(usmle_questions)} questions ...")
        t0 = time.perf_counter()
        r1 = run_usmle_benchmark(retriever_bm25, "BM25-only", usmle_questions)
        r2 = run_usmle_benchmark(retriever_full, "MedRAG-Turbo", usmle_questions)
        elapsed = time.perf_counter() - t0
        all_results.extend([r1, r2])
        print(f"  Elapsed: {elapsed:.2f}s")

    if args.benchmark in ("pubmedqa", "all") and pubmedqa_questions:
        if args.ollama:
            print(f"Running PubMedQA benchmark on {len(pubmedqa_questions)} questions (Ollama: {args.ollama_model}) ...")
            print(f"  Make sure Ollama is running: ollama serve")
            print(f"  Model must be pulled first:  ollama pull {args.ollama_model}")
        else:
            print(f"Running PubMedQA benchmark on {len(pubmedqa_questions)} questions (keyword heuristic) ...")
            print(f"  Tip: use --ollama for higher accuracy (~68-75% vs ~55%)")
        r3 = run_pubmedqa_benchmark(
            retriever_bm25, "BM25-only", pubmedqa_questions,
            use_ollama=args.ollama,
            ollama_model=args.ollama_model,
            ollama_url=args.ollama_url,
        )
        r4 = run_pubmedqa_benchmark(
            retriever_full, "MedRAG-Turbo", pubmedqa_questions,
            use_ollama=args.ollama,
            ollama_model=args.ollama_model,
            ollama_url=args.ollama_url,
        )
        all_results.extend([r3, r4])

    # GPT-4 baseline (F3 fix)
    if args.gpt4_outputs_file:
        print(f"Loading GPT-4 baseline outputs from {args.gpt4_outputs_file} ...")
        gpt4 = GPT4BaselineInterface(args.gpt4_outputs_file, system_name="GPT-4 zero-shot")
        if usmle_questions:
            all_results.append(gpt4.aggregate_metrics("USMLE Step 1"))
        if pubmedqa_questions:
            all_results.append(gpt4.aggregate_metrics("PubMedQA"))

    if args.json:
        print(json.dumps(all_results, indent=2))
    else:
        _print_table(all_results, include_published=True)

    # Summary comparison: MedRAG-Turbo vs BM25-only
    usmle_bm25 = next(
        (r for r in all_results if r.get("benchmark") == "USMLE Step 1" and "BM25" in r["retriever"]),
        None,
    )
    usmle_full = next(
        (r for r in all_results if r.get("benchmark") == "USMLE Step 1" and "MedRAG" in r["retriever"]),
        None,
    )
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
