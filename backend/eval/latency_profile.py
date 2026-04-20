#!/usr/bin/env python3
"""
Latency Profiler: End-to-End Retrieval Pipeline
=================================================
Measures the latency of each stage of the MedRAG-Turbo non-LLM retrieval
pipeline across 100 queries.  Reports mean ± std latency per stage.

Target: sub-10ms total non-LLM retrieval latency for clinical viability.

Stages Profiled
---------------
1. Query preprocessing (QueryProcessor.process)
2. BM25 retrieval (Retriever._bm25_score)
3. SCS scoring (Retriever._scs)
4. PRF expansion (Retriever._extract_expansion_terms)
5. Bayesian + GRADE final scoring
6. Causal re-ranking (MedCausalGraph.causal_scores)
7. RAPTOR routing (MedRAPTOR.route_query)
8. Conformal prediction set (ConformalPredictor.predict_set)
9. Symptom attribution (compute_symptom_attributions)
10. Active inquiry VOI (recommend_next_question)
11. Contrastive DDx + Bayes factor (compute_pairwise_contrasts)
12. Total (non-LLM) pipeline

References
----------
Dean, J., & Ghemawat, S. (2013). The tail at scale. *Communications of the
ACM*, 56(2), 74–80.  (Motivates sub-10ms requirements for clinical CDS.)
"""
from __future__ import annotations

import math
import os
import sys
import time
from typing import Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag.dataset import MedicalDataset
from app.rag.retriever import Retriever
from app.rag.query_processor import QueryProcessor
from app.rag.causal_graph import MedCausalGraph
from app.rag.raptor import MedRAPTOR
from app.rag.conformal import ConformalPredictor
from app.rag.symptom_attribution import compute_symptom_attributions
from app.rag.active_inquiry import recommend_next_question
from app.rag.contrastive import compute_pairwise_contrasts


# ---------------------------------------------------------------------------
# 100 Benchmark Queries
# ---------------------------------------------------------------------------

BENCHMARK_QUERIES: list[tuple[str, list[str]]] = [
    ("I have chest pain and shortness of breath", ["chest pain", "shortness of breath"]),
    ("fever cough body aches", ["fever", "cough", "body aches"]),
    ("runny nose sore throat mild fever", ["runny nose", "sore throat"]),
    ("burning when urinating frequent urination", ["burning urination", "frequent urination"]),
    ("severe headache nausea photophobia", ["severe headache", "nausea"]),
    ("fatigue weight loss night sweats", ["fatigue", "unexplained weight loss"]),
    ("chest tightness palpitations dizziness", ["chest tightness", "palpitations"]),
    ("nausea vomiting diarrhea abdominal pain", ["nausea", "vomiting", "diarrhea"]),
    ("increased thirst frequent urination blurred vision", ["increased thirst", "frequent urination"]),
    ("persistent sadness no interest insomnia", ["persistent sadness", "loss of interest"]),
    ("shortness of breath swollen ankles", ["shortness of breath", "leg swelling"]),
    ("cold intolerance weight gain dry skin constipation", ["cold intolerance", "weight gain"]),
    ("heat intolerance tremor weight loss palpitations", ["heat intolerance", "palpitations"]),
    ("sudden weakness facial droop slurred speech", ["sudden weakness", "facial droop"]),
    ("wheezing cough shortness of breath at night", ["wheezing", "cough"]),
    ("rash fever joint pain", ["skin rash", "fever", "joint pain"]),
    ("back pain flank pain nausea fever", ["lower back pain", "fever", "nausea"]),
    ("excessive worry restlessness difficulty concentrating", ["anxiety"]),
    ("productive cough fever chills pleuritic pain", ["productive cough", "fever", "chills"]),
    ("abdominal pain after fatty meal nausea vomiting", ["abdominal pain", "nausea"]),
    ("hair loss fatigue weight gain puffiness", ["hair loss", "fatigue", "weight gain"]),
    ("leg swelling shortness of breath fatigue", ["leg swelling", "shortness of breath"]),
    ("skin rash itching hives", ["skin rash", "itching"]),
    ("blood in stool abdominal pain", ["blood in stool", "abdominal pain"]),
    ("dizziness vertigo nausea when standing", ["dizziness", "nausea"]),
    ("chest pain radiating to left arm jaw sweating", ["chest pain", "left arm pain", "sweating"]),
    ("fever productive cough loss of appetite", ["fever", "productive cough", "loss of appetite"]),
    ("headache stiff neck high fever", ["headache", "stiff neck", "fever"]),
    ("unexplained weight loss cough hemoptysis fatigue", ["unexplained weight loss", "cough"]),
    ("swollen lymph nodes fatigue night sweats", ["lymphadenopathy", "fatigue"]),
    ("I feel very tired and weak all the time", ["fatigue", "weakness"]),
    ("my heart is racing and I feel faint", ["palpitations", "dizziness"]),
    ("I can't breathe and my legs are swollen", ["shortness of breath", "leg swelling"]),
    ("tummy ache nausea sick to stomach", ["abdominal pain", "nausea"]),
    ("throbbing headache one side of head with flashing lights", ["severe headache"]),
    ("high temperature and sweating", ["fever", "sweating"]),
    ("no fever but chest pain", ["chest pain"]),
    ("dizzy spells when getting up quickly", ["dizziness"]),
    ("joint pain and stiffness in the morning", ["joint pain"]),
    ("calf pain swollen leg after long flight", ["calf pain", "leg swelling"]),
    ("vomiting blood black tarry stool", ["vomiting", "blood in stool"]),
    ("confusion memory loss personality change", ["confusion", "memory impairment"]),
    ("double vision headache neck stiffness fever", ["double vision", "headache", "stiff neck"]),
    ("shoulder tip pain and abdominal pain", ["shoulder tip pain", "abdominal pain"]),
    ("ear pain fever child", ["ear pain", "fever"]),
    ("red eyes discharge", ["red eye"]),
    ("difficulty swallowing throat pain", ["difficulty swallowing", "sore throat"]),
    ("chest pain worse on breathing", ["chest pain", "shortness of breath"]),
    ("weight loss and cough smoker 65 years old", ["unexplained weight loss", "cough"]),
    ("frequent infections easy bruising", ["fatigue"]),
    # 50 more varied queries
    ("I have a headache and feel sick", ["headache", "nausea"]),
    ("stomach hurts and I have diarrhea", ["abdominal pain", "diarrhea"]),
    ("my throat is really sore", ["sore throat"]),
    ("runny nose and sneezing", ["runny nose", "sneezing"]),
    ("pain when I urinate", ["burning urination"]),
    ("I feel dizzy when I stand up", ["dizziness"]),
    ("bad indigestion after meals", ["heartburn", "indigestion"]),
    ("chest feels tight and I am wheezing", ["chest tightness", "wheezing"]),
    ("I have been losing weight without trying", ["unexplained weight loss"]),
    ("I feel sad and hopeless all the time", ["persistent sadness", "hopelessness"]),
    ("my face is puffy and I feel cold all the time", ["facial puffiness", "cold intolerance"]),
    ("I have been drinking a lot of water and urinating a lot", ["increased thirst", "frequent urination"]),
    ("severe pain in my lower back that goes to my groin", ["lower back pain", "groin pain"]),
    ("rash all over my body that itches", ["skin rash", "itching"]),
    ("I fainted after chest pain", ["chest pain", "dizziness"]),
    ("numbness in my hands and feet", ["sudden numbness"]),
    ("I can not remember things anymore", ["memory impairment"]),
    ("my stomach is swollen and hard", ["abdominal pain", "bloating"]),
    ("I see flashes of light in my vision", ["blurred vision"]),
    ("painful swollen joints in my hands", ["joint pain"]),
    ("I have been coughing blood", ["cough"]),
    ("my ankles are swollen", ["leg swelling"]),
    ("I have a burning feeling in my chest after eating", ["heartburn"]),
    ("my child has a high fever and stiff neck", ["fever", "stiff neck"]),
    ("I can not stop worrying about everything", ["anxiety"]),
    ("I feel very anxious and my heart races", ["anxiety", "palpitations"]),
    ("I feel sad no energy and can not sleep", ["persistent sadness", "fatigue"]),
    ("I have been itching all over", ["itching"]),
    ("terrible headache at the back of my head", ["severe headache"]),
    ("I have been vomiting for 2 days", ["vomiting"]),
    ("pain in my side and need to urinate often", ["flank pain", "frequent urination"]),
    ("my vision is blurry and I have a headache", ["blurred vision", "headache"]),
    ("I feel breathless when I walk upstairs", ["shortness of breath"]),
    ("I have a dry cough that won't go away", ["dry cough"]),
    ("my leg is red and swollen and painful", ["leg swelling"]),
    ("I have heartburn every day", ["heartburn"]),
    ("I feel confused and disoriented", ["confusion"]),
    ("nose keeps running and eyes are itching", ["runny nose", "itching"]),
    ("I have had a fever for a week", ["fever"]),
    ("chest pain that gets worse when I breathe in", ["chest pain"]),
    ("I have been losing my hair", ["hair loss"]),
    ("sore muscles and fever after the gym", ["body aches", "fever"]),
    ("I have pink eye", ["red eye"]),
    ("my ear hurts and I have a temperature", ["ear pain", "fever"]),
    ("pain behind my eye and a bad headache", ["severe headache"]),
    ("I have been constipated for a week", ["constipation"]),
    ("night sweats and weight loss", ["sweating", "unexplained weight loss"]),
    ("I feel short of breath lying flat", ["shortness of breath"]),
    ("I have blood in my urine", ["burning urination"]),
    ("my throat is sore and I have white patches", ["sore throat"]),
]


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------

def _time_fn(fn: Callable, *args) -> tuple[float, object]:
    """Return (elapsed_ms, result)."""
    t0 = time.perf_counter()
    result = fn(*args)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return elapsed_ms, result


def _stats(times: list[float]) -> dict:
    n = len(times)
    mean = sum(times) / n
    variance = sum((t - mean) ** 2 for t in times) / n
    std = math.sqrt(variance)
    return {
        "mean_ms": round(mean, 3),
        "std_ms": round(std, 3),
        "min_ms": round(min(times), 3),
        "max_ms": round(max(times), 3),
        "p95_ms": round(sorted(times)[int(0.95 * n)], 3),
    }


# ---------------------------------------------------------------------------
# Main profiler
# ---------------------------------------------------------------------------


def run_latency_profile(
    dataset_path: str = "data/sample_medical_knowledge.json",
) -> dict:
    """Run latency profiling over all benchmark queries."""
    dataset = MedicalDataset(dataset_path)
    if not dataset.entries:
        print(f"ERROR: No entries loaded from {dataset_path}", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(dataset.entries)} KB entries.")

    retriever = Retriever(dataset.entries)
    causal = MedCausalGraph()
    causal.build_from_entries(dataset.entries)
    raptor = MedRAPTOR()
    raptor.build_from_entries(dataset.entries)
    conformal = ConformalPredictor(retriever)

    # Calibrate conformal predictor
    conformal.calibrate()

    n = len(BENCHMARK_QUERIES)
    print(f"Profiling {n} queries across all pipeline stages...")

    stage_times: dict[str, list[float]] = {
        "query_processing": [],
        "retrieval": [],
        "causal_rerank": [],
        "raptor_routing": [],
        "conformal": [],
        "symptom_attribution": [],
        "active_inquiry_voi": [],
        "contrastive_ddx": [],
        "total_non_llm": [],
    }

    for query, symptoms in BENCHMARK_QUERIES:
        t_total_start = time.perf_counter()

        # Stage 1: Query processing
        t_qp, pq = _time_fn(QueryProcessor.process, query, symptoms)
        stage_times["query_processing"].append(t_qp)

        # Stage 2+3: Full retrieval (BM25 + SCS + PRF + GRADE)
        t_ret, results = _time_fn(
            retriever.retrieve_results, pq.expanded_query, pq.medical_tokens, 5
        )
        stage_times["retrieval"].append(t_ret)

        if not results:
            for k in stage_times:
                if k not in ("query_processing", "retrieval"):
                    stage_times[k].append(0.0)
            stage_times["total_non_llm"].append(
                (time.perf_counter() - t_total_start) * 1000
            )
            continue

        # Stage 4: Causal re-ranking
        affirmed = pq.medical_tokens
        t_causal, causal_scores = _time_fn(
            causal.causal_scores, affirmed, pq.negated_terms, None
        )
        stage_times["causal_rerank"].append(t_causal)

        # Stage 5: RAPTOR routing
        t_raptor, _ = _time_fn(raptor.route_systems, query)
        stage_times["raptor_routing"].append(t_raptor)

        # Stage 6: Conformal prediction set
        top_conditions = [r.entry.get("condition", "") for r in results]
        t_conf, _ = _time_fn(conformal.predict_set, results)
        stage_times["conformal"].append(t_conf)

        # Stage 7: Symptom attribution (Shapley)
        top_condition = results[0].entry.get("condition", "") if results else ""
        t_attr, _ = _time_fn(
            compute_symptom_attributions,
            retriever,
            pq.medical_tokens,
            top_condition,
        )
        stage_times["symptom_attribution"].append(t_attr)

        # Stage 8: Active inquiry VOI
        t_voi, _ = _time_fn(
            recommend_next_question,
            retriever,
            pq.medical_tokens,
            results,
            30,  # cap candidates
        )
        stage_times["active_inquiry_voi"].append(t_voi)

        # Stage 9: Contrastive DDx + Bayes factor
        affirmed_set = set(pq.medical_tokens)
        t_contrast, _ = _time_fn(
            compute_pairwise_contrasts, results, affirmed_set
        )
        stage_times["contrastive_ddx"].append(t_contrast)

        total_ms = (time.perf_counter() - t_total_start) * 1000
        stage_times["total_non_llm"].append(total_ms)

    # Compute statistics
    profile = {}
    for stage, times in stage_times.items():
        if times:
            profile[stage] = _stats(times)

    return profile


def _print_profile(profile: dict) -> None:
    """Print latency profile as formatted table."""
    print()
    print("=" * 72)
    print(f"{'Stage':<26} {'Mean (ms)':>10} {'Std (ms)':>9} {'Min':>8} {'P95':>8}")
    print("-" * 72)

    order = [
        "query_processing",
        "retrieval",
        "causal_rerank",
        "raptor_routing",
        "conformal",
        "symptom_attribution",
        "active_inquiry_voi",
        "contrastive_ddx",
        "total_non_llm",
    ]

    for stage in order:
        if stage not in profile:
            continue
        s = profile[stage]
        sep = "─" * 72 if stage == "total_non_llm" else ""
        if sep:
            print(sep)
        print(
            f"{stage:<26} {s['mean_ms']:>10.3f} {s['std_ms']:>9.3f} "
            f"{s['min_ms']:>8.3f} {s['p95_ms']:>8.3f}"
        )

    print("=" * 72)
    total = profile.get("total_non_llm", {})
    mean = total.get("mean_ms", 0)
    status = "✓ Sub-10ms target met" if mean < 10 else f"△ {mean:.1f}ms > 10ms target"
    print(f"\nTotal non-LLM pipeline mean: {mean:.3f}ms  →  {status}")
    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Latency profiler for MedRAG-Turbo")
    parser.add_argument("--kb", default="data/sample_medical_knowledge.json")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    profile = run_latency_profile(args.kb)

    if args.json:
        import json
        print(json.dumps(profile, indent=2))
    else:
        _print_profile(profile)
