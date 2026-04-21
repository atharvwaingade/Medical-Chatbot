import json
import random
from typing import List, Dict

from rag.retriever import Retriever
from rag.temporal_graph import TemporalGraph


# ============================================================
# STEP 1 — BUILD DATASET
# ============================================================

def build_cases():
    base_cases = [
        ("pneumonia", ["fever", "cough", "dyspnoea"]),
        ("acs", ["chest_pain", "sweating", "arm_pain"]),
        ("hepatitis", ["fatigue", "nausea", "jaundice"]),
        ("influenza", ["myalgia", "fever", "cough"]),
        ("uti", ["dysuria", "frequency", "suprapubic_pain"]),
    ]

    dataset = []

    for condition, symptoms in base_cases:
        for i in range(4):  # 5 × 4 = 20 cases
            canonical = symptoms.copy()
            reversed_order = list(reversed(symptoms))

            dataset.append({
                "id": f"{condition}_{i}",
                "condition": condition,
                "canonical": canonical,
                "reversed": reversed_order
            })

    return dataset


# ============================================================
# HELPER — FORMAT MULTI-TURN QUERY
# ============================================================

def build_query(symptoms: List[str]) -> List[str]:
    # simulate multi-turn interaction
    return [f"I have {s}" for s in symptoms]


# ============================================================
# STEP 2 — RETRIEVAL + TSPG
# ============================================================

def retrieve_with_optional_tspg(
    retriever: Retriever,
    temporal_graph: TemporalGraph,
    query_turns: List[str],
    use_tspg: bool
):
    # join turns into single query (or your pipeline equivalent)
    query_text = " ".join(query_turns)

    results = retriever.retrieve(query_text, top_k=5)

    if not use_tspg:
        return results

    # apply temporal scoring
    scored = []
    for r in results:
        score = temporal_graph.score(query_turns, r["text"])
        scored.append((r, score))

    # re-rank
    scored.sort(key=lambda x: x[1], reverse=True)

    return [x[0] for x in scored]


# ============================================================
# STEP 3 — METRICS
# ============================================================

def compute_acc_at_1(results, ground_truth):
    top = results[0]["label"] if results else None
    return 1 if top == ground_truth else 0


# ============================================================
# MAIN EVALUATION
# ============================================================

def run_ablation():
    dataset = build_cases()

    retriever = Retriever()
    temporal_graph = TemporalGraph()

    stats = {
        "canonical": {"on": [], "off": []},
        "reversed": {"on": [], "off": []}
    }

    for case in dataset:
        gt = case["condition"]

        # -------- canonical --------
        query_c = build_query(case["canonical"])

        res_off = retrieve_with_optional_tspg(
            retriever, temporal_graph, query_c, use_tspg=False
        )
        res_on = retrieve_with_optional_tspg(
            retriever, temporal_graph, query_c, use_tspg=True
        )

        stats["canonical"]["off"].append(compute_acc_at_1(res_off, gt))
        stats["canonical"]["on"].append(compute_acc_at_1(res_on, gt))

        # -------- reversed --------
        query_r = build_query(case["reversed"])

        res_off = retrieve_with_optional_tspg(
            retriever, temporal_graph, query_r, use_tspg=False
        )
        res_on = retrieve_with_optional_tspg(
            retriever, temporal_graph, query_r, use_tspg=True
        )

        stats["reversed"]["off"].append(compute_acc_at_1(res_off, gt))
        stats["reversed"]["on"].append(compute_acc_at_1(res_on, gt))

    # ========================================================
    # COMPUTE METRICS
    # ========================================================

    def mean(x): return sum(x) / len(x)

    acc1_c_on = mean(stats["canonical"]["on"])
    acc1_c_off = mean(stats["canonical"]["off"])

    acc1_r_on = mean(stats["reversed"]["on"])
    acc1_r_off = mean(stats["reversed"]["off"])

    delta_canonical = acc1_c_on - acc1_c_off
    delta_reversed = acc1_r_on - acc1_r_off

    results = {
        "Acc@1": {
            "canonical_on": acc1_c_on,
            "canonical_off": acc1_c_off,
            "reversed_on": acc1_r_on,
            "reversed_off": acc1_r_off
        },
        "Delta": {
            "canonical": delta_canonical,
            "reversed": delta_reversed
        },
        "Hypothesis": {
            "expected": "Delta_canonical > 0 and |Delta_reversed| < Delta_canonical",
            "observed": {
                "delta_canonical": delta_canonical,
                "delta_reversed": delta_reversed
            }
        }
    }

    # ========================================================
    # STEP 4 — SAVE RESULTS
    # ========================================================

    with open("results/tspg_ablation_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\n===== TSPG ABLATION RESULTS =====")
    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    run_ablation()