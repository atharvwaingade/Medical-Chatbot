# Benchmark Results — MedRAG-Turbo

## Main Comparison Table

> **Status:** Rows marked **[PLACEHOLDER]** require running the evaluation commands
> below with the official MedQA and PubMedQA test splits and a StatPearls KB.
> Published-paper reference rows (Wu et al. 2024) are filled with the original
> reported numbers and are included for context.

### MedQA (USMLE Step 1/2/3, US test split, *n* = 1,273)

| System | Acc@1 | Acc@1 95% CI | MRR | MRR 95% CI | NDCG@5 | ECE | Source |
|--------|------:|:------------:|----:|:----------:|-------:|----:|--------|
| BM25-only (ours) | **[PLACEHOLDER]** | — | **[PLACEHOLDER]** | — | — | — | this work |
| MedRAG-Turbo (ours) | **[PLACEHOLDER]** | — | **[PLACEHOLDER]** | — | — | — | this work |
| GPT-4 zero-shot | 0.7370 | N/A | N/A | N/A | N/A | N/A | Wu et al. (2024) |
| MedRAG BM25 | 0.6630 | N/A | N/A | N/A | N/A | N/A | Wu et al. (2024) |
| MedRAG full | 0.8270 | N/A | N/A | N/A | N/A | N/A | Wu et al. (2024) |

### PubMedQA (test split, *n* = 500)

| System | Acc@1 | Acc@1 95% CI | MRR | MRR 95% CI | ECE | Source |
|--------|------:|:------------:|----:|:----------:|----:|--------|
| BM25-only (ours) | **[PLACEHOLDER]** | — | **[PLACEHOLDER]** | — | — | this work |
| MedRAG-Turbo (ours) | **[PLACEHOLDER]** | — | **[PLACEHOLDER]** | — | — | this work |
| GPT-4 zero-shot | 0.7820 | N/A | N/A | N/A | N/A | Wu et al. (2024) |
| MedRAG BM25 | 0.7450 | N/A | N/A | N/A | N/A | Wu et al. (2024) |
| MedRAG full | 0.7960 | N/A | N/A | N/A | N/A | Wu et al. (2024) |

---

## How to Populate the Placeholder Rows

### Step 1 — Build the StatPearls KB

```bash
cd backend

# Download StatPearls JSONL from HuggingFace (MedRAG/textbooks, statpearls split):
# https://huggingface.co/datasets/MedRAG/textbooks

python eval/kb_indexer.py \
    --format statpearls \
    --source-file /path/to/statpearls.jsonl \
    --output data/statpearls_kb.json
# Expected output: ≥9,000 KB entries
```

### Step 2 — Run MedQA Benchmark

```bash
# Download MedQA-4opt test split (Jin et al. 2021):
# https://drive.google.com/drive/folders/1ImYUSLk9JbgHXOemfvyiDiirluZHPeQw
# File: data/questions/US/test.jsonl  (1,273 questions)

python eval/benchmark_runner.py \
    --dataset-file /path/to/medqa/questions/US/test.jsonl \
    --dataset-format medqa \
    --kb data/statpearls_kb.json \
    --json > results/medqa_results.json
```

### Step 3 — Run PubMedQA Benchmark

```bash
# Download PubMedQA test split (Jin et al. 2019):
# https://pubmedqa.github.io/
# File: data/test_ground_truth.json  (500 questions)

python eval/benchmark_runner.py \
    --dataset-file /path/to/pubmedqa/data/test_ground_truth.json \
    --dataset-format pubmedqa \
    --kb data/statpearls_kb.json \
    --json > results/pubmedqa_results.json
```

### Step 4 — Add GPT-4 Baseline (Optional)

Run GPT-4 zero-shot on the same question set and save results to a CSV:

```
question_id,predicted_condition,confidence_score,is_correct
medqa-1,Community-Acquired Pneumonia,0.82,1
medqa-2,Influenza,0.74,0
```

Then run:

```bash
python eval/benchmark_runner.py \
    --dataset-file /path/to/medqa/questions/US/test.jsonl \
    --dataset-format medqa \
    --kb data/statpearls_kb.json \
    --gpt4-outputs-file results/gpt4_outputs.csv \
    --json > results/medqa_results_with_gpt4.json
```

---

## Notes

All results must be on the **official test splits** listed above.
95% confidence intervals are computed with 1,000 bootstrap resamples (seed = 42) using
`compute_bootstrap_ci` in `eval/benchmark_runner.py`.

> **Do NOT use `--demo` mode.** Demo results are circular (questions written to match
> the 33-entry KB) and cannot be reported as benchmarks.  See the module docstring
> in `eval/benchmark_runner.py` for a full explanation.

---

## References

- Jin, D. et al. (2021). What disease does this patient have? MedQA. *Applied Sciences*, 11(14), 6421.
- Jin, Q. et al. (2019). PubMedQA. *EMNLP 2019*. https://arxiv.org/abs/1909.06146
- Wu, S. et al. (2024). MedRAG. *arXiv*:2402.13178.
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
