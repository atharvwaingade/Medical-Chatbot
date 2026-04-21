# Medical-Chatbot — Complete Setup & Benchmark Guide

## Quick Start (3 commands)

```bash
# 1. Install dependencies
pip install -r backend/requirements.txt

# 2. Quick smoke test (demo mode — works offline, no download needed)
python run_benchmark.py --demo

# 3. Real PubMedQA benchmark (uses bundled ori_pqal.json + bundled textbooks_kb.json)
python run_benchmark.py --pubmedqa --kb backend/data/textbooks_kb.json
```

The PubMedQA dataset (`ori_pqal.json`, 1000 questions) and the textbooks KB
(`textbooks_kb.json`, 125,847 entries from 18 medical textbooks) are **already
bundled** in `backend/data/`. No download needed for the benchmark.

---

## What Each Command Does

| Command | What it runs | KB used |
|---|---|---|
| `python run_benchmark.py --demo` | 15 circular demo questions | 33-entry toy KB |
| `python run_benchmark.py --pubmedqa` | 1000 real PubMedQA questions | 33-entry toy KB |
| `python run_benchmark.py --pubmedqa --kb backend/data/textbooks_kb.json` | 1000 real PubMedQA questions | 125,847-entry textbooks KB |

---

## Expected Results

With the bundled textbooks KB (~125K entries):
- **BM25-only:** ~55–62% accuracy on 1000 PubMedQA questions
- **MedRAG-Turbo:** ~55–65% accuracy

Published baselines (Wu et al. 2024, full StatPearls corpus):
- GPT-4 zero-shot: 78.2%
- MedRAG BM25: 74.5%
- MedRAG full: 79.6%

The gap vs published is expected — the heuristic yes/no predictor in
`_predict_pubmedqa_answer()` does not use an LLM. To reach publication-level
numbers, replace it with a Groq/OpenAI API call using the retrieved context
(see `backend/app/services/groq_client.py`).

---

## Optional: Re-download / Rebuild the Textbooks KB

If `textbooks_kb.json` is missing or you want to rebuild it:

```bash
# Step 1: Download the textbook JSONL files (~200MB)
pip install huggingface_hub
python download_textbooks.py
# Files land in: textbooks_hf/chunk/*.jsonl

# Step 2: Index into KB format
python backend/index_textbooks.py --chunk-dir textbooks_hf/chunk
# Output: backend/data/textbooks_kb.json (125,847 entries)

# Step 3: Run benchmark
python run_benchmark.py --pubmedqa --kb backend/data/textbooks_kb.json
```

---

## Running the API Server

```bash
# Set up .env
cp backend/.env.example backend/.env
# Edit backend/.env and add: GROQ_API_KEY=your_key_here

# Start backend
cd backend
uvicorn app.main:app --reload --port 8000

# Start frontend (separate terminal)
cd frontend
npm install
npm run dev
```

---

## All Benchmark Options

```bash
# Quick smoke test (instant)
python run_benchmark.py --demo

# Real PubMedQA, sample KB (fast)
python run_benchmark.py --pubmedqa

# Real PubMedQA, textbooks KB (best results)
python run_benchmark.py --pubmedqa --kb backend/data/textbooks_kb.json

# First 100 questions only (quick sanity check)
python run_benchmark.py --pubmedqa --kb backend/data/textbooks_kb.json --max-questions 100

# JSON output (for logging / paper tables)
python run_benchmark.py --pubmedqa --kb backend/data/textbooks_kb.json --json > results.json
```

---

## File Layout

```
Medical-Chatbot/
├── run_benchmark.py              # One-command benchmark launcher (run from here)
├── download_textbooks.py         # Download textbook JSONL files
├── SETUP.md                      # This file
├── backend/
│   ├── requirements.txt
│   ├── index_textbooks.py        # Build textbooks_kb.json from JSONL chunks
│   ├── data/
│   │   ├── pubmedqa/ori_pqal.json     # 1000 PubMedQA questions (bundled)
│   │   ├── textbooks_kb.json          # 125,847-entry KB (bundled)
│   │   └── sample_medical_knowledge.json  # 33-entry toy KB
│   ├── app/
│   │   ├── main.py
│   │   ├── rag/
│   │   │   ├── dataset.py
│   │   │   ├── retriever.py
│   │   │   ├── query_processor.py
│   │   │   └── ...
│   │   └── services/groq_client.py
│   └── eval/
│       ├── benchmark_runner.py   # Core benchmark logic
│       ├── kb_indexer.py         # Alternative KB indexer
│       └── download_pubmedqa.py  # PubMedQA downloader
└── frontend/
    └── ...
```
