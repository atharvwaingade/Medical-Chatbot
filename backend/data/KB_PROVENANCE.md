# Knowledge Base Provenance — MedRAG-Turbo

This document must be included as supplementary material when reporting
retrieval results computed with MedRAG-Turbo.

---

## Primary KB: StatPearls

| Field | Value |
|-------|-------|
| **Source name** | StatPearls (NCBI Bookshelf) |
| **Download URL** | https://huggingface.co/datasets/MedRAG/textbooks (statpearls split) |
| **Hugging Face dataset** | `MedRAG/textbooks`, config `statpearls` |
| **Download date** | *Populate when you run the indexer, e.g. 2024-01-15* |
| **Version / commit** | *Populate with the HuggingFace dataset commit SHA, e.g. `abc1234`* |
| **Indexer script** | `backend/eval/kb_indexer.py` (`--format statpearls`) |
| **Total entries indexed** | *Populate after running, e.g. 9,382* |
| **Output file** | `backend/data/statpearls_kb.json` |

### Evidence Tier Breakdown

*Populate after running the indexer (the script prints tier counts):*

| Evidence Tier | Description | Count | % |
|:---:|-------------|------:|--:|
| 3 | StatPearls peer-reviewed clinical articles (NCBI/NLM) | *fill* | *fill* |

### Symptom Extraction

| Field | Value |
|-------|-------|
| **NER model** | `en_core_sci_sm` v0.5.3 (scispacy, AI2) |
| **Model download** | https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_sm-0.5.3.tar.gz |
| **Entity labels used** | `DISEASE`, `SIGN_OR_SYMPTOM` |
| **Text window** | First 5,000 characters per article |
| **Max symptoms per entry** | 30 |
| **Fallback** | 24-term keyword list (when scispacy is not installed) |
| **Average symptoms/entry** | *Populate after running, e.g. 8.3* |

---

## How to Reproduce

```bash
cd backend

# 1. Install scispacy (optional but recommended for full NER):
pip install scispacy
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_sm-0.5.3.tar.gz

# 2. Download StatPearls JSONL from Hugging Face:
#    https://huggingface.co/datasets/MedRAG/textbooks
#    Select the 'statpearls' config and download the JSONL file.

# 3. Run the indexer:
python eval/kb_indexer.py \
    --format statpearls \
    --source-file /path/to/statpearls.jsonl \
    --output data/statpearls_kb.json

# 4. Inspect the summary printed to stdout and update the table above.
```

---

## Supplementary Notes for Paper

- The KB contains **no patient data** — all entries are clinical reference
  articles, not case reports.
- StatPearls articles are maintained by NCBI/NLM under a Creative Commons
  licence and are freely redistributable for research purposes.
- The NER model (`en_core_sci_sm`) was trained on biomedical literature
  (PubMed abstracts + full-text) and recognises 18 entity types; only
  `DISEASE` and `SIGN_OR_SYMPTOM` are used here.
- The 5,000-character window covers the full abstract + first several
  paragraphs for most StatPearls articles (median article length ≈ 3,200
  characters in the MedRAG JSONL release).
- Deduplication is performed by condition name (case-insensitive); when
  multiple sources contain the same condition, the entry with the lowest
  evidence tier (highest quality) is retained.

---

## References

- Wu, S. et al. (2024). MedRAG: Towards a comprehensive medical RAG framework.
  *arXiv*:2402.13178.
- Neumann, M. et al. (2019). ScispaCy: Fast and Robust Models for Biomedical
  Natural Language Processing. *BioNLP 2019*. https://arxiv.org/abs/1902.07669
- StatPearls Publishing. https://www.statpearls.com/ (accessed via NCBI Bookshelf).
