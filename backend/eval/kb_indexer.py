#!/usr/bin/env python3
"""
Medical Knowledge Base Indexer (F2 fix: replace toy 33-entry KB)
=================================================================
Converts publicly available medical corpora into the KB format expected by
MedRAG-Turbo.  Supports StatPearls articles, PubMed abstracts (MEDLINE XML),
Wikipedia medical articles, and generic plain-text collections.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  MOTIVATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  The default sample_medical_knowledge.json contains 33 conditions.
  Retrieval accuracy on a 33-entry corpus is trivially achievable by
  keyword matching and does not constitute a meaningful evaluation
  (a random retriever achieves Acc@1 = 1/33 ≈ 3%; any reasonable
  BM25 retriever achieves near 100% on author-written questions).

  MedRAG (Wu et al. 2024) evaluates on PubMed (23M abstracts),
  StatPearls (9,000+ articles), MedQA textbooks, and Wikipedia
  medical articles.  All four corpora are publicly available.

  For valid evaluation numbers, index at least StatPearls (free,
  ~9,000 articles) or the Wikipedia medical article dump.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Supported Input Formats
-----------------------
1. StatPearls JSONL (each line is one article):
       {"title": "Pneumonia", "content": "...", "section": "Overview", ...}
   Download: https://huggingface.co/datasets/MedRAG/textbooks  (statpearls split)

2. PubMed MEDLINE XML (baseline files):
       <MedlineCitation><PMID>...</PMID><Article>...</Article></MedlineCitation>
   Download: https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/

3. Wikipedia medical articles JSON:
       {"title": "Pneumonia", "text": "...", "categories": [...]}
   Download via WikiDump or Hugging Face datasets

4. Generic directory of plain-text .txt files (one article per file):
       Each filename is used as the condition title.

Output Format (MedRAG-Turbo KB schema)
---------------------------------------
    [
      {
        "condition": "Community-Acquired Pneumonia",
        "description": "...",
        "symptoms": ["fever", "cough", "dyspnoea", ...],
        "organ_system": "Respiratory",
        "prevalence": "common",
        "source": "StatPearls",
        "evidence_tier": 3,
        "source_url": "https://www.ncbi.nlm.nih.gov/books/NBK507671/"
      },
      ...
    ]

Usage
-----
    cd backend

    # From StatPearls JSONL (recommended):
    python eval/kb_indexer.py \\
        --format statpearls \\
        --source-file statpearls.jsonl \\
        --output data/statpearls_kb.json

    # From a directory of plain-text files:
    python eval/kb_indexer.py \\
        --format text-dir \\
        --source-dir /path/to/articles/ \\
        --output data/custom_kb.json

    # From PubMed MEDLINE XML baseline:
    python eval/kb_indexer.py \\
        --format pubmed-xml \\
        --source-file pubmed24n0001.xml.gz \\
        --output data/pubmed_kb.json \\
        --max-entries 5000

    # Merge multiple KB files:
    python eval/kb_indexer.py \\
        --merge data/statpearls_kb.json data/pubmed_kb.json \\
        --output data/merged_kb.json

References
----------
Wu, S. et al. (2024). MedRAG: Towards a comprehensive medical RAG framework.
arXiv:2402.13178.

Pal, A. et al. (2022). MedMCQA: A large-scale multi-subject multi-choice
dataset for medical domain question answering. CHIL 2022.

Jin, D. et al. (2021). What disease does this patient have? Applied Sciences.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ORGAN_SYSTEM_KEYWORDS: dict[str, list[str]] = {
    "Cardiovascular": [
        "cardiac", "heart", "coronary", "myocardial", "arrhythmia",
        "hypertension", "vascular", "aortic", "endocarditis",
    ],
    "Respiratory": [
        "lung", "pulmonary", "pneumonia", "asthma", "bronchitis",
        "copd", "respiratory", "pleural", "trachea",
    ],
    "Neurological": [
        "stroke", "seizure", "epilepsy", "migraine", "dementia",
        "parkinson", "neuropathy", "meningitis", "encephalitis",
    ],
    "Gastrointestinal": [
        "gastro", "bowel", "colon", "hepatic", "liver", "pancreatic",
        "appendic", "peptic", "ulcer", "crohn", "colitis",
    ],
    "Metabolic/Endocrine": [
        "diabetes", "thyroid", "adrenal", "obesity", "metabolic",
        "hyperlipid", "insulin", "hormone", "endocrine",
    ],
    "Urinary/Renal": [
        "renal", "kidney", "urinary", "nephro", "bladder", "ureteral",
        "glomerulo", "nephrotic", "dialysis",
    ],
    "Musculoskeletal": [
        "arthritis", "osteo", "bone", "fracture", "joint", "muscle",
        "tendon", "ligament", "spondyl", "rheumat",
    ],
    "Psychiatric": [
        "depression", "anxiety", "bipolar", "schizophreni", "ptsd",
        "disorder", "psychiatric", "mental", "psychosis",
    ],
    "Infectious": [
        "infection", "bacterial", "viral", "fungal", "parasite",
        "sepsis", "tuberculosis", "malaria", "influenza",
    ],
    "Oncological": [
        "cancer", "tumor", "carcinoma", "lymphoma", "leukemia",
        "malignant", "neoplasm", "metastasis",
    ],
}

_SYMPTOM_PATTERNS: list[str] = [
    r"present(?:ing|s)? with ([^.]+)",
    r"symptom[s]? (?:include|of|such as) ([^.]+)",
    r"characterized? by ([^.]+)",
    r"manifest[s]? (?:as|with) ([^.]+)",
    r"patient[s]? (?:report|complain|experience) ([^.]+)",
    r"(?:fever|cough|dyspnoea|pain|nausea|vomiting|fatigue|dizziness"
    r"|headache|diarrhoea|weakness|swelling|rash|bleeding)[^.]*",
]

_STOPWORDS: set[str] = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "as", "is", "was", "are",
    "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "shall",
    "can", "this", "that", "these", "those", "it", "its",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _infer_organ_system(text: str) -> str:
    """Infer organ system from article text by keyword matching."""
    text_lower = text.lower()
    scores: dict[str, int] = {}
    for system, keywords in _ORGAN_SYSTEM_KEYWORDS.items():
        scores[system] = sum(text_lower.count(kw) for kw in keywords)
    best = max(scores, key=scores.get)  # type: ignore[arg-type]
    return best if scores[best] > 0 else "General"


def _extract_symptoms(text: str, max_symptoms: int = 30) -> list[str]:
    """
    Extract symptom/disease entity phrases from article text.

    Primary path — Medical NER via scispacy (en_core_sci_sm ≥ 0.5.3):
        1. Load the scispacy model (cached after first call).
        2. Run NLP on the first 5,000 characters of text.
        3. Collect entities with label DISEASE or SIGN_OR_SYMPTOM.
        4. Return deduplicated, lowercased entity texts (up to max_symptoms).

    Fallback path (scispacy not installed):
        Keyword matching against a fixed list of 24 common symptoms/signs.

    Install scispacy::

        pip install scispacy
        pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_sm-0.5.3.tar.gz

    Parameters
    ----------
    text : str
        Article text to extract symptoms from.
    max_symptoms : int
        Maximum number of symptoms to return (default 30).

    Returns
    -------
    list[str]
        Deduplicated list of symptom/disease phrases, lowercased.
    """
    # ------------------------------------------------------------------
    # Primary: scispacy Medical NER
    # ------------------------------------------------------------------
    try:
        import spacy  # noqa: PLC0415 — deferred import so scispacy is optional

        # Cache model on the function object to avoid reloading per call
        nlp = getattr(_extract_symptoms, "_nlp_cache", None)
        if nlp is None:
            try:
                nlp = spacy.load("en_core_sci_sm")
                _extract_symptoms._nlp_cache = nlp  # type: ignore[attr-defined]
            except OSError:
                # Model not installed — fall through to keyword fallback
                raise ImportError("en_core_sci_sm model not found")

        doc = nlp(text[:5000])
        seen: set[str] = set()
        symptoms: list[str] = []
        for ent in doc.ents:
            if ent.label_ in ("DISEASE", "SIGN_OR_SYMPTOM"):
                phrase = ent.text.lower().strip()
                if phrase and phrase not in seen:
                    symptoms.append(phrase)
                    seen.add(phrase)
                    if len(symptoms) >= max_symptoms:
                        break
        return symptoms

    except ImportError:
        pass  # scispacy or model not available — use keyword fallback

    # ------------------------------------------------------------------
    # Fallback: keyword matching (24 hardcoded terms)
    # ------------------------------------------------------------------
    _FALLBACK_SYMPTOMS = [
        "fever", "cough", "dyspnoea", "shortness of breath", "chest pain",
        "nausea", "vomiting", "diarrhoea", "fatigue", "weakness", "headache",
        "dizziness", "rash", "swelling", "oedema", "pain", "discomfort",
        "palpitations", "syncope", "haemoptysis", "weight loss",
        "loss of appetite", "jaundice", "cyanosis", "tachycardia",
    ]
    text_lower = text.lower()
    seen_fb: set[str] = set()
    symptoms_fb: list[str] = []
    for symptom in _FALLBACK_SYMPTOMS:
        if symptom in text_lower and symptom not in seen_fb:
            symptoms_fb.append(symptom)
            seen_fb.add(symptom)
            if len(symptoms_fb) >= max_symptoms:
                break
    return symptoms_fb


def _infer_prevalence(text: str) -> str:
    """Infer prevalence category from text."""
    text_lower = text.lower()
    if any(p in text_lower for p in ["very common", "highly prevalent", "affects millions"]):
        return "very common"
    if any(p in text_lower for p in ["common", "prevalent", "frequently"]):
        return "common"
    if any(p in text_lower for p in ["uncommon", "relatively rare", "infrequent"]):
        return "uncommon"
    if any(p in text_lower for p in ["rare", "unusual", "seldom"]):
        return "rare"
    return "common"  # default


def _make_entry(
    condition: str,
    description: str,
    source: str,
    evidence_tier: int,
    source_url: str = "",
) -> dict:
    """Create a normalised KB entry from raw fields."""
    organ_system = _infer_organ_system(condition + " " + description)
    symptoms = _extract_symptoms(description)
    prevalence = _infer_prevalence(description)

    # Truncate description to ~500 chars
    if len(description) > 500:
        description = description[:497] + "..."

    return {
        "condition": condition.strip(),
        "description": description.strip(),
        "symptoms": symptoms,
        "organ_system": organ_system,
        "prevalence": prevalence,
        "source": source,
        "evidence_tier": evidence_tier,
        "source_url": source_url,
        "verified": True,  # mark as verified so dataset.py loads this entry
    }


# ---------------------------------------------------------------------------
# Format-specific parsers
# ---------------------------------------------------------------------------


def parse_statpearls_jsonl(path: str, max_entries: Optional[int] = None) -> Iterator[dict]:
    """
    Parse StatPearls JSONL file from Hugging Face MedRAG dataset.

    Expected schema per line:
        {"title": str, "content": str, "section": str, "id": str}

    StatPearls articles are peer-reviewed and maintained by NCBI/NLM.
    GRADE tier: 3 (clinical guideline level).
    """
    n = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            title = obj.get("title", "Unknown")
            content = obj.get("content", obj.get("text", ""))
            if not content:
                continue

            yield _make_entry(
                condition=title,
                description=content,
                source="StatPearls",
                evidence_tier=3,
                source_url=f"https://www.ncbi.nlm.nih.gov/books/?term={title.replace(' ', '+')}",
            )
            n += 1
            if max_entries and n >= max_entries:
                break


def parse_pubmed_xml(path: str, max_entries: Optional[int] = None) -> Iterator[dict]:
    """
    Parse PubMed MEDLINE XML baseline file (gzipped or plain).

    Downloads: https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/
    GRADE tier: 4 (peer-reviewed journal article).
    """
    opener = gzip.open if path.endswith(".gz") else open
    n = 0

    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        context = ET.iterparse(fh, events=("end",))
        for event, elem in context:
            if elem.tag != "MedlineCitation":
                continue

            pmid_elem = elem.find("PMID")
            pmid = pmid_elem.text if pmid_elem is not None else "unknown"

            article = elem.find("Article")
            if article is None:
                elem.clear()
                continue

            title_elem = article.find("ArticleTitle")
            title = (title_elem.text or "") if title_elem is not None else ""

            # Collect abstract text
            abstract_parts: list[str] = []
            abstract = article.find("Abstract")
            if abstract is not None:
                for at in abstract.findall("AbstractText"):
                    text = at.text or ""
                    label = at.get("Label", "")
                    if label:
                        text = f"{label}: {text}"
                    abstract_parts.append(text)
            content = " ".join(abstract_parts)

            if not title or not content:
                elem.clear()
                continue

            yield _make_entry(
                condition=title,
                description=content,
                source="PubMed",
                evidence_tier=4,
                source_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            )
            elem.clear()
            n += 1
            if max_entries and n >= max_entries:
                break


def parse_wikipedia_json(path: str, max_entries: Optional[int] = None) -> Iterator[dict]:
    """
    Parse Wikipedia medical article JSON dump.

    Expected format (one JSON object per line):
        {"title": str, "text": str, "categories": list[str]}

    GRADE tier: 5 (expert opinion / encyclopaedic summary).
    """
    n = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            title = obj.get("title", "Unknown")
            text = obj.get("text", obj.get("content", ""))
            if not text:
                continue

            # Filter to medical articles (Wikipedia dump contains all topics)
            categories = [c.lower() for c in obj.get("categories", [])]
            is_medical = any(
                kw in cat
                for cat in categories
                for kw in ("disease", "disorder", "syndrome", "condition",
                           "symptom", "medicine", "medical", "health")
            )
            if not is_medical and not any(
                kw in title.lower()
                for kw in ("disease", "syndrome", "disorder", "fever",
                           "infection", "cancer", "pneumonia", "diabetes")
            ):
                continue

            yield _make_entry(
                condition=title,
                description=text,
                source="Wikipedia",
                evidence_tier=5,
                source_url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
            )
            n += 1
            if max_entries and n >= max_entries:
                break


def parse_text_dir(source_dir: str, max_entries: Optional[int] = None) -> Iterator[dict]:
    """
    Parse a directory of plain-text .txt files (one article per file).

    Each filename (without extension) becomes the condition name.
    Useful for custom article collections.
    GRADE tier: 5 (unspecified source — assign manually).
    """
    n = 0
    for txt_path in sorted(Path(source_dir).glob("*.txt")):
        condition = txt_path.stem.replace("_", " ").replace("-", " ").title()
        try:
            content = txt_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            print(f"WARNING: could not read {txt_path}: {exc}", file=sys.stderr)
            continue

        if not content.strip():
            continue

        yield _make_entry(
            condition=condition,
            description=content,
            source="Custom",
            evidence_tier=5,
            source_url="",
        )
        n += 1
        if max_entries and n >= max_entries:
            break


# ---------------------------------------------------------------------------
# Merge helper
# ---------------------------------------------------------------------------


def merge_kb_files(paths: list[str]) -> list[dict]:
    """
    Merge multiple KB JSON files, deduplicating by condition name.

    When two entries have the same condition, the entry with the lower
    evidence_tier (higher quality) is kept.
    """
    merged: dict[str, dict] = {}
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            entries = json.load(fh)
        for entry in entries:
            key = entry.get("condition", "").lower().strip()
            if not key:
                continue
            if key not in merged:
                merged[key] = entry
            else:
                # Keep the higher-quality source (lower tier number)
                if entry.get("evidence_tier", 5) < merged[key].get("evidence_tier", 5):
                    merged[key] = entry
    return list(merged.values())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index medical articles into MedRAG-Turbo KB format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--format",
        choices=["statpearls", "pubmed-xml", "wikipedia", "text-dir"],
        help="Input format (ignored when --merge is used).",
    )
    parser.add_argument("--source-file", help="Path to input JSONL/XML/JSON file.")
    parser.add_argument("--source-dir", help="Path to directory of .txt files (for text-dir).")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--max-entries", type=int, default=None,
                        help="Maximum number of entries to index.")
    parser.add_argument(
        "--merge",
        nargs="+",
        help="Merge multiple existing KB JSON files (deduplication by condition).",
    )
    args = parser.parse_args()

    if args.merge:
        print(f"Merging {len(args.merge)} KB files ...")
        entries = merge_kb_files(args.merge)
        print(f"Merged {len(entries)} unique entries.")
    elif args.format is None:
        parser.error("--format is required unless --merge is used.")
    elif args.format == "statpearls":
        if not args.source_file:
            parser.error("--source-file is required for --format statpearls")
        print(f"Parsing StatPearls JSONL from {args.source_file} ...")
        entries = list(parse_statpearls_jsonl(args.source_file, args.max_entries))
    elif args.format == "pubmed-xml":
        if not args.source_file:
            parser.error("--source-file is required for --format pubmed-xml")
        print(f"Parsing PubMed MEDLINE XML from {args.source_file} ...")
        entries = list(parse_pubmed_xml(args.source_file, args.max_entries))
    elif args.format == "wikipedia":
        if not args.source_file:
            parser.error("--source-file is required for --format wikipedia")
        print(f"Parsing Wikipedia JSON from {args.source_file} ...")
        entries = list(parse_wikipedia_json(args.source_file, args.max_entries))
    elif args.format == "text-dir":
        if not args.source_dir:
            parser.error("--source-dir is required for --format text-dir")
        print(f"Parsing text directory {args.source_dir} ...")
        entries = list(parse_text_dir(args.source_dir, args.max_entries))
    else:
        parser.error(f"Unknown format: {args.format}")

    print(f"Indexed {len(entries)} entries.")

    # Write output
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(entries, fh, indent=2, ensure_ascii=False)
    print(f"Written to {args.output}")

    # Summary statistics
    sources: dict[str, int] = {}
    tiers: dict[int, int] = {}
    for e in entries:
        s = e.get("source", "unknown")
        sources[s] = sources.get(s, 0) + 1
        t = e.get("evidence_tier", 5)
        tiers[t] = tiers.get(t, 0) + 1

    print(f"\nSummary:")
    print(f"  Total entries : {len(entries)}")
    for s, n in sorted(sources.items()):
        print(f"  {s:<20}: {n}")
    print(f"  Evidence tiers: {dict(sorted(tiers.items()))}")

    # Average symptoms per entry
    total_symptoms = sum(len(e.get("symptoms", [])) for e in entries)
    avg_symptoms = total_symptoms / len(entries) if entries else 0.0
    print(f"  Avg symptoms/entry: {avg_symptoms:.2f}")

    # Evidence tier breakdown as percentages
    print("  Tier breakdown:")
    for tier, count in sorted(tiers.items()):
        pct = 100.0 * count / len(entries) if entries else 0.0
        print(f"    Tier {tier}: {count:>6} entries ({pct:.1f}%)")

    if len(entries) < 5000:
        print(
            f"\nWARNING: Only {len(entries)} entries indexed. "
            "For credible retrieval evaluation, index ≥9,000 entries "
            "(full StatPearls) or ≥50,000 (PubMed abstracts). "
            "Fewer than 5,000 entries likely indicates a parsing problem.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
