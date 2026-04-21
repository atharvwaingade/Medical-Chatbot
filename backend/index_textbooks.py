"""
index_textbooks.py  -  Build KB from MedRAG textbook JSONL chunks
==================================================================
Usage (run from anywhere):
    python backend/index_textbooks.py --chunk-dir path/to/textbooks_hf/chunk
    python backend/index_textbooks.py --chunk-dir path/to/textbooks_hf/chunk --output backend/data/textbooks_kb.json
    python backend/index_textbooks.py --chunk-dir path/to/textbooks_hf/chunk --max-entries 10000
"""

import argparse
import glob
import json
import os
import sys

_ORGAN_KEYWORDS = {
    "Cardiovascular":       ["cardiac","heart","coronary","myocardial","hypertension","vascular","aortic"],
    "Respiratory":          ["lung","pulmonary","pneumonia","asthma","bronchitis","copd","respiratory"],
    "Neurological":         ["stroke","seizure","epilepsy","migraine","dementia","parkinson","meningitis"],
    "Gastrointestinal":     ["gastro","bowel","colon","hepatic","liver","pancreatic","peptic","ulcer","crohn"],
    "Metabolic/Endocrine":  ["diabetes","thyroid","adrenal","metabolic","hyperlipid","insulin","hormone"],
    "Urinary/Renal":        ["renal","kidney","urinary","nephro","bladder","glomerulo"],
    "Musculoskeletal":      ["arthritis","osteo","bone","fracture","joint","muscle","rheumat"],
    "Psychiatric":          ["depression","anxiety","bipolar","schizophreni","ptsd","psychiatric","mental"],
    "Infectious":           ["infection","bacterial","viral","fungal","sepsis","tuberculosis","influenza"],
    "Oncological":          ["cancer","tumor","carcinoma","lymphoma","leukemia","malignant","neoplasm"],
}
_FALLBACK_SYMPTOMS = [
    "fever","cough","dyspnoea","shortness of breath","chest pain","nausea","vomiting",
    "diarrhoea","fatigue","weakness","headache","dizziness","rash","swelling","pain",
    "palpitations","weight loss","loss of appetite","jaundice","tachycardia",
]
_BOOK_TIERS = {
    "InternalMed_Harrison": 3, "Pathology_Robbins": 3, "Pharmacology_Katzung": 3,
    "First_Aid_Step1": 3, "First_Aid_Step2": 3, "Physiology_Levy": 3,
    "Biochemistry_Lippincott": 3, "Anatomy_Gray": 3, "Neurology_Adams": 3,
    "Pediatrics_Nelson": 3, "Surgery_Schwartz": 3, "Gynecology_Novak": 3,
    "Obstentrics_Williams": 3, "Histology_Ross": 3, "Immunology_Janeway": 3,
    "Cell_Biology_Alberts": 4, "Pathoma_Husain": 3, "Psichiatry_DSM-5": 3,
}

def infer_organ(text):
    tl = text.lower()
    scores = {s: sum(tl.count(k) for k in kws) for s, kws in _ORGAN_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "General"

def extract_symptoms(text):
    tl = text.lower()
    return [s for s in _FALLBACK_SYMPTOMS if s in tl]

def infer_prevalence(text):
    tl = text.lower()
    if any(p in tl for p in ["very common","highly prevalent","affects millions"]): return "very common"
    if any(p in tl for p in ["common","prevalent","frequently"]):                   return "common"
    if any(p in tl for p in ["uncommon","relatively rare","infrequent"]):           return "uncommon"
    if any(p in tl for p in ["rare","unusual","seldom"]):                           return "rare"
    return "common"

def main():
    parser = argparse.ArgumentParser(
        description="Index MedRAG textbook JSONL chunks into KB format",
    )
    parser.add_argument("--chunk-dir", required=True,
        help="Path to directory containing .jsonl chunk files from MedRAG/textbooks dataset.")
    parser.add_argument("--output", default=None,
        help="Output path for KB JSON (default: <script_dir>/data/textbooks_kb.json).")
    parser.add_argument("--max-entries", type=int, default=None,
        help="Maximum number of entries to index (useful for quick tests).")
    args = parser.parse_args()

    chunk_dir = os.path.abspath(args.chunk_dir)
    if not os.path.isdir(chunk_dir):
        print(f"ERROR: --chunk-dir not found: {chunk_dir}")
        sys.exit(1)

    files = sorted(glob.glob(os.path.join(chunk_dir, "*.jsonl")))
    if not files:
        print(f"ERROR: No .jsonl files found in {chunk_dir}")
        sys.exit(1)

    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, "data", "textbooks_kb.json")

    print(f"Found {len(files)} JSONL files:")
    for f in files:
        size_mb = os.path.getsize(f) / 1024 / 1024
        print(f"  {os.path.basename(f):45s} {size_mb:6.1f} MB")

    entries = []
    seen_ids = set()
    done = False

    for fpath in files:
        if done:
            break
        book_name = os.path.splitext(os.path.basename(fpath))[0]
        tier = _BOOK_TIERS.get(book_name, 3)
        book_entries = 0

        with open(fpath, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                chunk_id = obj.get("id", "")
                content  = obj.get("content", obj.get("contents", ""))
                if not content or not chunk_id or chunk_id in seen_ids:
                    continue
                seen_ids.add(chunk_id)

                entries.append({
                    "condition":     chunk_id.replace("_", " ").strip(),
                    "description":   content[:500] + ("..." if len(content) > 500 else ""),
                    "symptoms":      extract_symptoms(content),
                    "organ_system":  infer_organ(content),
                    "prevalence":    infer_prevalence(content),
                    "source":        book_name,
                    "evidence_tier": tier,
                    "source_url":    "",
                    "verified":      True,
                })
                book_entries += 1

                if args.max_entries and len(entries) >= args.max_entries:
                    done = True
                    break

        print(f"  Indexed {book_entries:6,} chunks from {book_name}" +
              (" [limit reached]" if done else ""))

    print(f"\nTotal entries: {len(entries):,}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(entries, fh, ensure_ascii=False)
    print(f"Written to {output_path}")
    print(f"\nNow run:")
    print(f"  python run_benchmark.py --pubmedqa --kb {output_path}")

if __name__ == "__main__":
    main()
