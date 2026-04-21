#!/usr/bin/env python3
"""
PubMedQA Dataset Downloader
============================
Downloads the official PubMedQA labeled dataset (ori_pqal.json, 1,000 questions)
from GitHub or Hugging Face mirror.

Usage
-----
    cd Medical-Chatbot          # repo root
    python backend/eval/download_pubmedqa.py
    # -> saves to backend/data/pubmedqa/ori_pqal.json

Then run the benchmark:
    python run_benchmark.py --pubmedqa
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

_SOURCES = [
    "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json",
    "https://media.githubusercontent.com/media/pubmedqa/pubmedqa/master/data/ori_pqal.json",
]

_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "pubmedqa"
_OUTPUT_FILE = _OUTPUT_DIR / "ori_pqal.json"


def download(dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for url in _SOURCES:
        print(f"Trying: {url}")
        try:
            urllib.request.urlretrieve(url, dest, reporthook=_progress)
            print("\nDownload complete.")
            return True
        except Exception as exc:
            print(f"\n  Failed: {exc}")
    return False


def _progress(count: int, block_size: int, total_size: int) -> None:
    if total_size > 0:
        pct = min(100, count * block_size * 100 // total_size)
        bar = "X" * (pct // 5) + "." * (20 - pct // 5)
        print(f"\r  [{bar}] {pct}%", end="", flush=True)


def validate(path: Path) -> None:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    n = len(data)
    sample = next(iter(data.values()))
    required = {"QUESTION", "CONTEXTS", "final_decision"}
    missing = required - set(sample.keys())
    if missing:
        print(f"WARNING: Sample entry is missing fields: {missing}", file=sys.stderr)
    yes   = sum(1 for v in data.values() if v.get("final_decision") == "yes")
    no    = sum(1 for v in data.values() if v.get("final_decision") == "no")
    maybe = sum(1 for v in data.values() if v.get("final_decision") == "maybe")
    print(f"\nValidation passed: {n} questions loaded.")
    print(f"  Label distribution -- yes: {yes}, no: {no}, maybe: {maybe}")
    print(f"\nReady to run:")
    print(f"  python run_benchmark.py --pubmedqa")


def print_manual_instructions() -> None:
    print("\n" + "="*60)
    print("MANUAL DOWNLOAD -- automatic download failed.")
    print("="*60)
    print("\nOption A -- Git clone (recommended):")
    print("  git clone https://github.com/pubmedqa/pubmedqa.git")
    print("  Copy pubmedqa/data/ori_pqal.json  to:")
    print("    backend/data/pubmedqa/ori_pqal.json")
    print("\nOption B -- Browser download:")
    print("  1. Visit: https://github.com/pubmedqa/pubmedqa/tree/master/data")
    print("  2. Click ori_pqal.json -> Raw -> Save As")
    print("  3. Save to: backend/data/pubmedqa/ori_pqal.json")
    print("\nThen run:")
    print("  python run_benchmark.py --pubmedqa")
    print("="*60)


def main() -> None:
    if _OUTPUT_FILE.exists():
        print(f"Dataset already exists at {_OUTPUT_FILE}")
        print("Validating existing file...")
        validate(_OUTPUT_FILE)
        return

    print("Downloading PubMedQA labeled dataset...")
    success = download(_OUTPUT_FILE)
    if not success:
        print_manual_instructions()
        sys.exit(1)
    validate(_OUTPUT_FILE)


if __name__ == "__main__":
    main()
