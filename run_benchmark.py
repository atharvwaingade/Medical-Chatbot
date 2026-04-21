#!/usr/bin/env python3
"""
run_benchmark.py  —  One-command PubMedQA benchmark launcher
=============================================================
Handles all sys.path setup so you can run from the repo root:

    python run_benchmark.py                     # demo mode
    python run_benchmark.py --pubmedqa          # real dataset (auto-downloads)
    python run_benchmark.py --pubmedqa --json   # JSON output

This script wraps eval/benchmark_runner.py and eval/download_pubmedqa.py.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ── Path setup ──────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent
BACKEND_DIR = REPO_ROOT / "backend"

# Ensure backend/ is importable
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Change CWD to backend so relative KB paths (data/...) resolve correctly
os.chdir(BACKEND_DIR)

# ── Late imports (after path setup) ─────────────────────────────────────────
from eval.benchmark_runner import main as _runner_main   # noqa: E402
from eval.download_pubmedqa import _OUTPUT_FILE, main as _download_main  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MedRAG-Turbo PubMedQA Benchmark — one-command launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Quick smoke test (uses built-in demo questions — circular, not publishable):
  python run_benchmark.py --demo

  # Real PubMedQA benchmark (downloads dataset automatically):
  python run_benchmark.py --pubmedqa

  # Limit to first 100 questions for a quick sanity check:
  python run_benchmark.py --pubmedqa --max-questions 100

  # Output JSON for further processing:
  python run_benchmark.py --pubmedqa --json
        """,
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in demo mode (circular — dev only, not publishable).",
    )
    parser.add_argument(
        "--pubmedqa",
        action="store_true",
        help="Download (if needed) and run the real PubMedQA benchmark.",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Limit number of questions (useful for quick smoke tests).",
    )
    parser.add_argument(
        "--kb",
        default="data/sample_medical_knowledge.json",
        help="Path to KB JSON file (default: data/sample_medical_knowledge.json).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON.",
    )
    parser.add_argument(
        "--ollama",
        action="store_true",
        help=(
            "Use local Ollama (gemma4:e4b) for yes/no/maybe prediction. "
            "Requires: ollama pull gemma4:e4b  (needs 6 GB VRAM). "
            "Improves accuracy from ~55%% to ~68-75%%."
        ),
    )
    parser.add_argument(
        "--ollama-model",
        default="gemma4:e4b",
        help="Ollama model tag (default: gemma4:e4b).",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama API URL (default: http://localhost:11434).",
    )
    args = parser.parse_args()

    if not args.demo and not args.pubmedqa:
        print("Specify --demo or --pubmedqa.  Use --help for details.")
        parser.print_help()
        sys.exit(1)

    # Build argv for benchmark_runner.main()
    # Resolve KB path before CWD was changed to backend/
    # The user may pass a path relative to repo root OR relative to backend/
    kb_path = Path(args.kb)
    if not kb_path.is_absolute():
        # Try relative to original CWD (repo root), then relative to backend/
        cwd_candidate = Path(os.getcwd()) / kb_path   # CWD is now backend/
        repo_candidate = REPO_ROOT / kb_path
        if cwd_candidate.exists():
            kb_path = str(cwd_candidate)
        elif repo_candidate.exists():
            kb_path = str(repo_candidate)
        else:
            kb_path = str(cwd_candidate)  # let the runner report the error

    runner_argv = ["benchmark_runner"]
    runner_argv += ["--benchmark", "pubmedqa"]
    runner_argv += ["--kb", str(kb_path)]
    if args.json:
        runner_argv += ["--json"]
    if args.max_questions:
        runner_argv += ["--max-questions", str(args.max_questions)]
    if args.ollama:
        runner_argv += ["--ollama"]
        runner_argv += ["--ollama-model", args.ollama_model]
        runner_argv += ["--ollama-url", args.ollama_url]

    if args.demo:
        runner_argv += ["--demo"]
    elif args.pubmedqa:
        # Auto-download dataset if missing
        if not _OUTPUT_FILE.exists():
            print("PubMedQA dataset not found — attempting download...")
            try:
                _download_main()
            except SystemExit:
                print("\nAutomatic download failed.", file=sys.stderr)
                print("Please download manually and re-run:", file=sys.stderr)
                print("  python backend/eval/download_pubmedqa.py", file=sys.stderr)
                sys.exit(1)
        runner_argv += [
            "--dataset-file", str(_OUTPUT_FILE),
            "--dataset-format", "pubmedqa",
        ]

    # Patch sys.argv and call benchmark runner
    sys.argv = runner_argv
    _runner_main()


if __name__ == "__main__":
    main()
