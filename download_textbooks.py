# download_textbooks.py - Download MedRAG textbooks dataset from HuggingFace
# Run from repo root: python download_textbooks.py
# Requires: pip install huggingface_hub

import os
import sys

def main():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed.")
        print("Run: pip install huggingface_hub")
        sys.exit(1)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "textbooks_hf")
    print("Downloading MedRAG/textbooks dataset (~200MB)...")
    print("This may take a few minutes.")
    try:
        path = snapshot_download(
            repo_id="MedRAG/textbooks",
            repo_type="dataset",
            local_dir=out_dir,
        )
        print(f"Downloaded to: {path}")
        print("Next step: python backend/index_textbooks.py --chunk-dir textbooks_hf/chunk")
    except Exception as e:
        print(f"Download failed: {e}")
        print("\nAlternative: clone manually:")
        print("  git clone https://huggingface.co/datasets/MedRAG/textbooks textbooks_hf")
        print("  cd textbooks_hf && git lfs pull && cd ..")
        sys.exit(1)

if __name__ == "__main__":
    main()
