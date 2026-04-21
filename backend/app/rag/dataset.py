import json
import os
from pathlib import Path


class MedicalDataset:
    def __init__(self, path: str):
        # Resolve relative paths against the backend/ directory so the dataset
        # can be loaded regardless of the working directory.
        p = Path(path)
        if not p.is_absolute() and not p.exists():
            # Try relative to this file's parent (app/) → parent (backend/)
            backend_dir = Path(__file__).resolve().parent.parent.parent
            candidate = backend_dir / path
            if candidate.exists():
                p = candidate
            else:
                # Also try relative to cwd
                cwd_candidate = Path(os.getcwd()) / path
                if cwd_candidate.exists():
                    p = cwd_candidate
        self.path = p
        self.entries = self._load()

    def _load(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # Accept entries that are explicitly verified=True OR that come from
        # kb_indexer (which does not set the verified flag — treat absence as True).
        return [
            entry for entry in data
            if entry.get("verified", True)  # default True for kb_indexer output
        ]
