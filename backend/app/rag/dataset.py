import json
from pathlib import Path


class MedicalDataset:
    def __init__(self, path: str):
        self.path = Path(path)
        self.entries = self._load()

    def _load(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return [entry for entry in data if entry.get("verified", False)]
