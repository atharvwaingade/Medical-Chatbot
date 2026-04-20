from __future__ import annotations


class Embedder:
    def __init__(self):
        self.available = False
        self.model = None
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.available = True
        except Exception:
            self.available = False

    def encode(self, texts: list[str]) -> list[list[float]]:
        if self.available and self.model is not None:
            return self.model.encode(texts).tolist()
        return [[float(hash(token) % 997) for token in self._tokens(text)[:24]] for text in texts]

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [t.strip(".,!?;:").lower() for t in text.split() if t.strip()]
