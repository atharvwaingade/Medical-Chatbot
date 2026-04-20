from __future__ import annotations

from collections import Counter


class Retriever:
    def __init__(self, entries: list[dict]):
        self.entries = entries

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [t.strip(".,!?;:").lower() for t in text.split() if t.strip()]

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        if not self.entries:
            return []
        q_tokens = Counter(self._tokenize(query))
        scored = []
        for entry in self.entries:
            text = " ".join(
                [
                    entry.get("condition", ""),
                    entry.get("explanation", ""),
                    " ".join(entry.get("symptoms", [])),
                    " ".join(entry.get("warnings", [])),
                ]
            )
            d_tokens = Counter(self._tokenize(text))
            overlap = sum((q_tokens & d_tokens).values())
            scored.append((overlap, entry))

        ranked = sorted(scored, key=lambda x: x[0], reverse=True)
        reranked = [item for score, item in ranked if score > 0][:top_k]
        if reranked:
            return reranked
        return [item for _, item in ranked[:1]]
