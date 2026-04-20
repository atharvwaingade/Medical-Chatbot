from __future__ import annotations

import math
from collections import Counter


class Retriever:
    """
    BM25 retriever (Robertson et al., 1995).

    The inverted-term index and IDF values are computed once at construction
    time so that every query runs in O(|query_terms| * |docs|) arithmetic — no
    file I/O or string processing per request.

    Hyperparameters k1=1.5 and b=0.75 are the standard BM25 defaults validated
    across many IR benchmarks (Trotman et al., 2014).
    """

    K1: float = 1.5
    B: float = 0.75

    def __init__(self, entries: list[dict]) -> None:
        self.entries = entries
        self._index: list[Counter] = []
        self._doc_lengths: list[int] = []
        self._avg_dl: float = 1.0
        self._df: Counter = Counter()
        self._n: int = len(entries)
        self._build_index()

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase whitespace-split tokens with punctuation stripped."""
        return [t.strip(".,!?;:()[]\"'").lower() for t in text.split() if t.strip()]

    def _entry_text(self, entry: dict) -> str:
        """
        Build a weighted pseudo-document for an entry.

        Condition name and symptoms are repeated to increase their term
        frequency, giving them more influence in BM25 scoring.
        """
        condition = entry.get("condition", "")
        symptoms = entry.get("symptoms", [])
        symptoms_str = " ".join(symptoms)
        return " ".join(
            [
                condition,
                condition,              # 2× weight
                entry.get("explanation", ""),
                symptoms_str,
                symptoms_str,           # 2× weight
                " ".join(entry.get("warnings", [])),
            ]
        )

    def _build_index(self) -> None:
        if not self.entries:
            return
        total_len = 0
        for entry in self.entries:
            tokens = self._tokenize(self._entry_text(entry))
            tf = Counter(tokens)
            self._index.append(tf)
            self._doc_lengths.append(len(tokens))
            total_len += len(tokens)
            for term in tf:
                self._df[term] += 1
        self._avg_dl = total_len / len(self.entries) if self.entries else 1.0

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _idf(self, term: str) -> float:
        """
        Smooth IDF as per Robertson & Zaragoza (2009):
          IDF(t) = log( (N - n(t) + 0.5) / (n(t) + 0.5) + 1 )
        The +1 offset ensures IDF > 0 even for terms present in all documents.
        """
        n_t = self._df.get(term, 0)
        return math.log((self._n - n_t + 0.5) / (n_t + 0.5) + 1.0)

    def _bm25_score(self, query_terms: list[str], doc_idx: int) -> float:
        tf_d = self._index[doc_idx]
        dl = self._doc_lengths[doc_idx]
        score = 0.0
        for term in query_terms:
            f = tf_d.get(term, 0)
            if f == 0:
                continue
            idf = self._idf(term)
            tf_norm = f * (self.K1 + 1) / (
                f + self.K1 * (1.0 - self.B + self.B * dl / self._avg_dl)
            )
            score += idf * tf_norm
        return score

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """Return the top-k most relevant entries (entries only, no scores)."""
        return [entry for _, entry in self.retrieve_scored(query, top_k)]

    def retrieve_scored(self, query: str, top_k: int = 3) -> list[tuple[float, dict]]:
        """Return ``[(score, entry), ...]`` sorted descending by BM25 score."""
        if not self.entries:
            return []

        q_terms = self._tokenize(query)
        if not q_terms:
            return [(0.0, self.entries[0])]

        scored = [
            (self._bm25_score(q_terms, i), self.entries[i])
            for i in range(len(self.entries))
        ]
        ranked = sorted(scored, key=lambda x: x[0], reverse=True)

        positive = [(s, e) for s, e in ranked if s > 0.0][:top_k]
        if positive:
            return positive
        # Fallback: return single best match even with score 0
        return [(ranked[0][0], ranked[0][1])] if ranked else []
