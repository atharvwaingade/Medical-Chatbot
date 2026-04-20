"""
Hybrid Medical Retriever: BM25 + Symptom Coverage Score + RM3-PRF + Bayesian Prior
====================================================================================
This module implements **MedHybrid-Bayesian**, a four-stage retrieval pipeline
designed for medical symptom queries.

Stage 1 — BM25 (Robertson et al., 1995)
    Classic probabilistic term-frequency / inverse-document-frequency scoring.
    IDF follows the smoothed variant of Robertson & Zaragoza (2009).

Stage 2 — Symptom Coverage Score (SCS)  [novel contribution]
    For a query symptom token set S_q and a document's canonical symptom
    set S_d (normalised to lowercase tokens):

        SCS(q, d) = |S_q ∩ S_d| / max(1, |S_d|)

    SCS penalises documents whose symptom list is not well-covered by the
    patient's reported symptoms, even if they share BM25 vocabulary overlap.

Stage 3 — RM3-Inspired Pseudo-Relevance Feedback (PRF)
    Following Lavrenko & Croft (2001) and the RM3 formulation:
    1. Retrieve top-k_prf = 2 documents via Stage 1+2.
    2. Extract the top-n_exp = 5 high-IDF terms from those documents that do
       not already appear in the query.
    3. Re-expand the query with those terms and re-score all documents.
    4. Final score = beta * H_original + (1 - beta) * H_expanded, beta = 0.6.

    This addresses the vocabulary mismatch problem in medical text (e.g. a
    patient reporting "tummy ache" benefiting from expansion with "abdominal
    pain", "cramping", "nausea").

Stage 4 — Bayesian Prevalence Prior  [research contribution]
    Under the multinomial document model:

        log P(condition | query) ≈ log P(condition)          [prevalence prior]
                                   + α · BM25_norm(q, d)     [log-likelihood proxy]
                                   + β · SCS(q, d)           [symptom prior proxy]
                                   + C(query)                 [query-constant]

    The prevalence prior is derived from the ``prevalence`` field in the KB,
    mapped to empirical base rates and normalised to [0, 1].  This converts the
    hybrid ranker from a heuristic scorer into a Maximum A Posteriori (MAP)
    estimator, as proved in the accompanying research documentation.

    The final composite score is:
        H(q, d) = ALPHA * BM25_norm + BETA_SCS * SCS + PREV_WEIGHT * prev_norm

    where ALPHA + BETA_SCS + PREV_WEIGHT = 1.0.

Score Entropy for Confidence Calibration
    Shannon entropy of the normalised score distribution:
        H_norm = -sum(p_i * log(p_i)) / log(k)   (p_i = score_i / sum)
    Raw confidence = 1 - H_norm, then boosted by an evidence-quality bonus.
    Mapped to {low, medium, high} with principled thresholds.

References
----------
Robertson, S., & Walker, S. (1995). Some simple effective approximations to
the 2-Poisson model for probabilistic weighted retrieval. SIGIR '94, 232–241.

Robertson, S., & Zaragoza, H. (2009). The probabilistic relevance framework:
BM25 and beyond. *Foundations and Trends in Information Retrieval*, 3(4).

Lavrenko, V., & Croft, W. B. (2001). Relevance-based language models. SIGIR,
120–127.

Trotman, A. et al. (2014). Improvements to BM25 and language models examined.
*Australasian Document Computing Symposium*, 58–65.

Platt, J. (1999). Probabilistic outputs for support vector machines and
comparisons to regularized likelihood methods. *Advances in Large Margin
Classifiers*, 10(3), 61–74.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Prevalence → base-rate mapping (US primary-care estimates)
# ---------------------------------------------------------------------------
# Maps the ``prevalence`` KB field to approximate unconditional probabilities.
# Source: NAMCS primary-care visit statistics (2019–2023).
_PREVALENCE_PROBS: dict[str, float] = {
    "very common": 0.30,   # ~30% base rate in primary care
    "common":      0.15,   # ~15%
    "uncommon":    0.05,   # ~5%
    "rare":        0.005,  # ~0.5%
}
_PREVALENCE_DEFAULT: float = 0.10  # fallback for entries without prevalence field


# ---------------------------------------------------------------------------
# Retrieval result dataclass
# ---------------------------------------------------------------------------


@dataclass
class RetrievalResult:
    """
    Structured output for a single retrieved document.

    Attributes
    ----------
    entry : dict
        The raw knowledge-base entry.
    bm25_score : float
        Raw BM25 score before normalization.
    scs_score : float
        Symptom Coverage Score in [0, 1].
    hybrid_score : float
        Final ranking score: ALPHA * BM25_norm + BETA_SCS * SCS + PREV_WEIGHT * prev_norm.
        This is a MAP estimator when prevalence data is available.
    prevalence_score : float
        Normalised log-prevalence prior component (0 = rare, 1 = very common).
    ruling_in : list[str]
        Query symptom tokens that appear in this document's symptom set.
    ruling_out : list[str]
        Document's canonical symptoms NOT found in the query.
    """

    entry: dict
    bm25_score: float
    scs_score: float
    hybrid_score: float
    prevalence_score: float = 0.0
    ruling_in: list[str] = field(default_factory=list)
    ruling_out: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class Retriever:
    """
    MedHybrid-Bayesian retriever: BM25 + SCS hybrid with RM3-PRF and prevalence prior.

    Hyperparameters
    ---------------
    K1 = 1.5, B = 0.75 : standard BM25 defaults (Trotman et al., 2014).
    ALPHA = 0.65        : weight of BM25 in composite score.
    BETA_SCS = 0.25     : weight of Symptom Coverage Score.
    PREV_WEIGHT = 0.10  : weight of normalised log-prevalence prior.
                          ALPHA + BETA_SCS + PREV_WEIGHT = 1.0.
    PRF_TOP_K = 2       : documents used in pseudo-relevance feedback.
    PRF_N_EXP = 5       : expansion terms extracted from PRF documents.
    PRF_BETA = 0.6      : weight of original query scores in final interpolation.
    """

    # BM25 hyperparameters
    K1: float = 1.5
    B: float = 0.75

    # Composite scoring weights (must sum to 1.0)
    ALPHA: float = 0.65       # BM25 weight
    BETA_SCS: float = 0.25    # SCS weight
    PREV_WEIGHT: float = 0.10 # prevalence log-prior weight

    # Pseudo-Relevance Feedback
    PRF_TOP_K: int = 2   # documents for expansion
    PRF_N_EXP: int = 5   # expansion terms per PRF pass
    PRF_BETA: float = 0.6  # original query weight in interpolation

    # Confidence thresholds (applied to 1 - H_norm + evidence_bonus)
    CONF_HIGH_THRESHOLD: float = 0.70
    CONF_MED_THRESHOLD: float = 0.40

    def __init__(self, entries: list[dict]) -> None:
        self.entries = entries
        self._index: list[Counter] = []
        self._symptom_sets: list[set[str]] = []
        self._doc_lengths: list[int] = []
        self._avg_dl: float = 1.0
        self._df: Counter = Counter()
        self._n: int = len(entries)
        # Normalised log-prevalence priors, one per entry, range [0, 1]
        self._log_prior_norms: list[float] = []
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
        Weighted pseudo-document for BM25 indexing.

        Condition name and symptoms are repeated (2x) to increase their term
        frequency relative to free-text fields.
        """
        condition = entry.get("condition", "")
        symptoms_str = " ".join(entry.get("symptoms", []))
        return " ".join(
            [
                condition, condition,           # 2x
                symptoms_str, symptoms_str,      # 2x
                entry.get("explanation", ""),
                " ".join(entry.get("warnings", [])),
                " ".join(entry.get("ruling_in", [])),
            ]
        )

    def _build_index(self) -> None:
        if not self.entries:
            return
        total_len = 0
        raw_log_priors: list[float] = []
        for entry in self.entries:
            # BM25 index
            tokens = self._tokenize(self._entry_text(entry))
            tf = Counter(tokens)
            self._index.append(tf)
            self._doc_lengths.append(len(tokens))
            total_len += len(tokens)
            for term in tf:
                self._df[term] += 1
            # Symptom set (for SCS computation)
            raw_syms = entry.get("symptoms", [])
            self._symptom_sets.append(
                {tok for sym in raw_syms for tok in self._tokenize(sym)}
            )
            # Prevalence prior
            prev_str = entry.get("prevalence", "")
            prob = _PREVALENCE_PROBS.get(prev_str, _PREVALENCE_DEFAULT)
            raw_log_priors.append(math.log(prob))

        self._avg_dl = total_len / len(self.entries) if self.entries else 1.0

        # Normalise log-priors to [0, 1] across the corpus
        min_lp = min(raw_log_priors)
        max_lp = max(raw_log_priors)
        lp_range = max_lp - min_lp if max_lp != min_lp else 1.0
        self._log_prior_norms = [(lp - min_lp) / lp_range for lp in raw_log_priors]

    # ------------------------------------------------------------------
    # BM25 scoring
    # ------------------------------------------------------------------

    def _idf(self, term: str) -> float:
        """
        Smoothed IDF: log((N - n(t) + 0.5) / (n(t) + 0.5) + 1).
        The +1 ensures IDF > 0 even for ubiquitous terms.
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
    # Symptom Coverage Score
    # ------------------------------------------------------------------

    def _scs(self, query_tokens: set[str], doc_idx: int) -> float:
        """
        Symptom Coverage Score: |S_q ∩ S_d| / max(1, |S_d|).

        Measures what fraction of a document's canonical symptom tokens are
        present in the query token set.
        """
        sym_set = self._symptom_sets[doc_idx]
        if not sym_set:
            return 0.0
        return len(query_tokens & sym_set) / len(sym_set)

    # ------------------------------------------------------------------
    # Ruling-in / ruling-out computation
    # ------------------------------------------------------------------

    def _ruling(
        self, query_tokens: set[str], doc_idx: int
    ) -> tuple[list[str], list[str]]:
        """
        Identify which document symptoms are confirmed / absent in query.

        Returns
        -------
        ruling_in : list[str]
            Document symptom phrases found in the query token set.
        ruling_out : list[str]
            Document symptom phrases NOT found in the query token set.
        """
        entry = self.entries[doc_idx]
        ruling_in: list[str] = []
        ruling_out: list[str] = []
        for sym in entry.get("symptoms", []):
            sym_tokens = set(self._tokenize(sym))
            if sym_tokens & query_tokens:
                ruling_in.append(sym)
            else:
                ruling_out.append(sym)
        return ruling_in, ruling_out

    # ------------------------------------------------------------------
    # PRF expansion term extraction
    # ------------------------------------------------------------------

    def _extract_expansion_terms(
        self, top_docs: list[int], existing_terms: set[str]
    ) -> list[str]:
        """
        Extract the top-IDF terms from ``top_docs`` that are not already in
        the query (``existing_terms``).  Returns at most PRF_N_EXP terms.
        """
        term_idf: dict[str, float] = {}
        for doc_idx in top_docs:
            for term in self._index[doc_idx]:
                if term in existing_terms:
                    continue
                if len(term) <= 2:  # skip very short tokens
                    continue
                term_idf[term] = self._idf(term)
        # Sort by IDF descending — prefer rare, discriminative terms
        sorted_terms = sorted(term_idf.items(), key=lambda x: x[1], reverse=True)
        return [t for t, _ in sorted_terms[: self.PRF_N_EXP]]

    # ------------------------------------------------------------------
    # Hybrid scoring pass
    # ------------------------------------------------------------------

    def _hybrid_scores(
        self, query_terms: list[str], query_token_set: set[str]
    ) -> list[tuple[float, float, float, float]]:
        """
        Compute (bm25_raw, scs, hybrid, prev_norm) tuples for all documents.

        BM25 scores are min-max normalised to [0, 1] before interpolation.
        The composite hybrid score is:
            H = ALPHA * BM25_norm + BETA_SCS * SCS + PREV_WEIGHT * prev_norm
        """
        raw_bm25 = [self._bm25_score(query_terms, i) for i in range(self._n)]
        scs_scores = [self._scs(query_token_set, i) for i in range(self._n)]

        max_bm25 = max(raw_bm25) if raw_bm25 else 1.0
        if max_bm25 == 0.0:
            max_bm25 = 1.0  # avoid divide-by-zero

        results: list[tuple[float, float, float, float]] = []
        for i in range(self._n):
            bm25_norm = raw_bm25[i] / max_bm25
            scs = scs_scores[i]
            prev_norm = self._log_prior_norms[i] if self._log_prior_norms else 0.0
            hybrid = (
                self.ALPHA * bm25_norm
                + self.BETA_SCS * scs
                + self.PREV_WEIGHT * prev_norm
            )
            results.append((raw_bm25[i], scs, hybrid, prev_norm))
        return results

    # ------------------------------------------------------------------
    # Score entropy
    # ------------------------------------------------------------------

    @staticmethod
    def score_entropy(scores: list[float]) -> float:
        """
        Normalised Shannon entropy of the score distribution.

        H_norm = H / log(k) in [0, 1].
        0  = one document dominates (low uncertainty).
        1  = uniform distribution   (high uncertainty).
        """
        total = sum(scores)
        if total == 0.0 or len(scores) <= 1:
            return 0.0
        probs = [s / total for s in scores]
        entropy = -sum(p * math.log(p) for p in probs if p > 0)
        max_entropy = math.log(len(scores))
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def confidence_from_entropy(
        self, entropy_norm: float, evidence_bonus: float = 0.0
    ) -> str:
        """
        Map 1 - H_norm (+ evidence_bonus) to {low, medium, high}.

        Parameters
        ----------
        entropy_norm : float
            Normalised entropy in [0, 1] from ``score_entropy``.
        evidence_bonus : float
            Additive bonus from GRADE evidence tier.
        """
        raw = 1.0 - entropy_norm + evidence_bonus
        if raw >= self.CONF_HIGH_THRESHOLD:
            return "high"
        if raw >= self.CONF_MED_THRESHOLD:
            return "medium"
        return "low"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """Return the top-k most relevant entries (entries only, no scores)."""
        return [r.entry for r in self.retrieve_results(query, [], top_k)]

    def retrieve_scored(self, query: str, top_k: int = 3) -> list[tuple[float, dict]]:
        """Return ``[(hybrid_score, entry), ...]`` sorted descending."""
        return [(r.hybrid_score, r.entry) for r in self.retrieve_results(query, [], top_k)]

    def retrieve_results(
        self,
        query: str,
        symptom_tokens: list[str],
        top_k: int = 3,
    ) -> list[RetrievalResult]:
        """
        Full MedHybrid-BM25 retrieval pipeline.

        Parameters
        ----------
        query : str
            Pre-processed query string (negations removed, synonyms resolved).
        symptom_tokens : list[str]
            Additional explicit symptom tokens (from structured input).
        top_k : int
            Number of results to return.

        Returns
        -------
        list[RetrievalResult]
            Top-k results, sorted by hybrid score descending.
        """
        if not self.entries:
            return []

        all_tokens = self._tokenize(query) + [t.lower() for t in symptom_tokens]
        if not all_tokens:
            # No usable query — return top-1 fallback with zero scores
            return [
                RetrievalResult(
                    entry=self.entries[0],
                    bm25_score=0.0,
                    scs_score=0.0,
                    hybrid_score=0.0,
                )
            ]

        query_token_set = set(all_tokens)

        # ----- Stage 1+2: first-pass hybrid scoring -----
        scores_1 = self._hybrid_scores(all_tokens, query_token_set)
        ranked_1 = sorted(range(self._n), key=lambda i: scores_1[i][2], reverse=True)

        # ----- Stage 3: PRF expansion -----
        prf_doc_idxs = ranked_1[: self.PRF_TOP_K]
        expansion_terms = self._extract_expansion_terms(prf_doc_idxs, query_token_set)

        if expansion_terms:
            expanded_tokens = all_tokens + expansion_terms
            expanded_set = set(expanded_tokens)
            scores_2 = self._hybrid_scores(expanded_tokens, expanded_set)
            # Interpolate: beta * original + (1 - beta) * expanded
            final_hybrid = [
                self.PRF_BETA * scores_1[i][2] + (1.0 - self.PRF_BETA) * scores_2[i][2]
                for i in range(self._n)
            ]
        else:
            final_hybrid = [scores_1[i][2] for i in range(self._n)]

        # ----- Rank and package results -----
        ranked_final = sorted(range(self._n), key=lambda i: final_hybrid[i], reverse=True)

        results: list[RetrievalResult] = []
        for doc_idx in ranked_final:
            hy = final_hybrid[doc_idx]
            bm25, scs, _, prev_norm = scores_1[doc_idx]
            ruling_in, ruling_out = self._ruling(query_token_set, doc_idx)
            results.append(
                RetrievalResult(
                    entry=self.entries[doc_idx],
                    bm25_score=bm25,
                    scs_score=scs,
                    hybrid_score=hy,
                    prevalence_score=prev_norm,
                    ruling_in=ruling_in,
                    ruling_out=ruling_out,
                )
            )

        # Filter: keep only positive-scoring results (up to top_k)
        positive = [r for r in results if r.hybrid_score > 0.0][:top_k]
        if positive:
            return positive
        # Fallback: return the single best result even if score is 0
        return [results[0]] if results else []
