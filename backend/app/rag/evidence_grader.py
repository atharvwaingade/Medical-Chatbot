"""
GRADE-Inspired Evidence Tier Grader
=====================================
Maps medical source citation strings to evidence quality tiers following the
GRADE (Grading of Recommendations, Assessment, Development and Evaluation)
evidence framework.

Reference tiers
---------------
Tier 1 — Systematic review / Meta-analysis (e.g. Cochrane Reviews)
Tier 2 — Randomised Controlled Trial (RCT)
Tier 3 — Clinical Practice Guideline (WHO, CDC, NIH, AHA, ACC, …)
Tier 4 — Observational / Peer-reviewed journal article (NEJM, JAMA, …)
Tier 5 — Expert consensus / Narrative review / Textbook

The evidence tier feeds into two downstream systems:

1. **Context ordering** — higher-tier documents are placed first in the LLM
   prompt context, ensuring the model attends to the strongest evidence first
   (as recommended by Liu et al., 2023, "Lost in the Middle").

2. **Confidence calibration** — each tier contributes a small additive bonus
   to the entropy-derived retrieval confidence score, reflecting that strong
   evidence warrants slightly higher expressed confidence.

References
----------
Guyatt, G. H. et al. (2008). GRADE: an emerging consensus on rating quality
of evidence and strength of recommendations. *BMJ*, 336(7650), 924–926.

GRADE Working Group (2011). GRADE guidelines: 1. Introduction — GRADE evidence
profiles and summary of findings tables. *Journal of Clinical Epidemiology*,
64(4), 383–394.

Liu, N. F. et al. (2023). Lost in the Middle: How Language Models Use Long
Contexts. *arXiv*:2307.03172.
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Tier metadata
# ---------------------------------------------------------------------------

TIER_DESCRIPTIONS: dict[int, str] = {
    1: "Systematic review / Meta-analysis",
    2: "Randomised Controlled Trial",
    3: "Clinical Practice Guideline",
    4: "Peer-reviewed research article",
    5: "Expert consensus / Narrative review",
}

# Confidence bonus added to entropy-based score (additive, not multiplicative)
TIER_CONFIDENCE_BONUS: dict[int, float] = {
    1: 0.15,
    2: 0.10,
    3: 0.07,
    4: 0.04,
    5: 0.00,
}

# ---------------------------------------------------------------------------
# Keyword routing table
# ---------------------------------------------------------------------------
# Each entry: (tier, tuple_of_lowercase_keywords).
# For keywords <= 4 characters we require a word-boundary match to prevent
# false positives (e.g. "ada" matching inside "headaches").
# For longer phrases, substring matching is safe and intentional.

_TIER_KEYWORDS: list[tuple[int, tuple[str, ...]]] = [
    # Tier 1 — Systematic reviews / Cochrane
    (1, ("cochrane", "meta-analysis", "systematic review", "meta analysis")),
    # Tier 2 — RCTs
    (2, ("randomised controlled", "randomized controlled", "rct", "clinical trial")),
    # Tier 3 — Major clinical guidelines and public health agencies
    (
        3,
        (
            "who",
            "cdc",
            "nih",
            "nhlbi",
            "niddk",
            "nimh",
            "aha",
            "acc",
            "ada",
            "ata",
            "aaaai",
            "acg",
            "aaos",
            "aap",
            "aao",
            "acs",
            "acp",
            "ats",
            "idsa",
            "accp",
            "aasld",
            "esc",
            "aafp",
            "american heart association",
            "american diabetes association",
            "american thyroid association",
            "american academy",
            "american college",
            "national heart",
            "national cancer",
            "national institute",
            "guidelines",
            "guideline",
        ),
    ),
    # Tier 4 — Peer-reviewed journals
    (
        4,
        (
            "nejm",
            "new england journal",
            "jama",
            "lancet",
            "bmj",
            "annals of internal medicine",
            "pubmed",
            "plos",
            "nature medicine",
            "journal of",
            "annals of",
            "archives of",
            "european heart journal",
            "circulation",
        ),
    ),
]

# Short keyword length threshold — use word-boundary matching for these
_ABBREV_MAX_LEN: int = 5


def _keyword_matches(keyword: str, text: str) -> bool:
    """
    Return True if *keyword* is found in *text*.

    Short keywords (≤ _ABBREV_MAX_LEN characters, e.g. abbreviations) are
    matched at word boundaries to prevent false positives such as "ada"
    matching inside "headaches".  Longer phrases use plain substring search.
    """
    if len(keyword) <= _ABBREV_MAX_LEN:
        return bool(re.search(r"\b" + re.escape(keyword) + r"\b", text))
    return keyword in text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def grade_source(source: str) -> int:
    """
    Return the evidence tier (1–5) for a source citation string.

    Matching is performed in ascending tier order so the strongest
    applicable tier is returned.

    Parameters
    ----------
    source : str
        Free-text citation, e.g. ``"WHO - Respiratory Tract Infection"``.

    Returns
    -------
    int
        Evidence tier 1 (strongest) through 5 (weakest).
    """
    lower = source.lower()
    for tier, keywords in _TIER_KEYWORDS:
        if any(_keyword_matches(kw, lower) for kw in keywords):
            return tier
    return 5  # default: expert consensus


def tier_description(tier: int) -> str:
    """Return a human-readable label for an evidence tier integer."""
    return TIER_DESCRIPTIONS.get(tier, "Unknown evidence tier")


def confidence_bonus(tier: int) -> float:
    """Return the additive confidence bonus for a given evidence tier."""
    return TIER_CONFIDENCE_BONUS.get(tier, 0.0)
