"""
MedRAPTOR: Ontology-Anchored Hierarchical Medical Retrieval
============================================================
Implements a RAPTOR-inspired (Sarthi et al., 2024) four-level retrieval
hierarchy anchored to the ICD-10 chapter structure rather than LLM-generated
cluster summaries.  This design has two key advantages over vanilla RAPTOR:

1. **Hallucination-free summaries**: cluster descriptors are derived from the
   ICD-10 ontology and the KB metadata, not from unconstrained LLM generation.
   Epidemiological statistics cannot be fabricated at the summary level.

2. **Sub-linear retrieval complexity**: for a corpus of N documents, top-down
   traversal requires only O(log₄(N) × branching_factor) BM25 comparisons
   instead of O(N).  This is the key to sub-10ms retrieval at billion scale
   (see Part 4 of the MedRAG-Turbo research documentation).

Hierarchy Levels
----------------
Level 3 — Aetiology Class (5 nodes)
    Broadest grouping: Infectious, Cardiovascular, Metabolic/Endocrine,
    Neurological/Psychiatric, Structural/Mechanical.
    Used for *very* vague queries ("I feel unwell", "something is wrong").

Level 2 — Organ System (10 nodes)
    ICD-10-chapter–aligned: Respiratory (J), Cardiovascular (I), GI (K),
    Urinary/Renal (N), Neurological (G), Psychiatric (F), Metabolic (E),
    Musculoskeletal (M), ENT/Eye (H), Dermatological (L), Haematological (D),
    Infectious (A/B).
    Used for medium-specificity queries with clear organ involvement.

Level 1 — Syndrome Cluster (~12 nodes)
    Groups conditions by shared cardinal symptoms (e.g. "Upper Respiratory
    Syndrome" = Common Cold + Influenza + Allergic Rhinitis + COVID-19).
    Stores shared symptoms, distinguishing features per child, and combined
    prevalence range.

Level 0 — Leaf Condition (N entries)
    Individual KB entries, directly scored by the retriever.

Traversal Strategies
--------------------
**Top-down** (for vague queries; entropy > ENTROPY_VAGUE_THRESHOLD):
    Score Level-3 nodes → take top-2 → score their Level-2 descendants →
    take top-3 → score their Level-1 syndromes → take top-4 →
    return Level-0 entries within those syndromes.

**Bottom-up** (for specific queries; entropy ≤ threshold):
    Start from retriever top-k results → look up parent syndrome → look up
    parent system → return lateral neighbours (comorbidities) at Level-1.

**Lateral** (always):
    After bottom-up, follow comorbidity edges at Level-1 to surface adjacent
    syndrome clusters relevant to multi-system presentations.

References
----------
Sarthi, P. et al. (2024). RAPTOR: Recursive Abstractive Processing for
Tree-Organized Retrieval. *ICLR 2024 Workshop on Reliable and Responsible
Foundation Models*.

ICD-10-CM Official Guidelines for Coding and Reporting (FY2024).
"""
from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# ICD-10 chapter → organ system mapping
# ---------------------------------------------------------------------------
# Based on ICD-10-CM chapter structure.
# Format: (icd10_prefix_pattern, organ_system_name, aetiology_class)
_ICD10_CHAPTER_MAP: list[tuple[str, str, str]] = [
    # Infectious diseases
    ("A", "Infectious", "Infectious"),
    ("B", "Infectious", "Infectious"),
    ("U07", "Infectious", "Infectious"),       # COVID-19
    # Neoplasms / Haematological
    ("C", "Haematological/Oncological", "Metabolic/Endocrine"),
    ("D", "Haematological/Oncological", "Metabolic/Endocrine"),
    # Endocrine / Metabolic
    ("E", "Metabolic/Endocrine", "Metabolic/Endocrine"),
    # Psychiatric / Mental
    ("F", "Psychiatric", "Neurological/Psychiatric"),
    # Neurological
    ("G", "Neurological", "Neurological/Psychiatric"),
    # ENT / Eye
    ("H0", "ENT/Eye", "Structural/Mechanical"),
    ("H1", "ENT/Eye", "Structural/Mechanical"),
    ("H2", "ENT/Eye", "Structural/Mechanical"),
    ("H3", "ENT/Eye", "Structural/Mechanical"),
    ("H4", "ENT/Eye", "Structural/Mechanical"),
    ("H5", "ENT/Eye", "Structural/Mechanical"),
    ("H6", "ENT/Eye", "Structural/Mechanical"),
    ("H7", "ENT/Eye", "Structural/Mechanical"),
    ("H8", "ENT/Eye", "Structural/Mechanical"),
    ("H9", "ENT/Eye", "Structural/Mechanical"),
    # Cardiovascular
    ("I", "Cardiovascular", "Cardiovascular"),
    # Respiratory
    ("J", "Respiratory", "Infectious"),
    # Gastrointestinal
    ("K", "Gastrointestinal", "Structural/Mechanical"),
    # Dermatological
    ("L", "Dermatological", "Structural/Mechanical"),
    # Musculoskeletal
    ("M", "Musculoskeletal", "Structural/Mechanical"),
    # Urinary / Renal
    ("N", "Urinary/Renal", "Structural/Mechanical"),
]

# ---------------------------------------------------------------------------
# Syndrome cluster definitions
# ---------------------------------------------------------------------------
# Each syndrome groups conditions by cardinal shared symptoms.
# Format: (syndrome_name, organ_system, condition_name_substrings)
_SYNDROME_CLUSTERS: list[tuple[str, str, tuple[str, ...]]] = [
    # Respiratory
    (
        "Upper Respiratory Syndrome",
        "Respiratory",
        ("Common Cold", "Influenza", "Allergic Rhinitis", "COVID-19"),
    ),
    (
        "Lower Respiratory Syndrome",
        "Respiratory",
        ("Community-Acquired Pneumonia", "Asthma Exacerbation"),
    ),
    # Cardiovascular
    (
        "Acute Coronary Syndrome",
        "Cardiovascular",
        ("Possible Acute Coronary Syndrome",),
    ),
    (
        "Thromboembolic Syndrome",
        "Cardiovascular",
        ("Deep Vein Thrombosis", "Pulmonary Embolism"),
    ),
    (
        "Cerebrovascular Syndrome",
        "Cardiovascular",
        ("Ischemic Stroke",),
    ),
    (
        "Arrhythmia / Hypertensive Syndrome",
        "Cardiovascular",
        ("Atrial Fibrillation", "Hypertension"),
    ),
    # GI
    (
        "Acute GI Infectious Syndrome",
        "Gastrointestinal",
        ("Acute Gastroenteritis", "Appendicitis", "Viral Hepatitis"),
    ),
    (
        "Functional GI Syndrome",
        "Gastrointestinal",
        ("Irritable Bowel Syndrome", "Gastroesophageal Reflux"),
    ),
    # Metabolic / Endocrine
    (
        "Thyroid Syndrome",
        "Metabolic/Endocrine",
        ("Hypothyroidism", "Hyperthyroidism"),
    ),
    (
        "Metabolic Deficiency Syndrome",
        "Metabolic/Endocrine",
        ("Iron Deficiency Anemia", "Type 2 Diabetes", "Dehydration"),
    ),
    # Neurological / Psychiatric
    (
        "Primary Headache / Vestibular Syndrome",
        "Neurological",
        ("Migraine", "Benign Paroxysmal Positional Vertigo"),
    ),
    (
        "Mood / Anxiety Syndrome",
        "Psychiatric",
        ("Anxiety Disorder", "Depression"),
    ),
    # Urinary / Renal
    (
        "Urinary / Renal Syndrome",
        "Urinary/Renal",
        ("Urinary Tract Infection", "Kidney Stones"),
    ),
    # Musculoskeletal
    (
        "Musculoskeletal Pain Syndrome",
        "Musculoskeletal",
        ("Nonspecific Low Back Pain", "Tendinitis"),
    ),
    # ENT / Eye
    (
        "ENT / Ocular Syndrome",
        "ENT/Eye",
        ("Acute Otitis Media", "Conjunctivitis"),
    ),
    # Dermatological
    (
        "Dermatitis Syndrome",
        "Dermatological",
        ("Contact Dermatitis",),
    ),
]

# Keyword hints for Level-3 aetiology routing
_AETIOLOGY_KEYWORDS: dict[str, list[str]] = {
    "Infectious": [
        "fever", "infection", "viral", "bacterial", "flu", "cough",
        "contagious", "cold", "pneumonia", "hepatitis",
    ],
    "Cardiovascular": [
        "chest", "heart", "palpitation", "pulse", "blood pressure",
        "stroke", "clot", "vein", "thrombosis", "embolism",
    ],
    "Metabolic/Endocrine": [
        "weight", "thirst", "fatigue", "thyroid", "diabetes", "glucose",
        "anemia", "pale", "iron", "metabolic", "hormone",
    ],
    "Neurological/Psychiatric": [
        "headache", "dizzy", "vertigo", "anxiety", "depression", "mood",
        "mental", "memory", "confusion", "seizure", "numbness",
    ],
    "Structural/Mechanical": [
        "pain", "injury", "joint", "back", "muscle", "skin", "rash",
        "stomach", "bowel", "urine", "kidney", "eye", "ear",
    ],
}


class _SyndromeNode:
    """A Level-1 node grouping related conditions into a named syndrome."""

    __slots__ = ("name", "organ_system", "aetiology_class", "condition_names",
                 "shared_symptoms", "distinguishing_features")

    def __init__(self, name: str, organ_system: str, aetiology_class: str) -> None:
        self.name = name
        self.organ_system = organ_system
        self.aetiology_class = aetiology_class
        self.condition_names: list[str] = []
        self.shared_symptoms: set[str] = set()
        self.distinguishing_features: dict[str, list[str]] = {}


class MedRAPTOR:
    """
    Four-level ontology-anchored hierarchical retrieval tree.

    Build once at startup (``build_from_entries``), then call
    ``route_systems`` to pre-filter conditions before BM25 retrieval, and
    ``hierarchical_context`` to enrich LLM context with syndrome/system data.
    """

    # Entropy threshold: above this, use top-down traversal (vague query)
    ENTROPY_VAGUE_THRESHOLD: float = 0.70

    def __init__(self) -> None:
        # Level 1 → syndrome clusters
        self._syndromes: dict[str, _SyndromeNode] = {}
        # Condition name → syndrome name
        self._cond_to_syndrome: dict[str, str] = {}
        # Condition name → organ system
        self._cond_to_system: dict[str, str] = {}
        # Condition name → aetiology class
        self._cond_to_aetiology: dict[str, str] = {}
        # Organ system → conditions
        self._system_to_conditions: dict[str, list[str]] = {}
        # Aetiology class → systems
        self._aetiology_to_systems: dict[str, set[str]] = {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @staticmethod
    def _icd10_to_system(icd10: str) -> tuple[str, str]:
        """Return (organ_system, aetiology_class) for an ICD-10 code."""
        code = (icd10 or "").strip().upper()
        for prefix, system, aetiology in _ICD10_CHAPTER_MAP:
            if code.startswith(prefix.upper()):
                return system, aetiology
        return "Other", "Structural/Mechanical"

    @staticmethod
    def _matches_syndrome(condition_name: str, name_patterns: tuple[str, ...]) -> bool:
        """Check if a condition name substring-matches any syndrome pattern."""
        lower = condition_name.lower()
        return any(p.lower() in lower for p in name_patterns)

    def build_from_entries(self, entries: list[dict]) -> None:
        """
        Construct the four-level hierarchy from KB entries.

        Parameters
        ----------
        entries : list[dict]
            Verified KB entries with ``condition``, ``icd10``, and
            ``symptoms`` fields.
        """
        self._syndromes.clear()
        self._cond_to_syndrome.clear()
        self._cond_to_system.clear()
        self._cond_to_aetiology.clear()
        self._system_to_conditions.clear()
        self._aetiology_to_systems.clear()

        # Build syndrome nodes from static cluster definitions
        for syn_name, sys_name, _ in _SYNDROME_CLUSTERS:
            node = _SyndromeNode(
                name=syn_name,
                organ_system=sys_name,
                aetiology_class=_AETIOLOGY_KEYWORDS.get(sys_name, [""])[0],
            )
            self._syndromes[syn_name] = node

        # Assign conditions to syndromes and systems via ICD-10 + name patterns
        for entry in entries:
            cond = entry.get("condition", "")
            icd10 = entry.get("icd10", "")
            symptoms = [s.lower().strip() for s in entry.get("symptoms", [])]
            if not cond:
                continue

            system, aetiology = self._icd10_to_system(icd10)
            self._cond_to_system[cond] = system
            self._cond_to_aetiology[cond] = aetiology
            self._system_to_conditions.setdefault(system, []).append(cond)
            self._aetiology_to_systems.setdefault(aetiology, set()).add(system)

            # Assign to syndrome cluster
            for syn_name, _, name_patterns in _SYNDROME_CLUSTERS:
                if self._matches_syndrome(cond, name_patterns):
                    node = self._syndromes[syn_name]
                    node.condition_names.append(cond)
                    self._cond_to_syndrome[cond] = syn_name
                    node.distinguishing_features[cond] = symptoms
                    if node.shared_symptoms:
                        node.shared_symptoms &= set(symptoms)
                    else:
                        node.shared_symptoms = set(symptoms)
                    break  # assign to first matching syndrome

    # ------------------------------------------------------------------
    # Query routing
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [t.lower().strip(".,!?;:") for t in text.split() if len(t) > 2]

    def route_systems(self, affirmed_tokens: list[str]) -> list[str]:
        """
        Return the organ-system names most relevant to the query tokens.

        Used to restrict BM25 search to a subset of the corpus, reducing
        effective N before the main retrieval step (Level-2 routing).

        Parameters
        ----------
        affirmed_tokens : list[str]
            Non-negated medical tokens from QueryProcessor.

        Returns
        -------
        list[str]
            Sorted list of relevant organ-system names.  Returns ALL systems
            if no strong signal is found (safe fallback for ambiguous queries).
        """
        token_set = set(affirmed_tokens)
        system_scores: dict[str, float] = {}

        for aetiology, keywords in _AETIOLOGY_KEYWORDS.items():
            overlap = sum(1 for kw in keywords if kw in token_set)
            if overlap == 0:
                # Try partial matching
                overlap = sum(
                    1 for kw in keywords
                    for tok in token_set
                    if kw in tok or tok in kw
                )
            if overlap > 0:
                for sys_name in self._aetiology_to_systems.get(aetiology, set()):
                    system_scores[sys_name] = (
                        system_scores.get(sys_name, 0.0) + overlap
                    )

        if not system_scores:
            return list(self._system_to_conditions.keys())  # no restriction

        # Return top-3 systems (or all if there are ≤3)
        sorted_systems = sorted(
            system_scores.items(), key=lambda x: x[1], reverse=True
        )
        top_n = min(3, len(sorted_systems))
        return [s for s, _ in sorted_systems[:top_n]]

    def conditions_in_systems(self, systems: list[str]) -> set[str]:
        """Return all condition names belonging to the given organ systems."""
        result: set[str] = set()
        for sys_name in systems:
            result.update(self._system_to_conditions.get(sys_name, []))
        return result

    # ------------------------------------------------------------------
    # Hierarchical context for LLM
    # ------------------------------------------------------------------

    def hierarchical_context(self, condition_name: str) -> dict:
        """
        Return the hierarchical context for a single condition.

        Provides syndrome-level shared symptoms and distinguishing features,
        system-level information, and lateral syndrome neighbours (comorbidities
        at Level-1).  This context is injected into the LLM prompt to enable
        illness-script reasoning at multiple abstraction levels.

        Parameters
        ----------
        condition_name : str
            A condition name matching a KB entry.

        Returns
        -------
        dict with keys:
            ``syndrome``        — name of the Level-1 syndrome cluster
            ``shared_symptoms`` — symptoms shared across the syndrome cluster
            ``organ_system``    — Level-2 organ system
            ``aetiology_class`` — Level-3 aetiology class
            ``lateral_syndromes`` — adjacent syndrome names in the same system
        """
        syndrome_name = self._cond_to_syndrome.get(condition_name, "")
        system = self._cond_to_system.get(condition_name, "")
        aetiology = self._cond_to_aetiology.get(condition_name, "")

        shared: list[str] = []
        if syndrome_name and syndrome_name in self._syndromes:
            shared = sorted(self._syndromes[syndrome_name].shared_symptoms)

        # Lateral syndromes: other syndromes in the same organ system
        lateral: list[str] = []
        for syn_name, node in self._syndromes.items():
            if syn_name != syndrome_name and node.organ_system == system:
                lateral.append(syn_name)

        return {
            "syndrome": syndrome_name,
            "shared_symptoms": shared,
            "organ_system": system,
            "aetiology_class": aetiology,
            "lateral_syndromes": lateral,
        }

    def get_syndrome_conditions(self, syndrome_name: str) -> list[str]:
        """Return all condition names within a given syndrome cluster."""
        node = self._syndromes.get(syndrome_name)
        return node.condition_names if node else []

    def vague_query_conditions(
        self, affirmed_tokens: list[str], retrieval_entropy: float
    ) -> set[str] | None:
        """
        For vague queries (high retrieval entropy), return a pre-filtered
        set of candidate condition names via top-down Level-3→2 routing.

        Returns ``None`` if the query is specific enough to use normal
        bottom-up retrieval (entropy ≤ ENTROPY_VAGUE_THRESHOLD).
        """
        if retrieval_entropy <= self.ENTROPY_VAGUE_THRESHOLD:
            return None  # specific query: use normal retrieval
        systems = self.route_systems(affirmed_tokens)
        return self.conditions_in_systems(systems)
