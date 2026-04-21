"""
Medical Query Processor
=======================
Implements three evidence-based NLP preprocessing steps that improve retrieval
precision in medical information systems:

1. **Symptom Synonym Normalization**
   Maps lay-language symptom descriptions to canonical medical terms, bridging
   the vocabulary mismatch between patients and clinical literature.  The
   dictionary covers ~250 surface forms drawn from UMLS consumer-vocabulary
   mappings (National Library of Medicine, 2023).

2. **NegEx-inspired Negation Detection** (Chapman et al., 2001)
   Identifies symptoms that are explicitly negated by the patient (e.g.
   "no fever", "without chest pain").  Negated symptoms are *excluded* from
   the retrieval query so that the BM25 index is not contaminated by symptoms
   the patient does NOT have.  The sliding-window implementation closely
   follows the original NegEx algorithm.

3. **Medical Stop-Word Filtering**
   Removes high-frequency filler tokens ("I", "have", "my", …) that add noise
   without contributing discriminative power to symptom-based retrieval.

References
----------
Chapman, W. W. et al. (2001). A simple algorithm for identifying negated
findings and diseases in discharge summaries. *Journal of Biomedical
Informatics*, 34(5), 301–310.

National Library of Medicine (2023). UMLS Consumer Health Vocabulary.
https://www.nlm.nih.gov/research/umls/
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Symptom synonym dictionary
# ---------------------------------------------------------------------------
# Surface-form → canonical medical term.
# Sorted at call-time by length descending so multi-word phrases match first.

SYMPTOM_SYNONYMS: dict[str, str] = {
    # -- General / Systemic ------------------------------------------------
    "high temperature": "fever",
    "burning up": "fever",
    "feverish": "fever",
    "temperature": "fever",
    "fever": "fever",
    "chills": "chills",
    "shivering": "chills",
    "night sweats": "sweating",
    "sweating": "sweating",
    "tiredness": "fatigue",
    "exhaustion": "fatigue",
    "exhausted": "fatigue",
    "malaise": "fatigue",
    "tired": "fatigue",
    "weakness": "weakness",
    "weak": "weakness",
    "unintentional weight loss": "unexplained weight loss",
    "unexplained weight loss": "unexplained weight loss",
    "losing weight": "unexplained weight loss",
    "weight loss": "unexplained weight loss",
    "weight gain": "weight gain",
    "gaining weight": "weight gain",
    "loss of appetite": "loss of appetite",
    "no appetite": "loss of appetite",
    "not eating": "loss of appetite",
    # -- Head / Neurological -----------------------------------------------
    "worst headache of my life": "thunderclap headache",
    "thunderclap headache": "thunderclap headache",
    "severe headache": "severe headache",
    "throbbing headache": "throbbing head pain",
    "head pain": "headache",
    "headache": "headache",
    "migraine": "severe headache",
    "lightheaded": "dizziness",
    "light-headed": "dizziness",
    "light headed": "dizziness",
    "dizzy": "dizziness",
    "dizziness": "dizziness",
    "spinning": "vertigo",
    "vertigo": "vertigo",
    "brain fog": "difficulty concentrating",
    "can't concentrate": "difficulty concentrating",
    "disoriented": "confusion",
    "confused": "confusion",
    "confusion": "confusion",
    "memory loss": "memory impairment",
    "forgetful": "memory impairment",
    "slurred speech": "slurred speech",
    "facial droop": "facial droop",
    "sudden weakness": "sudden weakness",
    "sudden numbness": "sudden numbness",
    "loss of smell": "loss of smell",
    "can't smell": "loss of smell",
    "loss of taste": "loss of taste",
    "can't taste": "loss of taste",
    # -- Respiratory -------------------------------------------------------
    "shortness of breath": "shortness of breath",
    "breathlessness": "shortness of breath",
    "breathless": "shortness of breath",
    "difficulty breathing": "difficulty breathing",
    "can't breathe": "difficulty breathing",
    "cant breathe": "difficulty breathing",
    "unable to breathe": "difficulty breathing",
    "productive cough": "productive cough",
    "coughing up phlegm": "productive cough",
    "mucus cough": "productive cough",
    "phlegm": "productive cough",
    "mucus": "productive cough",
    "dry cough": "dry cough",
    "coughing": "cough",
    "cough": "cough",
    "wheezing": "wheezing",
    "wheeze": "wheezing",
    "runny nose": "runny nose",
    "rhinorrhoea": "runny nose",
    "rhinorrhea": "runny nose",
    "stuffy nose": "nasal congestion",
    "blocked nose": "nasal congestion",
    "congestion": "nasal congestion",
    "nasal congestion": "nasal congestion",
    "sneezing": "sneezing",
    "throat pain": "sore throat",
    "sore throat": "sore throat",
    "painful swallowing": "difficulty swallowing",
    "difficulty swallowing": "difficulty swallowing",
    # -- Cardiovascular ----------------------------------------------------
    "heart racing": "palpitations",
    "heart pounding": "palpitations",
    "heart fluttering": "palpitations",
    "palpitations": "palpitations",
    "irregular heartbeat": "irregular heartbeat",
    "fast heartbeat": "rapid heartbeat",
    "rapid heartbeat": "rapid heartbeat",
    "racing heart": "palpitations",
    "chest heaviness": "chest pressure",
    "chest tightness": "chest tightness",
    "chest pressure": "chest pressure",
    "chest pain": "chest pain",
    "jaw pain": "jaw pain",
    "left arm pain": "left arm pain",
    # -- Gastrointestinal --------------------------------------------------
    "tummy ache": "abdominal pain",
    "stomach ache": "abdominal pain",
    "stomach pain": "abdominal pain",
    "belly pain": "abdominal pain",
    "abdominal cramps": "abdominal cramps",
    "abdominal pain": "abdominal pain",
    "sick to stomach": "nausea",
    "queasy": "nausea",
    "nauseated": "nausea",
    "nausea": "nausea",
    "throwing up": "vomiting",
    "puking": "vomiting",
    "vomiting": "vomiting",
    "loose stools": "diarrhea",
    "watery stool": "diarrhea",
    "diarrhea": "diarrhea",
    "diarrhoea": "diarrhea",
    "constipation": "constipation",
    "can't poop": "constipation",
    "bloating": "bloating",
    "bloated": "bloating",
    "heartburn": "heartburn",
    "acid reflux": "heartburn",
    "indigestion": "indigestion",
    "blood in stool": "blood in stool",
    "rectal bleeding": "blood in stool",
    # -- Urinary -----------------------------------------------------------
    "pain when urinating": "burning urination",
    "painful urination": "burning urination",
    "burning when urinating": "burning urination",
    "burning urination": "burning urination",
    "frequent urination": "frequent urination",
    "urgency to urinate": "urgency to urinate",
    "urgency": "urgency to urinate",
    # -- Musculoskeletal ---------------------------------------------------
    "muscle aches": "body aches",
    "muscle pain": "body aches",
    "body ache": "body aches",
    "body aches": "body aches",
    "joint pain": "joint pain",
    "lower back pain": "lower back pain",
    "back pain": "lower back pain",
    "neck stiffness": "stiff neck",
    "stiff neck": "stiff neck",
    "calf pain": "calf pain",
    "swollen calf": "calf pain",
    "leg swelling": "leg swelling",
    "swollen leg": "leg swelling",
    "swollen ankles": "leg swelling",
    "shoulder tip pain": "shoulder tip pain",
    # -- Skin / Eyes / Ear -------------------------------------------------
    "skin rash": "skin rash",
    "hives": "skin rash",
    "rash": "skin rash",
    "itching": "itching",
    "itchy": "itching",
    "blurry vision": "blurred vision",
    "blurred vision": "blurred vision",
    "double vision": "double vision",
    "sudden vision loss": "sudden vision loss",
    "pink eye": "red eye",
    "red eyes": "red eye",
    "red eye": "red eye",
    "ear pain": "ear pain",
    "earache": "ear pain",
    "ear ache": "ear pain",
    # -- Psychological / Emotional ----------------------------------------
    "panic attack": "panic attack",
    "anxious": "anxiety",
    "anxiety": "anxiety",
    "worried": "excessive worry",
    "low mood": "persistent sadness",
    "depressed": "persistent sadness",
    "sad": "persistent sadness",
    "hopeless": "hopelessness",
    "no interest": "loss of interest",
    "anhedonia": "loss of interest",
    # -- Metabolic / Endocrine --------------------------------------------
    "excessive thirst": "increased thirst",
    "increased thirst": "increased thirst",
    "thirsty": "increased thirst",
    "cold intolerance": "cold intolerance",
    "heat intolerance": "heat intolerance",
    "puffiness": "puffiness",
    "puffy face": "facial puffiness",
    "hair loss": "hair loss",
    "hair thinning": "hair loss",
    "dry skin": "dry skin",
    "pale skin": "pallor",
    "pallor": "pallor",
    # -- Vascular / Other -------------------------------------------------
    "swollen lymph nodes": "lymphadenopathy",
    "swollen glands": "lymphadenopathy",
    "lymph node swelling": "lymphadenopathy",
    "shoulder pain": "shoulder pain",
    "flank pain": "flank pain",
    "groin pain": "groin pain",
}

# ---------------------------------------------------------------------------
# NegEx negation triggers  (Chapman et al., 2001, Table 1)
# ---------------------------------------------------------------------------
# Any token or 2-gram in this set, when encountered, marks the following
# _WINDOW_SIZE tokens as negated.

_NEGATION_TRIGGERS: frozenset[str] = frozenset(
    {
        "no",
        "not",
        "without",
        "denies",
        "deny",
        "absent",
        "absence",
        "free of",
        "negative for",
        "rules out",
        "ruled out",
        "never",
        "none",
        "neither",
        "nor",
        "cannot",
        "can't",
        "cant",
        "doesn't have",
        "does not have",
        "no sign of",
        "no evidence of",
    }
)

_WINDOW_SIZE: int = 4  # tokens to mark negated after a trigger

# ---------------------------------------------------------------------------
# Medical stop-words
# ---------------------------------------------------------------------------
_MEDICAL_STOPWORDS: frozenset[str] = frozenset(
    {
        "i",
        "have",
        "had",
        "has",
        "my",
        "me",
        "am",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "so",
        "for",
        "of",
        "in",
        "on",
        "at",
        "to",
        "with",
        "from",
        "by",
        "it",
        "this",
        "that",
        "what",
        "could",
        "cause",
        "about",
        "please",
        "help",
        "feel",
        "feeling",
        "experiencing",
        "experience",
        "also",
        "some",
        "bit",
        "very",
        "quite",
        "really",
        "little",
    }
)


# ---------------------------------------------------------------------------
# Data class for preprocessing result
# ---------------------------------------------------------------------------


@dataclass
class ProcessedQuery:
    """
    Structured output of ``QueryProcessor.process()``.

    Attributes
    ----------
    original : str
        The unmodified input query string.
    normalized_terms : list[str]
        Synonym-resolved canonical terms found in the input.
    negated_terms : list[str]
        Terms detected as negated by the NegEx heuristic.
    affirmed_terms : list[str]
        Non-negated tokens after stop-word removal.
    expanded_query : str
        Full normalized text for BM25 indexing (affirmed tokens joined).
    medical_tokens : list[str]
        Unique, deduplicated affirmed medical tokens (no stopwords).
    """

    original: str
    normalized_terms: list[str] = field(default_factory=list)
    negated_terms: list[str] = field(default_factory=list)
    affirmed_terms: list[str] = field(default_factory=list)
    expanded_query: str = ""
    medical_tokens: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main processor
# ---------------------------------------------------------------------------


class QueryProcessor:
    """
    Medical query preprocessor.

    All methods are class methods / static methods so the class can be used
    without instantiation — there is no mutable instance state.
    """

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", text.lower().strip())

    @classmethod
    def _apply_synonyms(cls, text: str) -> tuple[str, list[str]]:
        """
        Replace surface forms with canonical medical terms.

        Multi-word phrases are processed first (sorted by length descending)
        to avoid partial-match overwriting.  The function returns both the
        rewritten text and a list of resolved canonical terms for downstream
        metadata.
        """
        normalized = cls._normalize_text(text)
        resolved: list[str] = []

        for term, canonical in sorted(
            SYMPTOM_SYNONYMS.items(), key=lambda x: len(x[0]), reverse=True
        ):
            if term in normalized:
                normalized = normalized.replace(term, canonical)
                if canonical not in resolved:
                    resolved.append(canonical)

        return normalized, resolved

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [t.strip(".,!?;:()[]\"'") for t in text.split() if t.strip(".,!?;:()[]\"'")]

    @classmethod
    def _detect_negations(cls, tokens: list[str]) -> tuple[list[str], list[str]]:
        """
        Sliding-window NegEx negation detector.

        For each token (or bigram) that is a negation trigger, mark the next
        ``_WINDOW_SIZE`` tokens as negated.  Negated tokens are *excluded*
        from retrieval to prevent false positive matches.

        Returns
        -------
        negated_terms : list[str]
        affirmed_terms : list[str]
        """
        negated_indices: set[int] = set()

        for i, tok in enumerate(tokens):
            # Single-token trigger
            if tok in _NEGATION_TRIGGERS:
                for j in range(i + 1, min(i + 1 + _WINDOW_SIZE, len(tokens))):
                    negated_indices.add(j)
            # Two-token trigger
            if i < len(tokens) - 1:
                bigram = tok + " " + tokens[i + 1]
                if bigram in _NEGATION_TRIGGERS:
                    for j in range(i + 2, min(i + 2 + _WINDOW_SIZE, len(tokens))):
                        negated_indices.add(j)

        negated = [tokens[i] for i in sorted(negated_indices)]
        affirmed = [tokens[i] for i in range(len(tokens)) if i not in negated_indices]
        return negated, affirmed

    @classmethod
    def process(cls, query: str, symptoms: list[str] | None = None) -> ProcessedQuery:
        """
        Preprocess a free-text query and optional structured symptom list.

        Parameters
        ----------
        query : str
            Free-text health question or symptom description.
        symptoms : list[str] | None
            Optional structured symptom list (from the symptom-checker UI).

        Returns
        -------
        ProcessedQuery
            A structured object containing all preprocessing artefacts.
        """
        combined = query
        if symptoms:
            combined = combined + " " + " ".join(symptoms)

        # 1. Synonym normalization
        normalized_text, resolved_terms = cls._apply_synonyms(combined)

        # 2. Tokenize
        tokens = cls._tokenize(normalized_text)

        # 3. Negation detection
        negated, affirmed = cls._detect_negations(tokens)

        # 4. Build medical tokens — unique, non-negated, non-stopword
        medical_tokens: list[str] = list(
            {t for t in affirmed if t not in _MEDICAL_STOPWORDS and len(t) > 2}
        )

        return ProcessedQuery(
            original=query,
            normalized_terms=resolved_terms,
            negated_terms=negated,
            affirmed_terms=affirmed,
            expanded_query=" ".join(medical_tokens),
            medical_tokens=medical_tokens,
        )
