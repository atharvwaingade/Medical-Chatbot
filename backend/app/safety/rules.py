from __future__ import annotations

# ---------------------------------------------------------------------------
# Emergency keyword phrases — any substring match in the lowercased query
# triggers an immediate emergency response.  Err on the side of over-detection
# (safety-first): a false positive costs a user a moment of caution; a false
# negative in a genuine emergency is catastrophic.
# ---------------------------------------------------------------------------
EMERGENCY_KEYWORDS: frozenset[str] = frozenset(
    {
        # Cardiac
        "chest pain",
        "chest pressure",
        "crushing chest",
        "heart attack",
        # Respiratory
        "cannot breathe",
        "can't breathe",
        "can not breathe",
        "difficulty breathing",
        "breathing difficulty",
        "shortness of breath",
        "stopped breathing",
        # Neurological
        "unconscious",
        "unconsciousness",
        "unresponsive",
        "loss of consciousness",
        "blacking out",
        "passed out",
        "facial droop",
        "slurred speech",
        "sudden weakness",
        "sudden numbness",
        "sudden vision loss",
        "sudden blindness",
        "stroke",
        # Seizure
        "seizure",
        "convulsion",
        "convulsing",
        # Headache emergency
        "worst headache of my life",
        "thunderclap headache",
        "sudden severe headache",
        # Self-harm / mental health emergency
        "suicidal",
        "want to kill myself",
        "want to die",
        "ending my life",
        # Overdose / poisoning
        "overdose",
        "took too many pills",
        "poisoning",
        # Trauma / bleeding
        "severe bleeding",
        "uncontrolled bleeding",
        "coughing blood",
        "vomiting blood",
        # Anaphylaxis
        "anaphylaxis",
        "severe allergic reaction",
        "throat swelling",
        "tongue swelling",
        # Other life-threatening
        "choking",
        "drowning",
        "electric shock",
    }
)

# ---------------------------------------------------------------------------
# Red-flag symptom *combinations* — each set must be wholly present in the
# reported symptom list to trigger an urgent response.  Multi-element sets
# improve specificity vs. single-symptom triggers.
# ---------------------------------------------------------------------------
RED_FLAGS: list[frozenset[str]] = [
    # Bacterial meningitis triad
    frozenset({"fever", "stiff neck", "severe headache"}),
    # Acute myocardial infarction cluster
    frozenset({"chest pain", "left arm pain", "sweating"}),
    # Ischemic stroke (FAST)
    frozenset({"confusion", "slurred speech", "facial droop"}),
    # Pulmonary embolism / DVT
    frozenset({"shortness of breath", "calf pain", "leg swelling"}),
    # Sepsis triad
    frozenset({"high fever", "rapid heartbeat", "confusion"}),
    # Hypertensive emergency
    frozenset({"severe headache", "blurred vision", "chest pain"}),
    # GI emergency (perforation, ischaemia)
    frozenset({"severe abdominal pain", "blood in stool", "fever"}),
    # Ectopic pregnancy / surgical abdomen
    frozenset({"severe abdominal pain", "shoulder tip pain", "dizziness"}),
    # Thyroid storm
    frozenset({"high fever", "rapid heartbeat", "confusion", "sweating"}),
]


def detect_emergency(text: str) -> bool:
    """Return True if any emergency keyword phrase is found in *text*."""
    lowered = text.lower()
    return any(keyword in lowered for keyword in EMERGENCY_KEYWORDS)


def detect_red_flags(symptoms: list[str]) -> bool:
    """
    Return True if the reported symptoms contain any complete red-flag
    combination.  Matching is case-insensitive and strips leading/trailing
    whitespace.
    """
    symptom_set = {s.lower().strip() for s in symptoms}
    return any(flag.issubset(symptom_set) for flag in RED_FLAGS)
