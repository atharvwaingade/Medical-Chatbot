from __future__ import annotations

EMERGENCY_KEYWORDS = {
    "chest pain",
    "breathing difficulty",
    "shortness of breath",
    "unconscious",
    "unconsciousness",
    "seizure",
}

RED_FLAGS = [
    {"fever", "stiff neck", "severe headache"},
    {"chest pain", "left arm pain", "sweating"},
    {"confusion", "slurred speech", "facial droop"},
]


def detect_emergency(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in EMERGENCY_KEYWORDS)


def detect_red_flags(symptoms: list[str]) -> bool:
    symptom_set = {s.lower().strip() for s in symptoms}
    return any(flag.issubset(symptom_set) for flag in RED_FLAGS)
