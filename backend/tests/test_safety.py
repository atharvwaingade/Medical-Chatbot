from app.safety.rules import detect_emergency, detect_red_flags


def test_detect_emergency_keywords():
    assert detect_emergency("severe chest pain and breathing difficulty") is True
    assert detect_emergency("mild seasonal allergy") is False


def test_detect_red_flags_combo():
    symptoms = ["Fever", "stiff neck", "severe headache"]
    assert detect_red_flags(symptoms) is True
