import unittest

from app.safety.rules import detect_emergency, detect_red_flags


class SafetyTests(unittest.TestCase):
    # ------------------------------------------------------------------
    # Emergency detection
    # ------------------------------------------------------------------
    def test_detect_emergency_chest_pain(self):
        self.assertTrue(detect_emergency("severe chest pain and shortness of breath"))

    def test_detect_emergency_breathing(self):
        self.assertTrue(detect_emergency("I cannot breathe properly"))

    def test_detect_emergency_seizure(self):
        self.assertTrue(detect_emergency("she had a seizure and fell down"))

    def test_detect_emergency_stroke(self):
        self.assertTrue(detect_emergency("facial droop and slurred speech suddenly"))

    def test_detect_emergency_suicidal(self):
        self.assertTrue(detect_emergency("I feel suicidal tonight"))

    def test_detect_emergency_overdose(self):
        self.assertTrue(detect_emergency("overdose on medication"))

    def test_detect_emergency_negative(self):
        self.assertFalse(detect_emergency("mild seasonal allergy with sneezing"))

    def test_detect_emergency_negative_fatigue(self):
        self.assertFalse(detect_emergency("I feel tired and have a mild headache"))

    def test_detect_emergency_case_insensitive(self):
        self.assertTrue(detect_emergency("CHEST PAIN and SEIZURE"))

    # ------------------------------------------------------------------
    # Red-flag combinations
    # ------------------------------------------------------------------
    def test_meningitis_triad(self):
        self.assertTrue(detect_red_flags(["fever", "stiff neck", "severe headache"]))

    def test_meningitis_triad_mixed_case(self):
        self.assertTrue(detect_red_flags(["Fever", "Stiff Neck", "Severe Headache"]))

    def test_mi_triad(self):
        self.assertTrue(detect_red_flags(["chest pain", "left arm pain", "sweating"]))

    def test_stroke_triad(self):
        self.assertTrue(detect_red_flags(["confusion", "slurred speech", "facial droop"]))

    def test_dvt_pe_combo(self):
        self.assertTrue(detect_red_flags(["shortness of breath", "calf pain", "leg swelling"]))

    def test_partial_red_flag_not_triggered(self):
        # Only 2 of the 3 meningitis symptoms — should NOT trigger
        self.assertFalse(detect_red_flags(["fever", "stiff neck"]))

    def test_no_red_flags_common_cold(self):
        self.assertFalse(detect_red_flags(["runny nose", "mild cough", "sneezing"]))

    def test_empty_symptoms(self):
        self.assertFalse(detect_red_flags([]))


if __name__ == "__main__":
    unittest.main()
