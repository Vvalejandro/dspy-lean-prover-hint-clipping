import unittest

from dspy_lean_prover.hinting import clip_hint, maybe_clip, HintMode


class HintingTests(unittest.TestCase):
    def test_clip_hint(self):
        self.assertEqual(clip_hint("rw [Nat.add_assoc, Nat.add_comm]"), "rw")
        self.assertEqual(clip_hint("apply Nat.add_comm a b"), "apply")
        self.assertEqual(clip_hint("simp"), "simp")
        self.assertEqual(clip_hint(""), "")

    def test_maybe_clip(self):
        full = "rw [Nat.add_assoc]"
        self.assertEqual(maybe_clip(full, HintMode.full), full)
        self.assertEqual(maybe_clip(full, HintMode.clipped), "rw")
        self.assertIsNone(maybe_clip(full, HintMode.none))


if __name__ == "__main__":
    unittest.main()

