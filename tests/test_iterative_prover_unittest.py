import ast
import unittest

import dspy

from dspy_lean_prover.hinting import HintMode
from dspy_lean_prover.modules import IterativeProver


class IterativeProverTests(unittest.TestCase):
    def test_with_stubbed_modules(self):
        oracle = ["intro a", "intro b", "refl"]
        prover = IterativeProver(max_steps=5, retries=1, hint_mode=HintMode.none)

        def propose_stub(goal_state: str, previous_tactics: str):
            hist = ast.literal_eval(previous_tactics) if previous_tactics else []
            idx = len(hist)
            return dspy.Prediction(proposed_tactic=oracle[idx])

        def refine_stub(goal_state: str, failed_tactic: str, error_message: str, hint: str):
            # Return the next expected tactic regardless
            return dspy.Prediction(refined_tactic=oracle[0])

        # Monkeypatch the DSPy modules with simple stubs
        prover.propose = propose_stub  # type: ignore
        prover.refine = refine_stub  # type: ignore

        pred = prover(theorem_id="thm_stub", oracle_steps=oracle)
        self.assertEqual(pred.proof, oracle)


if __name__ == "__main__":
    unittest.main()

