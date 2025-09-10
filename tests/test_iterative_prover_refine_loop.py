import re

import dspy

from dspy_lean_prover.hinting import HintMode
from dspy_lean_prover.modules import IterativeProver


def test_iterative_prover_refine_loop_exercised():
    # Oracle with three steps; propose will be wrong first, refine fixes it.
    oracle = ["intro a", "intro b", "refl"]
    prover = IterativeProver(max_steps=5, retries=2, hint_mode=HintMode.full)

    propose_calls = {"count": 0}
    refine_calls = {"count": 0}

    def propose_stub(goal_state: str, previous_tactics: str):
        # Always propose an incorrect tactic to trigger refine path
        propose_calls["count"] += 1
        return dspy.Prediction(proposed_tactic="sorry")

    def refine_stub(goal_state: str, failed_tactic: str, error_message: str, hint: str):
        # Use the goal_state to determine the current step and return the oracle tactic
        refine_calls["count"] += 1
        m = re.search(r"step (\d+)", goal_state)
        step_idx = int(m.group(1)) if m else 0
        return dspy.Prediction(refined_tactic=oracle[step_idx])

    prover.propose = propose_stub  # type: ignore
    prover.refine = refine_stub  # type: ignore

    pred = prover(theorem_id="thm_refine_loop", oracle_steps=oracle)

    assert pred.proof == oracle
    # Ensure refine loop actually ran at least once per step
    assert refine_calls["count"] >= len(oracle)

