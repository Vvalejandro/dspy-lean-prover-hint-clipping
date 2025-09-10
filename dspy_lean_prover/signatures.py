import dspy


class ProposeTactic(dspy.Signature):
    """Propose a Lean tactic to advance the proof.

    Guidelines:
    - Emit exactly one next-step tactic.
    - Prefer introducing variables first (e.g., `intro a`, `intro b`, ...).
    - Then consider rewrites (e.g., `rw [Nat.add_comm]`).
    - Finish with `refl` or `simp` when the goal is trivial.
    - Do not explain; output only the tactic string.
    """

    goal_state = dspy.InputField(desc="Pretty-printed current goal.")
    previous_tactics = dspy.InputField(desc="History of tactics applied so far.")
    proposed_tactic = dspy.OutputField(desc="The next tactic to try.")


class RefineTactic(dspy.Signature):
    """Refine a failed tactic using error feedback (and optional hint).

    Guidelines:
    - Keep changes minimal; correct the previous tactic.
    - Leverage the hint when present; if clipped, follow its head (e.g., `rw`).
    - Return only a single corrected tactic string.
    """

    goal_state = dspy.InputField(desc="Pretty-printed current goal.")
    failed_tactic = dspy.InputField(desc="The tactic that failed.")
    error_message = dspy.InputField(desc="Error from verifier.")
    hint = dspy.InputField(desc="Optional high-level hint.")
    refined_tactic = dspy.OutputField(desc="A corrected tactic to try next.")


class SolveTheorem(dspy.Signature):
    """Solve a theorem by producing a sequence of tactics.

    Output must be a list of tactic strings in order.
    """

    theorem_id = dspy.InputField(desc="Identifier of theorem or file:line for Lean.")
    proof = dspy.OutputField(desc="List of tactics that solve the theorem.")
