from __future__ import annotations

import dspy
import random
from typing import Callable, Dict, List, Optional

from .verifier import BaseVerifier, MockVerifier, LeanVerifier, VerifierFactory
from .signatures import ProposeTactic, RefineTactic, SolveTheorem
from .hinting import HintMode, scale_hint


class IterativeProver(dspy.Module):
    def __init__(
        self,
        max_steps: int = 10,
        retries: int = 2,
        hint_mode: HintMode = HintMode.none,
        hint_strength: float = 1.0,
        hint_noise_p: float = 0.0,
        hint_sparse: bool = False,
        force_refine_p: float = 0.0,
        verifier_factory: Optional[VerifierFactory] = None,
    ):
        super().__init__()
        self.propose = dspy.ChainOfThought(ProposeTactic)
        self.refine = dspy.ChainOfThought(RefineTactic)
        self.max_steps = max_steps
        self.retries = retries
        self.hint_mode = hint_mode
        self.hint_strength = hint_strength
        self.hint_noise_p = hint_noise_p
        self.hint_sparse = hint_sparse
        self.force_refine_p = force_refine_p
        self.verifier_factory = verifier_factory

    def _make_verifier(self, **kwargs) -> BaseVerifier:
        if self.verifier_factory is not None:
            return self.verifier_factory(**kwargs)
        # Default to mock for safety
        return MockVerifier(**kwargs)

    def forward(
        self,
        theorem_id: str,
        oracle_steps: Optional[List] = None,
        repo_path: Optional[str] = None,
        file_path: Optional[str] = None,
        line_num: Optional[int] = None,
    ) -> dspy.Prediction:
        use_mock = oracle_steps is not None
        if use_mock:
            verifier = self._make_verifier(theorem_id=theorem_id, oracle_steps=oracle_steps)
        else:
            if not (repo_path and file_path and line_num is not None):
                raise ValueError("repo_path, file_path, line_num required for LeanVerifier")
            verifier = self._make_verifier(repo_path=repo_path, file_path=file_path, line_num=line_num)

        tactics_history: List[str] = []
        current_goal = verifier.pp_goal()

        for _ in range(self.max_steps):
            pred = self.propose(goal_state=current_goal, previous_tactics=str(tactics_history))
            tactic = pred.proposed_tactic

            # Optional: force errors to exercise refine loop
            import random as _random
            if self.force_refine_p > 0.0 and _random.random() < self.force_refine_p:
                tactic = "sorry"

            res = verifier.run_tactic(tactic)

            remaining_retries = self.retries
            while res.status == "error" and remaining_retries > 0:
                # Build an optional hint using oracle (if available),
                # optionally add sparsity/noise, then scale by strength
                hint_val: Optional[str] = None
                if oracle_steps is not None and len(tactics_history) < len(oracle_steps):
                    step_item = oracle_steps[len(tactics_history)]
                    # If the dataset provides multiple acceptable variants, pick the first for hinting
                    if isinstance(step_item, list) and step_item:
                        step_item = step_item[0]
                    base_hint = str(step_item)
                    # Optional sparsity (head-only) when using full mode
                    if self.hint_sparse and self.hint_mode == HintMode.full:
                        from .hinting import clip_hint

                        base_hint = clip_hint(base_hint)
                    # Optional noise injection
                    if self.hint_noise_p > 0.0 and random.random() < self.hint_noise_p:
                        noise_pool = [
                            "intro a",
                            "intro b",
                            "simp",
                            "refl",
                            "rw [Nat.add_comm]",
                            "rw [Nat.mul_comm]",
                            "rw [Nat.add_assoc]",
                            "rw [Nat.mul_assoc]",
                        ]
                        base_hint = random.choice(noise_pool)
                    hint_val = scale_hint(base_hint, self.hint_mode, self.hint_strength)

                # Optional hint logging for visibility; no DSPy assertion primitives required
                _ = hint_val  # hint may be empty

                refined = self.refine(
                    goal_state=current_goal,
                    failed_tactic=tactic,
                    error_message=res.message,
                    hint=hint_val or "",
                )
                tactic = refined.refined_tactic
                res = verifier.run_tactic(tactic)
                remaining_retries -= 1

            if res.status != "success":
                raise RuntimeError(f"Failed to find a working tactic for goal: {current_goal}")
            tactics_history.append(tactic)

            if "Proof complete!" in res.message or not verifier.goals_remaining:
                return dspy.Prediction(proof=tactics_history)

            current_goal = res.goal or current_goal

        return dspy.Prediction(proof=None)


class SolveWithIterativeProver(dspy.Module):
    def __init__(self, prover: IterativeProver):
        super().__init__()
        self.prover = prover
        self.solver = dspy.Predict(SolveTheorem)

    def forward(self, theorem_id: str, oracle_steps: Optional[List[str]] = None, **kwargs):
        pred = self.prover(
            theorem_id=theorem_id,
            oracle_steps=oracle_steps,
            **kwargs,
        )
        return dspy.Prediction(proof=pred.proof)
