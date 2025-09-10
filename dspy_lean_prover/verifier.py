from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable, Union


class VerificationError(Exception):
    pass


@dataclass
class VerifierResult:
    status: str  # "success" | "error"
    message: str
    goal: Optional[str] = None


class BaseVerifier:
    def pp_goal(self) -> str:
        raise NotImplementedError

    def run_tactic(self, tactic: str) -> VerifierResult:
        raise NotImplementedError

    @property
    def goals_remaining(self) -> bool:
        raise NotImplementedError


class MockVerifier(BaseVerifier):
    """A lightweight verifier for experiments without Lean.

    The mock verifier simulates a proof as a sequence of tactic strings.
    - goal text is a simple synthetic string per step for visibility
    - if a tactic matches the next oracle step, progress; otherwise error
    - can emit delibrately informative error messages
    """

    def __init__(self, theorem_id: str, oracle_steps: List[Union[str, List[str]]]):
        self.theorem_id = theorem_id
        self.oracle_steps = oracle_steps
        self.step = 0

    def pp_goal(self) -> str:
        if self.step >= len(self.oracle_steps):
            return "⊢ ⊤"
        return f"goal[{self.theorem_id}]: step {self.step}"

    def _normalize(self, s: str) -> str:
        t = s.strip()
        # simple synonyms
        if t == "rfl":
            t = "refl"
        if t.startswith("simp"):
            # unify various simp variants (e.g., simp_all, simp [lemma])
            t = "simp"
        # normalize common synonyms and bracket styles for rw
        if t.startswith("rewrite "):
            t = "rw " + t[len("rewrite "):]
        if t.startswith("rw ") and "[" not in t and "]" not in t:
            # allow `rw lemma` -> `rw [lemma]`
            parts = t.split()
            if len(parts) == 2:
                t = f"rw [{parts[1]}]"
        return t

    def run_tactic(self, tactic: str) -> VerifierResult:
        if self.step >= len(self.oracle_steps):
            return VerifierResult(status="success", message="Proof complete!", goal=None)

        expected_item = self.oracle_steps[self.step]
        candidates: List[str]
        if isinstance(expected_item, list):
            candidates = [self._normalize(x) for x in expected_item]
        else:
            candidates = [self._normalize(expected_item)]

        tac_norm = self._normalize(tactic)
        if tac_norm in candidates:
            self.step += 1
            if self.step >= len(self.oracle_steps):
                return VerifierResult(status="success", message="Proof complete!", goal=None)
            return VerifierResult(status="success", message=f"New Goal: {self.pp_goal()}", goal=self.pp_goal())

        # Error with a simple heuristic hint in message
        hint_kw = candidates[0].split(" ")[0]
        return VerifierResult(
            status="error",
            message=f"tactic failed; expected something like '{hint_kw}'. got '{tactic}'.",
            goal=self.pp_goal(),
        )

    @property
    def goals_remaining(self) -> bool:
        return self.step < len(self.oracle_steps)


class LeanVerifier(BaseVerifier):
    """LeanDojo-backed verifier wrapper.

    This class defers imports of lean_dojo so it remains optional.
    It expects a Lean repository checked out and a theorem position.
    """

    def __init__(self, repo_path: str, file_path: str, line_num: int):
        try:
            from lean_dojo import LeanGitRepo, TacticState
        except Exception as e:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "lean-dojo is not installed or not usable in this environment"
            ) from e

        repo = LeanGitRepo(repo_path)
        self._TacticState = TacticState
        self.state = TacticState(repo, file_path, line_num)

    def pp_goal(self) -> str:
        return str(self.state.pp)

    def run_tactic(self, tactic: str) -> VerifierResult:
        try:
            result = self.state.run_tac(tactic)
        except Exception as e:  # pragma: no cover - runtime dependent
            return VerifierResult(status="error", message=str(e), goal=self.pp_goal())

        if isinstance(result, self._TacticState):
            self.state = result
            if not getattr(self.state, "goals", None):  # type: ignore[attr-defined]
                return VerifierResult(status="success", message="Proof complete!", goal=None)
            return VerifierResult(status="success", message=f"New Goal: {self.pp_goal()}", goal=self.pp_goal())

        return VerifierResult(status="error", message="Invalid tactic result.", goal=self.pp_goal())

    @property
    def goals_remaining(self) -> bool:
        goals = getattr(self.state, "goals", None)
        return bool(goals)


VerifierFactory = Callable[..., BaseVerifier]
