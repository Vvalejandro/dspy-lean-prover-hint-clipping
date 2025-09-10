from typing import List, Union


def normalize_tactic(s: str) -> str:
    """Normalize common tactic spelling variants for fair matching.

    - "rewrite X" -> "rw X"
    - "rw lemma" -> "rw [lemma]" (if no brackets present)
    """
    t = s.strip()
    if t == "rfl":
        t = "refl"
    if t.startswith("simp"):
        t = "simp"
    if t.startswith("rewrite "):
        t = "rw " + t[len("rewrite "):]
    if t.startswith("rw ") and "[" not in t and "]" not in t:
        parts = t.split()
        if len(parts) == 2:
            t = f"rw [{parts[1]}]"
    return t


def matches_oracle(
    predicted: List[str], oracle: List[Union[str, List[str]]]
) -> bool:
    """Return True if predicted steps match oracle allowing variants per step."""
    if len(predicted) != len(oracle):
        return False
    for p, expected in zip(predicted, oracle):
        p_norm = normalize_tactic(p)
        if isinstance(expected, list):
            options = [normalize_tactic(x) for x in expected]
            if p_norm not in options:
                return False
        else:
            if p_norm != normalize_tactic(expected):
                return False
    return True
