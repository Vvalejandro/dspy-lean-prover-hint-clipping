from __future__ import annotations

import random
from pathlib import Path
from typing import List, Dict, Union

import ujson as json


def thm(a: str) -> str:
    return a


def variants_rw(lemma: str) -> List[str]:
    # accepted by MockVerifier normalization
    return [
        f"rw [{lemma}]",
        f"rw {lemma}",
        f"rewrite {lemma}",
    ]


def build_examples(
    n_seed: int = 42
) -> List[Dict[str, Union[str, List[Union[str, List[str]]]]]]:
    rng = random.Random(n_seed)

    examples = []

    # Basic commutativity/associativity and identities
    lemmas = [
        ("add_comm", "Nat.add_comm"),
        ("mul_comm", "Nat.mul_comm"),
        ("add_assoc", "Nat.add_assoc"),
        ("mul_assoc", "Nat.mul_assoc"),
        ("add_zero", "Nat.add_zero"),
        ("zero_add", "Nat.zero_add"),
        ("mul_one", "Nat.mul_one"),
        ("one_mul", "Nat.one_mul"),
    ]

    # Commutativity and associativity with 2 or 3 intros
    for short, name in lemmas:
        tid = f"thm_{short}_2"
        steps: List[Union[str, List[str]]] = ["intro a", "intro b"]
        if "assoc" in short:
            steps.insert(2, "intro c")
        if short in {"add_zero", "zero_add", "mul_one", "one_mul"}:
            steps.append("simp")
        else:
            steps.append(variants_rw(name))
            steps.append("refl")
        examples.append({"theorem_id": tid, "oracle_steps": steps})

    # A few randomized variants
    for i in range(60):
        kind, lemma = rng.choice(lemmas[:4])  # prefer comm/assoc variants
        tid = f"thm_{kind}_{i}"
        arity = 3 if "assoc" in kind else 2
        steps: List[Union[str, List[str]]] = ["intro a", "intro b"]
        if arity == 3:
            steps.append("intro c")
        steps.append(variants_rw(lemma))
        steps.append("refl")
        examples.append({"theorem_id": tid, "oracle_steps": steps})

    return examples


def main(noise_level: float = 0.0):
    """Generates a mock theorem dataset with optional noise."""
    out = Path("data/mock_theorems_v2.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    examples = build_examples()

    noise_pool = [
        "intro x",
        "simp",
        "refl",
        "rw [Nat.add_comm]",
        "rw [Nat.mul_comm]",
        "sorry",
    ]

    for ex in examples:
        original_steps = ex["oracle_steps"]
        noisy_steps = []
        for step in original_steps:
            if random.random() < noise_level:
                noisy_steps.append(random.choice(noise_pool))
            else:
                noisy_steps.append(step)
        ex["noisy_oracle_steps"] = noisy_steps

    out.write_text(json.dumps(examples, indent=2))
    print(f"Wrote {len(examples)} examples to {out}")


if __name__ == "__main__":
    import typer
    typer.run(main)
