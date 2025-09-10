from dspy_lean_prover.eval_utils import matches_oracle


def test_matches_oracle_with_variants():
    oracle = [
        "intro a",
        "intro b",
        [
            "rw [Nat.add_comm]",
            "rw Nat.add_comm",
            "rewrite Nat.add_comm",
        ],
        "refl",
    ]

    # Exact canonical form
    pred1 = ["intro a", "intro b", "rw [Nat.add_comm]", "refl"]
    assert matches_oracle(pred1, oracle)

    # Accept non-bracketed rw form
    pred2 = ["intro a", "intro b", "rw Nat.add_comm", "refl"]
    assert matches_oracle(pred2, oracle)

    # Accept rewrite synonym
    pred3 = ["intro a", "intro b", "rewrite Nat.add_comm", "refl"]
    assert matches_oracle(pred3, oracle)

    # Mismatch should fail
    pred4 = ["intro a", "intro b", "simp", "refl"]
    assert not matches_oracle(pred4, oracle)
