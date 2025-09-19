from dspy_lean_prover.hinting import clip_hint, maybe_clip, scale_hint, HintMode


def test_clip_hint_basic():
    assert clip_hint("rw [Nat.add_assoc, Nat.add_comm]") == "rw"
    assert clip_hint("apply Nat.add_comm a b") == "apply"
    assert clip_hint("simp") == "simp"
    assert clip_hint("") == ""


def test_maybe_clip_modes():
    full = "rw [Nat.add_assoc]"
    assert maybe_clip(full, HintMode.full) == full
    assert maybe_clip(full, HintMode.clipped) == "rw"
    assert maybe_clip(full, HintMode.none) is None


def test_scale_hint_clipped_threshold():
    h = "rewrite Nat.add_comm"
    assert scale_hint(h, HintMode.clipped, 1.0) == "rewrite"
    assert scale_hint(h, HintMode.clipped, 0.51) == "rewrite"
    assert scale_hint(h, HintMode.clipped, 0.49) is None

def test_scale_hint_full_truncates_tokens():
    h = "rw [Nat.add_assoc] -- then refl"
    assert scale_hint(h, HintMode.full, 1.0) == h
    # With small strength, only the first token remains
    assert scale_hint(h, HintMode.full, 0.1) == "rw"
    # Mid strength should keep more than one token (deterministically rounded)
    mid = scale_hint(h, HintMode.full, 0.6)
    assert mid.startswith("rw ")
    assert len(mid.split()) >= 2


def test_scale_hint_clipped_threshold():
    h = "rewrite Nat.add_comm"
    assert scale_hint(h, HintMode.clipped, 1.0) == "rewrite"
    assert scale_hint(h, HintMode.clipped, 0.51) == "rewrite"
    assert scale_hint(h, HintMode.clipped, 0.49) is None
