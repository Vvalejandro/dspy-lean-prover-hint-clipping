from enum import Enum
from typing import Optional


class HintMode(str, Enum):
    none = "none"
    full = "full"
    clipped = "clipped"


def clip_hint(full_tactic_line: str) -> str:
    s = full_tactic_line.strip()
    if not s:
        return s
    head = s.split(" ")[0]
    return head


def maybe_clip(hint: Optional[str], mode: HintMode) -> Optional[str]:
    if hint is None:
        return None
    if mode == HintMode.clipped:
        return clip_hint(hint)
    if mode == HintMode.full:
        return hint
    return None


def scale_hint(hint: Optional[str], mode: HintMode, strength: float) -> Optional[str]:
    """Return a hint string modulated by a scalar strength in [0, 1].

    - none: always returns None.
    - clipped: returns head token if strength >= 0.5 else None.
    - full: returns a truncated token prefix proportional to strength
      (>=1 token when strength > 0).
    """
    if hint is None or mode == HintMode.none:
        return None

    s = max(0.0, min(1.0, float(strength)))
    if s <= 0.0:
        return None

    if mode == HintMode.clipped:
        return clip_hint(hint) if s >= 0.5 else None

    # mode == full: proportionally truncate tokens
    base = hint.strip()
    if not base:
        return base
    toks = base.split()
    keep = max(1, min(len(toks), int(round(s * len(toks)))))
    return " ".join(toks[:keep])
