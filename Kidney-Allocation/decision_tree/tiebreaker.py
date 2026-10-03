"""
Multi-step tie-breaker for ambiguous FFT predictions  (Priority 9 — prototype).
SURA 2026 · IIT Delhi

WHY THIS IS A SEPARATE MODULE
-----------------------------
This is an accuracy *investigation*, not a committed change to the model. It is
therefore built so that it can be switched on and off without touching the
model itself:

  * `fft_model.FastFrugalTree._row_predict` is NOT modified. The tree predicts
    exactly as it always did.
  * Nothing here runs unless `enabled()` returns True.
  * The only integration point is `predict_with_tiebreak()`, which callers opt
    into explicitly. When disabled it returns the plain tree prediction
    unchanged, so removing the feature is a one-line revert.

WHAT IT DOES
------------
When the tree reaches a decision by a hair — the deciding cue only barely
cleared its threshold, or nothing fired at all and the default leaf is close to
a coin flip — the call is treated as ambiguous. A short ordered chain of 2–3
extra cues (the "right-side" tie-breaker) is then consulted. The first cue in
the chain that separates the two patients decisively enough decides the case;
if none does, the tree's original answer stands.

Enable with the environment variable:

    FFT_TIEBREAKER=1            # off unless set to 1/true/yes/on
    FFT_TIEBREAKER_STEPS=3      # chain length, clamped to 2..3
"""

import os

import numpy as np

# ── configuration ────────────────────────────────────────────────────────────

#: A decision counts as ambiguous when the deciding cue cleared its threshold
#: by less than this many units.
AMBIGUOUS_MARGIN = 1.0

#: A fall-through to the default leaf counts as ambiguous when the leaf's
#: purity is below this (i.e. it is close to a coin flip).
AMBIGUOUS_PURITY = 0.65

#: A tie-breaker cue only decides if the two patients differ by at least this
#: much on it — otherwise it is no more informative than the cue that tied.
MIN_DECISIVE_GAP = 1.0

_TRUTHY = {"1", "true", "yes", "on"}


def enabled():
    """True when the tie-breaker prototype is switched on. Off by default."""
    return str(os.environ.get("FFT_TIEBREAKER", "")).strip().lower() in _TRUTHY


def max_steps():
    try:
        n = int(os.environ.get("FFT_TIEBREAKER_STEPS", 3))
    except (TypeError, ValueError):
        n = 3
    return max(2, min(3, n))


# ── learning the chain ───────────────────────────────────────────────────────

def fit_chain(F, y, feat_names):
    """
    Learn the ordered fallback cues from the same training data the tree saw.

    For each feature, work out which direction the participant tended to favour
    (higher value or lower value) and how reliably, then rank the features by
    that reliability. Returns a list of dicts, longest-supported cue first:

        [{"feature": "urgency_score_diff", "prefer_higher": True,
          "purity": 0.81, "gap": 1.0}, ...]

    Pure data summary — it neither reads nor writes tree state.
    """
    X = np.asarray(F.values if hasattr(F, "values") else F, dtype=float)
    y = np.asarray(y).astype(int)
    ranked = []
    if X.size == 0 or len(y) == 0:
        return ranked

    for idx, name in enumerate(feat_names):
        col = X[:, idx]
        moved = np.abs(col) >= MIN_DECISIVE_GAP
        if not moved.any():
            continue
        signs = np.sign(col[moved])
        picks = y[moved]
        # "Prefer higher" = the participant chose A exactly when A's value was
        # the larger one. Count how often each reading matches the answers.
        votes_higher = int(((signs > 0) == (picks == 1)).sum())
        votes_lower = int(moved.sum()) - votes_higher
        prefer_higher = votes_higher >= votes_lower
        agree = max(votes_higher, votes_lower)
        purity = agree / float(moved.sum())
        ranked.append({
            "feature":       name,
            "feature_idx":   idx,
            "prefer_higher": bool(prefer_higher),
            "purity":        float(purity),
            "support":       float(moved.mean()),
        })

    ranked.sort(key=lambda r: (r["purity"], r["support"]), reverse=True)
    return ranked


# ── ambiguity test ───────────────────────────────────────────────────────────

def is_ambiguous(tree, row, exit_index, purity):
    """
    Was this prediction a close call?

    Two ways to qualify: the deciding cue only just cleared its threshold, or
    nothing fired at all and the default leaf is near a coin flip.
    """
    row = np.asarray(row, dtype=float)
    if exit_index is None or exit_index < 0:
        return float(purity if purity is not None else 0.5) < AMBIGUOUS_PURITY

    try:
        node = tree.nodes[exit_index]
    except (IndexError, TypeError):
        return False

    x = float(row[node["feature_idx"]])
    thr = float(node["threshold"])
    margin = abs(x) - thr if node.get("use_abs") else abs(x - thr)
    return margin < AMBIGUOUS_MARGIN


# ── resolution ───────────────────────────────────────────────────────────────

def resolve(chain, row, exclude_features=(), steps=None):
    """
    Walk up to `steps` cues from `chain`, skipping any already used by the cue
    that tied. Returns (cls, trace) where cls is 1 (A) / 0 (B), or None when no
    cue in the chain was decisive. `trace` lists the cues considered, so the
    interface can show its working.
    """
    row = np.asarray(row, dtype=float)
    steps = max_steps() if steps is None else steps
    exclude = set(exclude_features or ())
    trace, used = [], 0

    for cue in chain:
        if used >= steps:
            break
        if cue["feature"] in exclude:
            continue
        used += 1
        x = float(row[cue["feature_idx"]])
        decisive = abs(x) >= MIN_DECISIVE_GAP
        entry = {
            "feature":  cue["feature"],
            "diff":     x,
            "decisive": bool(decisive),
        }
        if decisive:
            cls = 1 if (x > 0) == cue["prefer_higher"] else 0
            entry["cls"] = cls
            trace.append(entry)
            return cls, trace
        trace.append(entry)

    return None, trace


def predict_with_tiebreak(tree, row, chain=None):
    """
    The single integration point.

    Returns a dict:
        {"cls", "exit_index", "purity", "refine_branch",
         "ambiguous", "tiebreak_used", "tiebreak_trace"}

    With the prototype disabled (the default) this is the tree's own prediction
    with the extra keys set to False/empty — identical behaviour, no side
    effects on the tree.
    """
    row = np.asarray(row, dtype=float)
    cls, exit_index, purity, refine_branch = tree._row_predict(row)
    out = {
        "cls":            int(cls),
        "exit_index":     int(exit_index),
        "purity":         float(purity) if purity is not None else None,
        "refine_branch":  refine_branch,
        "ambiguous":      False,
        "tiebreak_used":  False,
        "tiebreak_trace": [],
    }
    if not enabled() or not chain:
        return out

    if not is_ambiguous(tree, row, exit_index, purity):
        return out
    out["ambiguous"] = True

    exclude = ()
    if 0 <= exit_index < len(tree.nodes):
        node = tree.nodes[exit_index]
        exclude = [node["feature"]]
        if node.get("refine"):
            exclude.append(node["refine"]["feature"])

    new_cls, trace = resolve(chain, row, exclude_features=exclude)
    out["tiebreak_trace"] = trace
    if new_cls is not None:
        out["cls"] = int(new_cls)
        out["tiebreak_used"] = True
    return out