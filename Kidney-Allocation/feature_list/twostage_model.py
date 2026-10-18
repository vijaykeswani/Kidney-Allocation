"""
Two-stage cognitively-faithful decision model.
SURA 2026 · IIT Delhi

Based on "Towards Cognitively-Faithful Decision-Making Models to Improve AI
Alignment" (Cousins, Keswani, Conitzer, Heidari, Schaich Borg, Sinnott-
Armstrong; arXiv:2509.04445, ICLR 2026): each feature is first passed through
its own learned monotonic rule (a direction, a shape, and a strength),
then the per-feature scores are combined by a *fixed* aggregation rule —
a weighted sum, a tally (count how many features favor each side), or
letting the single most important feature decide.

This is an additional model *type*, not a replacement for the FFT
(fft_model.py). A participant is assigned one or the other once, at the
start of the study (see app.py's model_type handling), and stays on it.

Unlike the FFT, which trains on A-minus-B *differences*, this model needs
each patient's raw feature values separately (some rules, like "only past a
point", depend on the absolute value, not just the gap) — so training reads
straight from the A_*/B_* columns already stored per decision, rather than
going through fft_model.build_difference_features.
"""

import math

import numpy as np

import features

# Strength buckets 0 (off) .. 3 (a lot) -> score multiplier.
AMP = [0.0, 1.0, 2.0, 3.0]   # strengths 1/2/3 = "a little"/"somewhat"/"a lot", evenly spaced so the on-screen scale is the actual weight
GAIN = 0.9          # sharpness of the "weighted"/"prominent" verdict sigmoid
TALLY_GAIN = 0.95   # sharpness of the "tally" verdict sigmoid

MIN_DECISIONS = 6

# Discretization cutoffs used when reading a fitted curve into an editable rule:
#   a feature is shown as "no effect" if its fitted curve's total rise is below OFF_RANGE;
#   it's shown as "only past a point" if one bin holds more than STEP_SHARE of that rise.
OFF_RANGE = 0.8
STEP_SHARE = 0.6

# A feature WITHOUT a direction prior only starts "on" if its raw correlation with the
# participant's choices (about -1..1) is at least this strong; with ~20 answers a weaker
# correlation is mostly noise, which is how a feature can end up with the wrong sign.
SIGN_EVIDENCE_MIN = 0.15
# How much steeper one end of a curve must be than the other to be called "first/last counts most".
SHAPE_SLOPE_RATIO = 1.4


def _softplus(r):
    return np.log1p(np.exp(-np.abs(r))) + np.maximum(r, 0)


def _dsp(r):
    """Derivative of softplus = sigmoid."""
    return 1.0 / (1.0 + np.exp(-r))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _extract_queries(decisions):
    """decisions -> list of (a_dict, b_dict, y); y=1 means Patient A was chosen."""
    keys = features.FEATURE_KEYS
    out = []
    for d in decisions:
        if d.get("choice") not in ("A", "B"):
            continue
        a = {k: float(d.get(f"A_{k}", 0.0)) for k in keys}
        b = {k: float(d.get(f"B_{k}", 0.0)) for k in keys}
        out.append((a, b, 1 if d["choice"] == "A" else 0))
    return out


# ── Stage 1 training: per-feature monotonic step function ────────────────────
# A direct translation of the reference prototype's train()/curveToParams(),
# same hyperparameters, so behaviour matches what was validated there.

# A feature's curve is fit over its *levels* — the values it can actually take
# (features.json "values"; for a feature without one, every integer in its
# range). One increment is learned per gap between consecutive levels, so
# life_years_gained {5,10,15,20,25} has 4 parameters, not 20 unit steps.
# If a feature has more than MAX_BINS gaps, adjacent gaps are grouped into at
# most MAX_BINS shared parameters: with only ~20 noisy answers that cuts
# variance (see SIMULATION_NOTES.md). With the current value sets every
# feature has <= 4 gaps, so each gap gets its own parameter.
MAX_BINS = 5


def _levels(key):
    entry = features.FEATURE_BY_KEY[key]
    vals = entry.get("values")
    if vals:
        return np.array(sorted(float(v) for v in vals))
    return np.arange(int(entry["min"]), int(entry["max"]) + 1, dtype=float)


def _level_index(levels, v):
    """Index of the level nearest to v (answers always sit exactly on a level)."""
    return int(np.argmin(np.abs(levels - float(v))))


def _bin_layout(key):
    """(gap->bin index, bin sizes in gaps, bin edges as level indices) for a feature."""
    n = len(_levels(key)) - 1
    if n <= 0:
        return np.zeros(0, dtype=int), np.ones(1), [0, 0]
    nb = min(MAX_BINS, n)
    binof = np.minimum((np.arange(n) * nb) // n, nb - 1)
    width = np.bincount(binof, minlength=nb).astype(float)
    edges = [0] + [j for j in range(1, n) if binof[j] != binof[j - 1]] + [n]
    return binof, width, edges


def _direction_prior(key):
    """+1 / -1 if features.json fixes this feature's starting direction, else None."""
    p = features.FEATURE_BY_KEY[key].get("direction_prior")
    return 1.0 if p == "favor" else (-1.0 if p == "against" else None)


def _sign_evidence(queries):
    """Per-feature correlation-like evidence in [-1, 1] between (A - B) gap and the choice."""
    ys = np.array([y - 0.5 for a, b, y in queries], dtype=float)
    out = {}
    for k in features.FEATURE_KEYS:
        lv = _levels(k)
        span = float(lv[-1] - lv[0]) or 1.0
        x = np.array([(a[k] - b[k]) / span for a, b, y in queries], dtype=float)
        denom = float(np.sqrt((x ** 2).sum() * (ys ** 2).sum()))
        out[k] = float((x * ys).sum() / denom) if denom > 0 else 0.0
    return out


def _fit_curves(queries, lr=0.15, lam=0.05, iters=700, gain=GAIN):
    """
    Jointly fits one monotonic step function per feature against the same set
    of pairwise comparisons (the prediction for a query sums every feature's
    contribution, so each feature's curve is shaped by how well the *whole*
    model currently explains the data — not fit alone).

    Returns (learned, polarity):
      learned[key] -> list with one entry per level of the feature: the
                      sign-adjusted cumulative score at that level.
      polarity[key] -> +1 or -1, which direction the feature initially leans.
    """
    keys = features.FEATURE_KEYS
    levels = {k: _levels(k) for k in keys}
    gaps = {k: len(levels[k]) - 1 for k in keys}
    layout = {k: _bin_layout(k) for k in keys}
    nb = {k: (int(layout[k][0].max()) + 1 if gaps[k] > 0 else 0) for k in keys}

    # level index of every patient value in every query, once
    ia = {k: np.array([_level_index(levels[k], a[k]) for a, b, y in queries]) for k in keys}
    ib = {k: np.array([_level_index(levels[k], b[k]) for a, b, y in queries]) for k in keys}

    raw = {k: np.zeros(nb[k]) for k in keys}
    polarity = {}
    for k in keys:
        c = sum((a[k] - b[k]) * (y - 0.5) for a, b, y in queries)
        prior = _direction_prior(k)
        polarity[k] = prior if prior is not None else (1.0 if c >= 0 else -1.0)

    def cumulative(k):
        C = np.zeros(gaps[k] + 1)
        if gaps[k] > 0:
            binof, width, _ = layout[k]
            C[1:] = np.cumsum(_softplus(raw[k])[binof] / width[binof])
        return C

    N = max(1, len(queries))
    ys = np.array([y for a, b, y in queries], dtype=float)
    for _ in range(iters):
        cum = {k: cumulative(k) for k in keys}
        z = np.zeros(len(queries))
        for k in keys:
            z += polarity[k] * (cum[k][ia[k]] - cum[k][ib[k]])
        err = _sigmoid(gain * z) - ys
        for k in keys:
            if gaps[k] == 0:
                continue
            binof, width, _ = layout[k]
            ks = np.arange(gaps[k])
            ind = (ia[k][:, None] > ks[None, :]).astype(float) - (ib[k][:, None] > ks[None, :]).astype(float)
            grad_gap = (err[:, None] * gain * polarity[k] * ind).sum(axis=0)
            gb = np.bincount(binof, weights=grad_gap / width[binof], minlength=nb[k])
            sp = _softplus(raw[k])
            raw[k] = raw[k] - lr * (gb / N + 2 * lam * sp) * _dsp(raw[k])

    learned = {k: (polarity[k] * cumulative(k)).tolist() for k in keys}
    return learned, polarity


def _curve_to_params(key, curve, polarity):
    """
    Reads the continuous fitted step function off into one of three editable
    shapes plus a 0-3 strength bucket — showing someone the raw curve isn't
    editable, but "first counts most, somewhat strongly" is. Shape is judged
    from the increments between the bin edges the curve was fit on, and an
    "only past a point" cutoff is always one of the feature's real values.
    """
    levels = _levels(key)
    S = np.asarray(curve, dtype=float)
    mx = float(np.max(np.abs(S))) if S.size else 0.0
    rng = float(np.max(S) - np.min(S)) if S.size else 0.0

    if rng < OFF_RANGE:
        matters = 0
    elif mx < 1.1:
        matters = 1
    elif mx < 2.1:
        matters = 2
    else:
        matters = 3

    direction = "favor" if polarity >= 0 else "against"
    mode = "steady"
    cutoff = float(levels[1] if len(levels) > 1 else levels[0])

    if matters > 0 and S.size > 1:
        _, _, edges = _bin_layout(key)
        inc = np.abs(np.diff(S[edges]))
        total = float(inc.sum())
        biggest = float(inc.max()) if inc.size else 0.0
        if inc.size > 1 and total > 0 and biggest > STEP_SHARE * total:
            mode = "anyvsnone"
            j = int(np.argmax(inc))
            cutoff = float(levels[(edges[j] + edges[j + 1] + 1) // 2])
        else:
            # Judge the shape from slopes (score per unit of the feature), not raw increments:
            # levels are not evenly spaced (e.g. rejection risk 1,2,5,10,15), and a steady
            # rule means equal slopes.
            gaps = np.diff(levels[edges])
            slopes = inc / np.where(gaps > 0, gaps, 1.0)
            if slopes.size > 2 and slopes.max() > 0:
                falling = all(slopes[i] <= slopes[i - 1] * 1.1 for i in range(1, slopes.size))
                rising = all(slopes[i] >= slopes[i - 1] / 1.1 for i in range(1, slopes.size))
                if falling and slopes[0] >= SHAPE_SLOPE_RATIO * max(slopes[-1], 1e-9):
                    mode = "first"
                elif rising and slopes[-1] >= SHAPE_SLOPE_RATIO * max(slopes[0], 1e-9):
                    mode = "last"

    others = [k for k in features.FEATURE_KEYS if k != key]
    return {
        "matters": matters,
        "dir": direction,
        "mode": mode,
        "cutoff": cutoff,
        "cond": {"on": False, "when": (others[0] if others else key), "test": "nonzero", "n": 1},
    }


def _default_model():
    """An all-zero model (every feature off) — used only if training can't run."""
    return {
        "aggregation": "weighted",
        "features": {
            k: {"matters": 0, "dir": "favor", "mode": "steady", "cutoff": float(_levels(k)[1] if len(_levels(k)) > 1 else _levels(k)[0]),
                "cond": {"on": False, "when": k, "test": "nonzero", "n": 1}}
            for k in features.FEATURE_KEYS
        },
    }


# ── Stage 2 (fixed aggregation) + evaluation ──────────────────────────────────

def _amp_for(matters):
    """
    AMP[1..3] ("a little"/"somewhat"/"a lot") as fixed anchor points, with the
    fine-grained strength slider (see twostage.html/twostage_v2.html) able to
    land anywhere between them — the three buttons sit at equal 1-unit steps
    on that same 1..3 scale, so a slider drag just linearly interpolates
    between whichever two buttons it's currently between.
    """
    if matters <= 0:
        return 0.0
    if matters <= 1:
        return AMP[1] * matters
    if matters >= 3:
        return AMP[3]
    if matters <= 2:
        return AMP[1] + (AMP[2] - AMP[1]) * (matters - 1)
    return AMP[2] + (AMP[3] - AMP[2]) * (matters - 2)


def edit_value(model, key, v):
    """This feature's signed contribution for one patient's raw value v."""
    st = model["features"].get(key)
    if not st or st["matters"] == 0:
        return 0.0
    lo, hi = features.ranges()[key]
    if st["mode"] == "anyvsnone":
        mag = 1.0 if v >= st["cutoff"] else 0.0
    elif st["mode"] == "first":
        mag = math.log1p(max(0.0, v - lo)) / math.log1p(hi - lo) if hi > lo else 0.0
    elif st["mode"] == "last":
        mag = 1.0 - math.log1p(max(0.0, hi - v)) / math.log1p(hi - lo) if hi > lo else 0.0
    else:
        mag = (v - lo) / (hi - lo) if hi > lo else 0.0
    sign = 1.0 if st["dir"] == "favor" else -1.0
    return _amp_for(st["matters"]) * mag * sign


def cond_pass(model, key, patient):
    c = model["features"][key]["cond"]
    if not c.get("on"):
        return True
    w = patient.get(c["when"], 0)
    if c["test"] == "zero":
        return w == 0
    if c["test"] == "nonzero":
        return w != 0
    return w >= c.get("n", 1)


def contrib(model, key, patient):
    return edit_value(model, key, patient.get(key, 0)) if cond_pass(model, key, patient) else 0.0


def predict_prob(model, a, b):
    """P(patient A is chosen), under the model's current aggregation rule."""
    agg = model.get("aggregation", "weighted")
    keys = features.FEATURE_KEYS
    if agg == "tally":
        pa = pb = 0
        for k in keys:
            if model["features"][k]["matters"] == 0:
                continue
            sa, sb = contrib(model, k, a), contrib(model, k, b)
            if abs(sa - sb) < 1e-9:
                continue
            if sa > sb:
                pa += 1
            else:
                pb += 1
        return _sigmoid(TALLY_GAIN * (pa - pb))
    if agg == "prominent":
        order = sorted(keys, key=lambda k: _amp_for(model["features"][k]["matters"]), reverse=True)
        for k in order:
            if model["features"][k]["matters"] == 0:
                continue
            sa, sb = contrib(model, k, a), contrib(model, k, b)
            if abs(sa - sb) > 1e-9:
                return 0.85 if sa > sb else 0.15
        return 0.5
    z = sum(contrib(model, k, a) - contrib(model, k, b) for k in keys)
    return _sigmoid(GAIN * z)


def predict(model, a, b):
    """1 if Patient A is predicted, else 0."""
    return 1 if predict_prob(model, a, b) > 0.5 else 0


def exit_reason(model, a, b):
    """One short line on why the model landed where it did — mirrors
    fft_model.explain_prediction's role for the tree."""
    agg = model.get("aggregation", "weighted")
    keys = features.FEATURE_KEYS
    if agg == "tally":
        pa = pb = 0
        for k in keys:
            if model["features"][k]["matters"] == 0:
                continue
            sa, sb = contrib(model, k, a), contrib(model, k, b)
            if abs(sa - sb) < 1e-9:
                continue
            if sa > sb:
                pa += 1
            else:
                pb += 1
        return f"Tally: {pa} factor(s) favored A, {pb} favored B."
    if agg == "prominent":
        order = sorted(keys, key=lambda k: _amp_for(model["features"][k]["matters"]), reverse=True)
        for k in order:
            if model["features"][k]["matters"] == 0:
                continue
            sa, sb = contrib(model, k, a), contrib(model, k, b)
            if abs(sa - sb) > 1e-9:
                label = features.FEATURE_BY_KEY[k]["label"]
                return f"Decided by the most important factor: {label}."
        return "No factor discriminated between the two patients."
    return "Decided by the weighted sum of every factor's contribution."


# ── Plain-English summary ─────────────────────────────────────────────────────

def _noun(key):
    return features.FEATURE_BY_KEY[key]["gap_phrase"].replace(" gap", "")


def _fmt_num(v):
    v = float(v)
    return str(int(v)) if v == int(v) else str(v)


def feature_sentence(model, key):
    st = model["features"][key]
    lo, hi = features.ranges()[key]
    noun = _noun(key)
    if st["matters"] == 0:
        return f"You don't take {noun} into account."
    side = "in a patient's favor" if st["dir"] == "favor" else "against a patient"
    if st["mode"] == "anyvsnone":
        if st["cutoff"] <= (_levels(key)[1] if len(_levels(key)) > 1 else _levels(key)[0]):
            s = f"Having any {noun} at all counts {side}."
        else:
            s = f"Reaching {_fmt_num(st['cutoff'])} or more {noun} counts {side}, while less makes no difference."
    elif st["mode"] == "first":
        s = f"More {noun} counts {side}, and the first bit matters the most."
    elif st["mode"] == "last":
        s = f"More {noun} counts {side}, and the last bit matters the most."
    else:
        s = f"More {noun} counts {side}, with each additional bit mattering the same."
    if st["cond"]["on"]:
        w_noun = _noun(st["cond"]["when"])
        test = st["cond"]["test"]
        if test == "zero":
            cond_txt = f"has no {w_noun}"
        elif test == "nonzero":
            cond_txt = f"has any {w_noun}"
        else:
            cond_txt = f"has at least {_fmt_num(st['cond']['n'])} {w_noun}"
        s += f" This only applies when the patient {cond_txt}."
    return s


def aggregation_sentence(agg):
    if agg == "tally":
        return ("When choosing between two patients, you count how many factors favor "
                "each one, then pick whoever has more.")
    if agg == "prominent":
        return ("When choosing between two patients, you pick whoever is favored by "
                "the single most important factor.")
    return ("When choosing between two patients, you weigh how much each factor counts "
            "for or against each of them, then pick whoever comes out ahead overall.")


def summary_text(model):
    lines = [feature_sentence(model, k) for k in features.FEATURE_KEYS]
    lines.append(aggregation_sentence(model.get("aggregation", "weighted")))
    return " ".join(lines)


# ── Main entry point (mirrors fft_model.train_fft's calling convention) ──────

def _is_valid_twostage_model(model):
    """
    Guards against a stale/mismatched override — both model types currently
    share one override column per user (see app.py), so if this username
    previously saved a tree-shaped override (or anything else unexpected),
    we should silently retrain fresh rather than crash on a missing key.
    """
    if not isinstance(model, dict):
        return False
    feats = model.get("features")
    if not isinstance(feats, dict):
        return False
    return all(k in feats and isinstance(feats[k], dict) for k in features.FEATURE_KEYS)


def train_twostage(decisions, override=None):
    """
    decisions: list of decision dicts (same shape app.py already has for FFT).
    override: a previously-saved/edited model dict, or None to train fresh.
    Returns (stats, error): stats is None and error is a message if there
    isn't enough data yet; otherwise stats = {"model", "n_matches", "n_total",
    "summary"}.
    """
    queries = _extract_queries(decisions)
    if len(queries) < MIN_DECISIONS:
        return None, "Not enough decisions yet."

    if override and _is_valid_twostage_model(override):
        model = override
    else:
        try:
            learned, polarity = _fit_curves(queries)
            model = {
                "type": "twostage",
                "aggregation": "weighted",
                "features": {k: _curve_to_params(k, learned[k], polarity[k])
                             for k in features.FEATURE_KEYS},
            }
            evidence = _sign_evidence(queries)
            for k in features.FEATURE_KEYS:
                if _direction_prior(k) is None and abs(evidence[k]) < SIGN_EVIDENCE_MIN:
                    model["features"][k]["matters"] = 0
        except Exception:
            model = {"type": "twostage", **_default_model()}

    n_matches = sum(1 for a, b, y in queries if predict(model, a, b) == y)
    stats = {
        "model": model,
        "n_matches": n_matches,
        "n_total": len(queries),
        "summary": summary_text(model),
    }
    return stats, None