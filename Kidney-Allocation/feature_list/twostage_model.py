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
AMP = [0.0, 1.0, 2.0, 3.2]
GAIN = 0.9          # sharpness of the "weighted"/"prominent" verdict sigmoid
TALLY_GAIN = 0.95   # sharpness of the "tally" verdict sigmoid

MIN_DECISIONS = 6


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

def _fit_curves(queries, lr=0.15, lam=0.02, iters=700, gain=GAIN):
    """
    Jointly fits one monotonic step function per feature against the same set
    of pairwise comparisons (the prediction for a query sums every feature's
    contribution, so the gradient for each feature's curve is shaped by how
    well the *whole* model currently explains the data — not fit in isolation).

    Returns (learned, polarity):
      learned[key] -> list of length (max-min+1), the sign-adjusted cumulative
                      score at each value from min to max.
      polarity[key] -> +1 or -1, which direction the feature initially leans.
    """
    keys = features.FEATURE_KEYS
    ranges = features.ranges()
    spans = {k: max(0, int(ranges[k][1] - ranges[k][0])) for k in keys}

    raw = {k: np.zeros(spans[k]) for k in keys}
    polarity = {}
    for k in keys:
        c = sum((a[k] - b[k]) * (y - 0.5) for a, b, y in queries)
        polarity[k] = 1.0 if c >= 0 else -1.0

    N = max(1, len(queries))

    for _ in range(iters):
        cum = {}
        for k in keys:
            C = np.zeros(spans[k] + 1)
            if spans[k] > 0:
                C[1:] = np.cumsum(_softplus(raw[k]))
            cum[k] = C

        def sc(k, v):
            lo = ranges[k][0]
            idx = int(round(v - lo))
            idx = max(0, min(idx, len(cum[k]) - 1))
            return polarity[k] * cum[k][idx]

        grad = {k: np.zeros(spans[k]) for k in keys}
        for a, b, y in queries:
            z = sum(sc(k, a[k]) - sc(k, b[k]) for k in keys)
            err = _sigmoid(gain * z) - y
            for k in keys:
                if spans[k] == 0:
                    continue
                lo = ranges[k][0]
                pa, pb = a[k] - lo, b[k] - lo
                ks = np.arange(spans[k])
                grad[k] += err * gain * polarity[k] * ((pa > ks).astype(float) - (pb > ks).astype(float))

        for k in keys:
            if spans[k] == 0:
                continue
            sp = _softplus(raw[k])
            gk = grad[k] / N + 2 * lam * sp
            gk = gk * _dsp(raw[k])
            raw[k] = raw[k] - lr * gk

    learned = {}
    for k in keys:
        C = np.zeros(spans[k] + 1)
        if spans[k] > 0:
            C[1:] = np.cumsum(_softplus(raw[k]))
        learned[k] = (polarity[k] * C).tolist()

    return learned, polarity


def _curve_to_params(key, curve, polarity):
    """
    Reads the continuous fitted step function off into one of three editable
    shapes plus a 0-3 strength bucket — showing someone the raw curve isn't
    editable, but "first counts most, somewhat strongly" is.
    """
    lo, hi = features.ranges()[key]
    S = np.asarray(curve, dtype=float)
    mx = float(np.max(np.abs(S))) if S.size else 0.0
    rng = float(np.max(S) - np.min(S)) if S.size else 0.0

    if rng < 0.4:
        matters = 0
    elif mx < 1.1:
        matters = 1
    elif mx < 2.1:
        matters = 2
    else:
        matters = 3

    direction = "favor" if polarity >= 0 else "against"
    mode, cutoff = "steady", lo + 1

    if matters > 0 and S.size > 1:
        inc = np.abs(np.diff(S))
        total = float(inc.sum())
        biggest = float(inc.max()) if inc.size else 0.0
        if inc.size > 1 and total > 0 and biggest > 0.6 * total:
            mode = "anyvsnone"
            cutoff = lo + int(np.argmax(inc)) + 1
        else:
            diminishing = True
            for i in range(1, inc.size):
                if inc[i] > inc[i - 1] + 1e-6:
                    diminishing = False
                    break
            mode = "first" if (diminishing and inc.size > 2) else "steady"

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
            k: {"matters": 0, "dir": "favor", "mode": "steady", "cutoff": lo + 1,
                "cond": {"on": False, "when": k, "test": "nonzero", "n": 1}}
            for k, (lo, hi) in features.ranges().items()
        },
    }


# ── Stage 2 (fixed aggregation) + evaluation ──────────────────────────────────

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
    else:
        mag = (v - lo) / (hi - lo) if hi > lo else 0.0
    sign = 1.0 if st["dir"] == "favor" else -1.0
    return AMP[st["matters"]] * mag * sign


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
        order = sorted(keys, key=lambda k: AMP[model["features"][k]["matters"]], reverse=True)
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
        order = sorted(keys, key=lambda k: AMP[model["features"][k]["matters"]], reverse=True)
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


def feature_sentence(model, key):
    st = model["features"][key]
    lo, hi = features.ranges()[key]
    noun = _noun(key)
    if st["matters"] == 0:
        return f"You don't take {noun} into account."
    side = "in a patient's favor" if st["dir"] == "favor" else "against a patient"
    if st["mode"] == "anyvsnone":
        if st["cutoff"] <= lo + 1:
            s = f"Having any {noun} at all counts {side}."
        else:
            s = f"Reaching {st['cutoff']} or more {noun} counts {side}, while less makes no difference."
    elif st["mode"] == "first":
        s = f"More {noun} counts {side}, and the first bit matters the most."
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
            cond_txt = f"has at least {st['cond']['n']} {w_noun}"
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