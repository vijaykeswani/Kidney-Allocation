"""
Loads features.json once and exposes it in the shapes the rest of the app
needs. To swap in a different set of study features, edit features.json —
nothing in this file (or anywhere else) needs to change as long as each
feature entry has the same fields.

FEATURE_KEYS     ["dependents", "age", ...]                 — study order
FEATURE_BY_KEY   {"age": {...the full JSON entry...}, ...}
"""

import json
import os

_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "features.json")


def _load():
    with open(_PATH) as f:
        data = json.load(f)
    feats = data["features"]
    required = ("key", "label", "description", "min", "max",
                "gap_phrase", "direction_higher", "direction_lower",
                "direction_higher_short", "direction_lower_short")
    for f_ in feats:
        missing = [k for k in required if k not in f_]
        if missing:
            raise ValueError(f"features.json entry {f_.get('key', '?')!r} is missing: {missing}")
    return feats


FEATURES_DATA = _load()
FEATURE_KEYS = [f["key"] for f in FEATURES_DATA]
FEATURE_BY_KEY = {f["key"]: f for f in FEATURES_DATA}


def descriptions():
    """key -> description, for the start page."""
    return {k: v["description"] for k, v in FEATURE_BY_KEY.items()}


def ranges():
    """key -> (min, max), for scenario generation and sliders."""
    return {k: (v["min"], v["max"]) for k, v in FEATURE_BY_KEY.items()}


def direction_labels(short=False):
    """key -> (label when higher, label when lower)."""
    suffix = "_short" if short else ""
    return {k: (v[f"direction_higher{suffix}"], v[f"direction_lower{suffix}"])
            for k, v in FEATURE_BY_KEY.items()}


def gap_phrases():
    """key -> (gap phrase, unit) — e.g. ("age gap", "years"), used in captions."""
    return {k: (v["gap_phrase"], v.get("unit", "")) for k, v in FEATURE_BY_KEY.items()}


def values_map():
    """key -> list of allowed discrete values, for features with a fixed value set.
    Features without a 'values' entry are continuous and are not included."""
    return {k: v["values"] for k, v in FEATURE_BY_KEY.items() if "values" in v}

def value_steps():
    """key -> natural increment between allowed values (gcd of the gaps), e.g.
    life_years_gained {5,10,...} -> 5, years_waiting {1,3,5,...} -> 2. Features with no
    fixed value set step by 1. Used to snap slider steps and fitted thresholds to values
    that can actually occur."""
    from math import gcd
    out = {}
    for k, v in FEATURE_BY_KEY.items():
        vals = sorted(v.get("values") or [])
        g = 0
        for a, b in zip(vals, vals[1:]):
            g = gcd(g, int(b - a))
        out[k] = g if g > 0 else 1
    return out

def ordered_data(order):
    """FEATURES_DATA entries in a participant's presentation order (any key not in `order` goes last)."""
    keys = [k for k in (order or []) if k in FEATURE_BY_KEY]
    keys += [k for k in FEATURE_KEYS if k not in keys]
    return [FEATURE_BY_KEY[k] for k in keys]


def value_suffixes():
    """key -> text shown after a value (e.g. "%" for a chance), only for features that have one."""
    return {k: v["value_suffix"] for k, v in FEATURE_BY_KEY.items() if v.get("value_suffix")}


def fmt_value(key, v):
    """A value as participants see it: whole numbers, plus the feature's suffix if it has one."""
    v = float(v)
    return (str(int(v)) if v == int(v) else str(v)) + FEATURE_BY_KEY[key].get("value_suffix", "")
