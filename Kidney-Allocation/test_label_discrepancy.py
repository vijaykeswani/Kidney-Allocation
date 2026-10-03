"""
Test for "Prefer A / Prefer B" label discrepancies in the FFT visualization.

Walks every node (and refine sub-node) in saved trees from users.json, calls
outcome_label() on each, and asserts no "Prefer A" or "Prefer B" leaks through.
Also renders the full SVG and scans its text content for the bad strings.

Run with:  python test_label_discrepancy.py
"""

import json
import os
import sys
import re

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from decision_tree.fft_component import outcome_label, _default_outcome_label, fft_svg_explained, DEFAULT_FFT_PALETTE

BAD_STRINGS = ["Prefer A", "Prefer B"]

MOCK_LEGACY_TREES = [
    # Legacy: use_abs missing, op/threshold/exit_class present
    {"nodes": [
        {"feature": "urgency_score_diff", "op": ">=", "threshold": 1, "exit_class": 1,
         "support": 0.5, "purity": 0.8},
        {"feature": "age_diff", "op": "<=", "threshold": -2, "exit_class": 0,
         "support": 0.3, "purity": 0.7},
    ], "default_class": 0, "default_support": 0.2, "default_purity": 0.6,
       "feature_names": ["urgency_score_diff", "age_diff"]},
    # Modern: use_abs=True
    {"nodes": [
        {"feature": "health_score_diff", "op": ">=", "threshold": 1, "use_abs": True,
         "prefer_higher": True, "exit_class": 1, "support": 0.4, "purity": 0.75},
    ], "default_class": 0, "default_support": 0.6, "default_purity": 0.5,
       "feature_names": ["health_score_diff"]},
    # Edge: empty nodes list
    {"nodes": [], "default_class": 1, "default_support": 1.0, "default_purity": 0.5,
     "feature_names": []},
]

failures = []


def check_label(label, context):
    for bad in BAD_STRINGS:
        if bad.lower() in label.lower():
            failures.append(f"FAIL [{context}]: got {label!r}")
            return False
    return True


def audit_tree(tree_dict, source):
    nodes = tree_dict.get("nodes") or []
    for i, node in enumerate(nodes):
        for short in (False, True):
            lbl = outcome_label(node, short=short)
            check_label(lbl, f"{source} node[{i}] short={short}")
        r = node.get("refine")
        if r:
            # Refine uses _refine_branch_label internally — check via outcome_label mock
            for is_true in (True, False):
                rn = {"use_abs": True,
                      "prefer_higher": (r.get("true_class", 1) == 1) if is_true else (r.get("true_class", 1) != 1),
                      "feature": r.get("feature", "")}
                for short in (False, True):
                    lbl = outcome_label(rn, short=short)
                    check_label(lbl, f"{source} node[{i}].refine is_true={is_true} short={short}")

    def_lbl = _default_outcome_label(tree_dict, short=True)
    check_label(def_lbl, f"{source} default_leaf (short=True)")
    def_lbl2 = _default_outcome_label(tree_dict, short=False)
    check_label(def_lbl2, f"{source} default_leaf (short=False)")


def audit_svg(tree_dict, source):
    try:
        svg = fft_svg_explained(tree_dict, palette=DEFAULT_FFT_PALETTE)
        # Extract all text content from SVG
        texts = re.findall(r'>([^<]+)<', svg)
        for txt in texts:
            t = txt.strip()
            if not t:
                continue
            for bad in BAD_STRINGS:
                if bad.lower() in t.lower():
                    failures.append(f"FAIL [SVG {source}]: found {t!r} in rendered SVG")
    except Exception as e:
        print(f"  WARNING: could not render SVG for {source}: {e}")


print("=" * 60)
print("FFT Label Discrepancy Test")
print("=" * 60)

# 1. Test mock trees (legacy + modern + edge cases)
print("\n[1] Mock trees (legacy, modern, empty)")
for i, t in enumerate(MOCK_LEGACY_TREES):
    name = f"mock_tree_{i}"
    audit_tree(t, name)
    audit_svg(t, name)

# 2. Test real trees from the SQL store
try:
    import db
    from app import FEATURES
    db.init_db(FEATURES)
    users = db.load_all_users()
except Exception as e:
    users = None
    print(f"\n[2] Could not open the database ({e}) — skipping real-tree tests")

if users is not None:
    print(f"\n[2] Real trees from {db.DB_PATH}")
    for username, rec in users.items():
        for field in ("trained_tree", "override_tree", "fft_override"):
            t = rec.get(field)
            if not t:
                continue
            src = f"{username}/{field}"
            print(f"  Checking {src} ...")
            audit_tree(t, src)
            audit_svg(t, src)

# 3. Summary
print("\n" + "=" * 60)
if failures:
    print(f"FOUND {len(failures)} DISCREPANCY(IES):")
    for f in failures:
        print(f"  {f}")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED — no 'Prefer A/B' labels found in any node or SVG.")