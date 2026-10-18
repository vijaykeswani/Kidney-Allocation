"""
Plain-language, step-by-step explanation of ONE decision-tree prediction ("why did the model pick
Patient B here?"). Used for the static final page; the live model page has a JavaScript twin,
`explainTreeCase` in templates/model.html. The two must produce identical text — test_explanations.py
checks that — so change both together.
"""
import re

import features


def _round(x):                       # JS Math.round(Math.abs(x)): halves round up
    return int(abs(float(x)) + 0.5)


def _unit_str(n, unit):
    if not unit:
        return ""
    return " " + (re.sub(r"s$", "", unit) if n == 1 else unit)


def _thr_with_unit(thr, unit, op):
    if op == ">=":
        qual = "or more"
    elif op == "<=":
        qual = "or less"
    elif op == ">":
        qual = "more than "
    else:
        qual = "less than "
    if op in (">", "<"):
        return qual + str(thr) + (" " + (re.sub(r"s$", "", unit) if thr == 1 else unit) if unit else "")
    u = (" " + (re.sub(r"s$", "", unit) if thr == 1 else unit)) if unit else ""
    return f"{thr}{u} {qual}"


def _gap(feature):                   # feature key like "life_years_gained"
    f = features.FEATURE_BY_KEY[feature]
    return f["gap_phrase"], f.get("unit", "")


def _prefer_higher(node):
    if node.get("use_abs"):
        return node.get("prefer_higher") is not False
    op, thr, ec = node.get("op") or ">=", float(node.get("threshold") or 0), node.get("exit_class")
    if op == ">=" and thr >= 0:
        return ec == 1
    if op == "<=" and thr <= 0:
        return ec == 0
    return ec == 1


def _outcome_lower(node, base):
    f = features.FEATURE_BY_KEY[base]
    lbl = f["direction_higher"] if _prefer_higher(node) else f["direction_lower"]
    return lbl[:1].lower() + lbl[1:]


def _fires(node, x):
    thr = float(node.get("threshold") or 0)
    if node.get("use_abs"):
        return abs(x) >= thr
    return x >= thr if (node.get("op") or ">=") == ">=" else x <= thr


def explain_tree_case(tree, a, b, user_choice=None, final=False):
    """
    tree: the tree dict (nodes, default_class); a, b: {feature_key: value} for Patient A and B.
    Returns {"winner", "exit_index", "headline", "steps": [{"n", "text", "applies"}], "note"}.
    `final=True` drops the "you can reorder or edit the steps" hint (nothing is editable any more).
    """
    nodes = tree.get("nodes") or []
    steps, exit_index, cls = [], -1, None
    for i, nd in enumerate(nodes):
        base = nd["feature"].replace("_diff", "")
        gap, unit = _gap(base)
        x = float(a[base]) - float(b[base])
        ax = _round(x)
        thr = _round(nd.get("threshold") or 0)
        n = i + 1
        refine = nd.get("refine")
        if not _fires(nd, x):
            if refine:
                text = f"Step {n}: the {gap} is {ax}{_unit_str(ax, unit)}, more than the {thr}{_unit_str(thr, unit)} that counts as a tie, so this step doesn't apply."
            else:
                text = f"Step {n}: the {gap} is {ax}{_unit_str(ax, unit)}; this step needs {_thr_with_unit(thr, unit, nd.get('op') or '>=')}, so it doesn't apply."
            steps.append({"n": n, "text": text, "applies": False})
            continue
        exit_index = i
        if refine:
            rb = refine["feature"].replace("_diff", "")
            rgap, runit = _gap(rb)
            rx = float(a[rb]) - float(b[rb])
            rax = _round(rx)
            rthr = _round(refine.get("threshold") or 0)
            if refine.get("use_abs"):
                rcond = abs(rx) >= float(refine.get("threshold") or 0)
                cls = (1 if ((rx > 0) == refine.get("prefer_higher")) else 0) if rcond else refine.get("false_class")
            else:
                rop = refine.get("op") or ">="
                rcond = rx >= float(refine.get("threshold") or 0) if rop == ">=" else rx <= float(refine.get("threshold") or 0)
                cls = refine.get("true_class") if rcond else refine.get("false_class")
            W = "A" if cls == 1 else "B"
            steps.append({"n": n, "applies": True, "text":
                f"Step {n}: the {gap} is {ax}{_unit_str(ax, unit)}, within the {thr}{_unit_str(thr, unit)} that counts as a tie on {base.replace('_', ' ')}, so a follow-up check decides."})
            rop_txt = refine.get("op") or ">="
            need = _thr_with_unit(rthr, runit, rop_txt)
            if rcond:
                steps.append({"n": n, "applies": True, "text": f"Follow-up: the {rgap} is {rax}{_unit_str(rax, runit)}; it needs {need}, so it applies. Rule: {_outcome_lower(refine, rb)} \u2192 Patient {W}."})
            else:
                steps.append({"n": n, "applies": True, "text": f"Follow-up: the {rgap} is {rax}{_unit_str(rax, runit)}; it needs {need}, so it doesn't apply, and the follow-up falls back to Patient {W}."})
        else:
            cls = 1 if ((x > 0) == nd.get("prefer_higher")) else 0
            W = "A" if cls == 1 else "B"
            steps.append({"n": n, "applies": True, "text":
                f"Step {n}: the {gap} is {ax}{_unit_str(ax, unit)} (Patient A {_round(a[base])}, Patient B {_round(b[base])}); this step needs "
                f"{_thr_with_unit(thr, unit, nd.get('op') or '>=')}, so it applies. Rule: {_outcome_lower(nd, base)} \u2192 Patient {W}."})
        break
    if exit_index == -1:
        cls = tree.get("default_class") or 0
    W = "A" if cls == 1 else "B"
    headline = (f"The model picks Patient {W}. It decided at step {exit_index + 1}." if exit_index >= 0
                else f"The model picks Patient {W}. No step applied, so this is its default choice.")

    note = None
    if user_choice and user_choice != W:
        if exit_index >= 0:
            later = len(nodes) - exit_index - 1
            note = (f"You chose Patient {user_choice}. The tree stops at the first step that applies, so it never looked at the later steps."
                    if later > 0 else f"You chose Patient {user_choice}. The tree stops at the first step that applies.")
            if not final:
                note += " If a different factor should come first, you can reorder or edit the steps."
        else:
            note = f"You chose Patient {user_choice}. No rule applied to these two patients, so the model fell back to its default choice."
            if not final:
                note += " Adding a step for a factor that mattered to you here would change this."
    return {"winner": W, "exit_index": exit_index, "headline": headline, "steps": steps, "note": note}
