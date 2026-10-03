"""
Decision Tree route handlers.

All FFT/decision-tree-specific logic extracted from app.py lives here.
app.py dispatches to these functions after handling auth and shared setup.
Functions call db.* and features.* directly to avoid circular imports.
"""

import json
from datetime import datetime

from flask import render_template, session, url_for

import db
import features
from decision_tree import tiebreaker
from decision_tree.fft_model import (
    FastFrugalTree, augment, build_difference_features, feature_row, train_fft,
)
from decision_tree.fft_component import (
    DEFAULT_FFT_PALETTE, PARAM_DIRECTION_LABELS, fft_svg_explained,
)

FEATURES = features.FEATURE_KEYS

# How many new checks a participant may add in one editing pass (enforced here
# and in the browser).
MAX_NEW_FEATURES = 2

_train_cache = {}


def train_fft_cached(decisions, params, override):
    key = (json.dumps(decisions, sort_keys=True), tuple(params), json.dumps(override or {}))
    if key not in _train_cache:
        _train_cache[key] = train_fft(
            json.dumps(decisions), list(params), json.dumps(override) if override else None
        )
    return _train_cache[key]


def clear_cache():
    _train_cache.clear()


def _live_tree(rec):
    """
    Rebuild the participant's current tree (edits included) for server-side
    prediction. Returns (tree, feat_names, chain) or (None, [], []).
    """
    decisions = [d for d in rec.get("part1", []) if d.get("choice") in ("A", "B")]
    if len(decisions) < 6:
        return None, [], []

    username = session["username"]
    override = db.get_fft_override(username)
    fft, nodes_df, stats, feat_names, error = train_fft_cached(decisions, FEATURES, override)
    if error or stats is None:
        return None, [], []

    tree = FastFrugalTree.from_dict(stats["tree"], feature_names=feat_names)

    chain = []
    if tiebreaker.enabled():
        F, _ = build_difference_features(decisions, FEATURES)
        y = [1 if d["choice"] == "A" else 0 for d in decisions]
        F_aug, y_aug = augment(F, y)
        chain = tiebreaker.fit_chain(F_aug, y_aug, feat_names)

    return tree, feat_names, chain


def _predict_pair(tree, feat_names, chain, scenario):
    """Predict one A/B pair and describe how the model got there."""
    a = {p: float(scenario["A"][p]) for p in FEATURES}
    b = {p: float(scenario["B"][p]) for p in FEATURES}
    row = feature_row(FEATURES, a, b, feat_names).values[0]

    out = tiebreaker.predict_with_tiebreak(tree, row, chain)
    out["choice"] = "A" if out["cls"] == 1 else "B"
    out["reason"] = _exit_reason(out)
    return out


def _exit_reason(result):
    idx = result.get("exit_index", -1)
    if result.get("tiebreak_used"):
        decisive = [t for t in result.get("tiebreak_trace", []) if t.get("decisive")]
        if decisive:
            name = decisive[-1]["feature"].replace("_diff", "").replace("_", " ")
            return f"Too close to call at step {idx + 1}, so the tie-breaker used {name}."
    if idx is None or idx < 0:
        return "No step applied, so the model used its default choice."
    if result.get("refine_branch") is not None:
        return f"Decided by the follow-up check on step {idx + 1}."
    return f"Decided at step {idx + 1}."


def _scenarios_for_audit(decisions):
    return [
        {
            "A": {p: d.get(f"A_{p}") for p in FEATURES},
            "B": {p: d.get(f"B_{p}") for p in FEATURES},
            "choice": d.get("choice"),
        }
        for d in decisions
    ]


def _feature_ranges():
    out = {}
    for p in FEATURES:
        lo, hi = features.ranges()[p]
        out[p] = {"min": lo, "max": hi, "step": 1}
    return out


# ── View handlers (called from app.py route functions) ────────────────────────

def results_view():
    username = session["username"]
    users = db.load_all_users()
    rec = users.get(username, {})
    override = db.get_fft_override(username)

    decisions = [d for d in rec.get("part1", []) if d.get("choice") in ("A", "B")]
    if len(decisions) < 6:
        return render_template("not_ready.html", n_decisions=len(decisions))

    fft_obj, nodes_df, stats, feat_names, error = train_fft_cached(decisions, FEATURES, override)
    if error or stats is None:
        return render_template("not_ready.html", n_decisions=len(decisions))

    version = db.get_model_version(username)
    views = db.get_tutorial_seen_count(username)
    tutorial_mode = "full" if views == 0 else ("reminder" if views == 2 else "")
    db.bump_tutorial_seen_count(username)

    return render_template(
        "model.html",
        apply_url=url_for("apply_edit"),
        reset_url=url_for("reset_edit"),
        rate_url=url_for("rate"),
        continue_url=url_for("part2_intro"),
        review_url=url_for("review"),
        new_scenario_url=url_for("new_scenario"),
        answer_scenario_url=url_for("answer_scenario"),
        feedback_url=url_for("scenario_feedback"),
        log_url=url_for("log_event"),
        tutorial_seen_url=url_for("tutorial_seen"),
        current_score=rec.get("alignment_score"),
        has_override=bool(override),
        tree_json=stats["tree"],
        params_json=[f"{p}_diff" for p in FEATURES],
        labels_json=PARAM_DIRECTION_LABELS,
        node_explanations_json=stats.get("node_explanations") or [],
        summary_explanation_json=stats.get("summary_explanation") or "",
        past_scenarios_json=_scenarios_for_audit(decisions),
        feature_ranges_json=_feature_ranges(),
        features_json=features.FEATURES_DATA,
        model_version=version,
        trial_count=len(db.list_trials(username, model_version=version)),
        tutorial_mode=tutorial_mode,
        max_new_features=MAX_NEW_FEATURES,
    )


def final_view():
    username = session["username"]
    users = db.load_all_users()
    rec = users.get(username, {})
    override = db.get_fft_override(username)

    all_decisions = (
        [d for d in rec.get("part1", []) if d.get("choice") in ("A", "B")]
        + [d for d in rec.get("part2", []) if d.get("choice") in ("A", "B")]
    )
    if len(all_decisions) < 6:
        return render_template("not_ready.html", n_decisions=len(all_decisions))

    fft_obj, nodes_df, stats, feat_names, error = train_fft_cached(all_decisions, FEATURES, override)
    if error or stats is None:
        return render_template("not_ready.html", n_decisions=len(all_decisions))

    svg = fft_svg_explained(
        tree=stats["tree"],
        palette=DEFAULT_FFT_PALETTE,
        node_explanations=stats.get("node_explanations"),
        summary_explanation=stats.get("summary_explanation"),
    )
    return render_template("final.html", svg=svg)


def apply_edit_handler(username, payload):
    """Apply a tree edit. Returns (response_dict, http_status_code)."""
    tree = payload.get("tree")
    if not tree:
        return {"error": "missing tree"}, 400

    baseline = payload.get("baseline_node_count")
    nodes = tree.get("nodes") or []
    if isinstance(baseline, int) and len(nodes) - baseline > MAX_NEW_FEATURES:
        return {
            "error": f"You can add at most {MAX_NEW_FEATURES} new checks at a time.",
            "limit": MAX_NEW_FEATURES,
        }, 400

    db.set_fft_override(username, tree)
    clear_cache()
    version = db.bump_model_version(username)
    db.log_event(username, "model_edit_applied",
                 {"nodes": len(nodes), "model_version": version})
    return {"ok": True, "redirect": url_for("results")}, 200


def review_view():
    """Render the agreements/disagreements review page (decision-tree only)."""
    username = session["username"]
    current_version = db.get_model_version(username)
    versions = db.list_trial_versions(username)
    if current_version not in versions:
        versions = [current_version] + versions

    groups = []
    for v in versions:
        trials = db.list_trials(username, model_version=v)
        groups.append({
            "version":       v,
            "is_current":    v == current_version,
            "agreements":    [t for t in trials if t["agreed"]],
            "disagreements": [t for t in trials if t["agreed"] is False],
            "total":         len(trials),
        })

    return render_template(
        "review.html",
        groups=groups,
        features=FEATURES,
        feature_labels={k: v["label"] for k, v in features.FEATURE_BY_KEY.items()},
        back_url=url_for("results"),
        new_scenario_url=url_for("new_scenario"),
        current_version=current_version,
    )


def answer_scenario_handler(payload, username, rec):
    """
    Run the decision tree against the user's scenario choice.
    Returns (response_dict, http_status_code).
    """
    trial_id = payload.get("trial_id")
    choice = payload.get("choice")
    if choice not in ("A", "B") or trial_id is None:
        return {"error": "invalid answer"}, 400

    trial = db.get_trial(trial_id, username)
    if not trial:
        return {"error": "unknown scenario"}, 404

    tree, feat_names, chain = _live_tree(rec)
    if tree is None:
        return {"error": "model not ready"}, 400

    scenario = {"A": trial["A"], "B": trial["B"]}
    result = _predict_pair(tree, feat_names, chain, scenario)

    db.record_trial_outcome(
        trial_id, username, choice, result["choice"],
        exit_index=result.get("exit_index"), exit_reason=result.get("reason"),
    )
    db.log_event(username, "scenario_answered", {
        "trial_id": trial_id, "choice": choice,
        "model_choice": result["choice"], "agreed": choice == result["choice"],
    })

    return {
        "ok": True,
        "trial_id": trial_id,
        "user_choice": choice,
        "model_choice": result["choice"],
        "agreed": choice == result["choice"],
        "reason": result["reason"],
        "ambiguous": result.get("ambiguous", False),
        "tiebreak_used": result.get("tiebreak_used", False),
    }, 200
