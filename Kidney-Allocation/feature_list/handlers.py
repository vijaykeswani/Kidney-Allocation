"""
Feature List route handlers.

All two-stage/feature-list-specific logic extracted from app.py lives here.
app.py dispatches to these functions after handling auth and shared setup.
Functions call db.* and features.* directly to avoid circular imports.
"""

import features
from datetime import datetime

from flask import render_template, session, url_for

import db
import survey_spec as survey_spec_module
from feature_list import twostage_model

FEATURES = features.FEATURE_KEYS


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
        out[p] = {"min": lo, "max": hi, "step": features.value_steps()[p],
                  "values": features.FEATURE_BY_KEY[p].get("values")}
    return out


# ── View handlers (called from app.py route functions) ────────────────────────
#
# v1 ("twostage") shows every feature's card at once in a grid; v2
# ("twostage_v2") shows one feature at a time via a selector + single pane.
# Both share the exact same model, training, and edit-saving logic — only the
# template differs — so results_view()/final_view() take the template name
# as a parameter rather than duplicating the view logic, and the v2 route
# just calls through with a different name.

def results_view(template="twostage.html"):
    username = session["username"]
    users = db.load_all_users()
    rec = users.get(username, {})

    decisions = [d for d in rec.get("part1", []) if d.get("choice") in ("A", "B")]
    override = db.get_fft_override(username)
    stats, err = twostage_model.train_twostage(decisions, override=override)

    if stats is None:
        return render_template("not_ready.html", n_decisions=len(decisions))

    # Compute the un-edited baseline match count so the UI can show "started
    # at X of 20" even after the participant has applied edits.
    baseline_matches = stats["n_matches"]
    if override:
        baseline_stats, _ = twostage_model.train_twostage(decisions, override=None)
        if baseline_stats:
            baseline_matches = baseline_stats["n_matches"]

    db.log_event(username, "results_viewed", {"edited": bool(override), "n_part1": len(decisions)})
    views = db.get_tutorial_seen_count(username)
    tutorial_mode = "full" if views == 0 else ("reminder" if views == 2 else "")
    db.bump_tutorial_seen_count(username)

    return render_template(
        template,
        final_mode=False,
        survey_url="",
        final_edited=False,
        final_arm="treatment",
        log_url=url_for("log_event"),
        apply_url=url_for("apply_edit"),
        reset_url=url_for("reset_edit"),
        rate_url=url_for("rate"),
        continue_url=url_for("part2_intro"),
        tutorial_seen_url=url_for("tutorial_seen"),
        current_score=rec.get("alignment_score"),
        has_override=bool(override),
        model_json=stats["model"],
        n_matches=stats["n_matches"],
        n_total=stats["n_total"],
        baseline_matches=baseline_matches,
        summary_text=stats["summary"],
        past_scenarios_json=_scenarios_for_audit(decisions),
        feature_ranges_json=_feature_ranges(),
        features_json=features.FEATURES_DATA,
        tutorial_mode=tutorial_mode,
    )


def _part(rec, name):
    return [d for d in rec.get(name, []) if d.get("choice") in ("A", "B")]


def freeze_model(username, use_override=True):
    """
    Fix the participant's model as it stands: their edits (treatment arms) or
    the unedited Part-1 model (control). Returns a JSON-able dict, or None if
    there are too few answers to build a model. Never retrains afterwards.
    """
    rec = db.load_all_users().get(username, {})
    decisions = _part(rec, "part1")
    override = db.get_fft_override(username) if use_override else None
    stats, _ = twostage_model.train_twostage(decisions, override=override)
    if stats is None:
        return None
    base, _ = twostage_model.train_twostage(decisions, override=None)
    return {
        "model": stats["model"],
        "edited": bool(stats["model"] != base["model"]),
        "matches_before_edit": base["n_matches"],
        "matches_at_lock": stats["n_matches"],
        "n_part1": stats["n_total"],
        "locked_at": datetime.now().isoformat(),
    }


def _holdout_matches(model, decisions):
    q = twostage_model._extract_queries(decisions)
    return sum(1 for a, b, y in q if twostage_model.predict(model, a, b) == y), len(q)


def final_view(template="twostage.html"):
    """The locked-in model, shown read-only, with how it did on the Part-2 hold-out."""
    username = session["username"]
    rec = db.load_all_users().get(username, {})
    frozen = db.get_frozen_model(username)
    part1, part2 = _part(rec, "part1"), _part(rec, "part2")
    if not frozen or not part2:
        return render_template("not_ready.html", n_decisions=len(part1) + len(part2))

    model = frozen["model"]
    n_matches, n_total = _holdout_matches(model, part2)
    base, _ = twostage_model.train_twostage(part1, override=None)
    base_matches, _ = _holdout_matches(base["model"], part2)

    if "holdout" not in frozen:     # record once, for analysis
        frozen["holdout"] = {"n": n_total, "matches": n_matches, "matches_if_unedited": base_matches}
        db.set_frozen_model(username, frozen)
    db.log_event(username, "final_viewed", frozen["holdout"])

    return render_template(
        template,
        final_mode=True,
        final_edited=bool(frozen.get("edited")),
        final_arm=frozen.get("arm", "treatment"),
        survey_url=url_for("survey"),
        log_url=url_for("log_event"),
        apply_url="", reset_url="", rate_url="", continue_url="", tutorial_seen_url="",
        current_score=None, has_override=False,
        model_json=model,
        n_matches=n_matches,
        n_total=n_total,
        baseline_matches=base_matches,
        summary_text=twostage_model.summary_text(model),
        past_scenarios_json=_scenarios_for_audit(part2),
        feature_ranges_json=_feature_ranges(),
        features_json=features.FEATURES_DATA,
        tutorial_mode="",
    )


def results_view_v2():
    return results_view(template="twostage_v2.html")


def final_view_v2():
    return final_view(template="twostage_v2.html")


def apply_edit_handler(username, payload):
    """Apply a feature-list model edit (shared by v1 and v2 — same model
    schema, same storage). Returns (response_dict, http_status_code)."""
    model = payload.get("model")
    if not model:
        return {"error": "missing model"}, 400
    db.set_fft_override(username, model)
    version = db.bump_model_version(username)
    db.log_event(username, "model_edit_applied", {"model_version": version, "type": "twostage"})
    return {"ok": True, "redirect": url_for("results")}, 200