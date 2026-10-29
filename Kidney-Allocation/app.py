"""
Preference Elicitation Portal — Flask edition
SURA 2026 · IIT Delhi

Run:  gunicorn app:app --bind 0.0.0.0:5001 --workers 1 --timeout 120

Structure
---------
This file owns the *shared* base pipeline:
  /              → consent page (Image 2)
  /intro         → landing page with model picker (Image 1)
  /start         → create session, generate scenarios
  /questionnaire → A/B scenario loop (Parts 1 and 2)

Everything specific to a model section lives in its own folder:
  decision_tree/   Fast-and-Frugal Tree (Decision Tree)
  feature_list/    Two-stage feature list (Feature List)

After the model picker, routes dispatch to the appropriate section's handlers.
"""

import os
import random
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, session, url_for

import db
import features
import survey_spec
import feature_order
import decision_tree
import feature_list
import decision_tree.handlers as dt_handlers
import feature_list.handlers as fl_handlers

APP_DIR       = os.path.dirname(os.path.abspath(__file__))
RESPONSES_DIR = os.environ.get("RESPONSES_DIR") or os.path.join(APP_DIR, "responses")

FEATURES = features.FEATURE_KEYS
PARAM_DESCRIPTIONS = features.descriptions()
FEATURE_LABELS = {k: v["label"] for k, v in features.FEATURE_BY_KEY.items()}

PART1_N = 20
PART2_N = 10
# Control sees no model mid-study, but should still spend about the same
# time as the treatment arms (20 + 10 scenarios) — so its "extra" batch is
# the same size as Part 2, not an additional chunk on top.
CONTROL_EXTRA_N = PART2_N

MODEL_TYPES = ("fft", "twostage", "twostage_v2", "control")
DISPLAYABLE_MODEL_TYPES = ("fft", "twostage", "twostage_v2")  # real models control can be shown

db.init_db(FEATURES)

app = Flask(__name__)
app.secret_key = os.environ["FLASK_SECRET_KEY"]
app.register_blueprint(decision_tree.bp)
app.register_blueprint(feature_list.bp)


# ── Scenario generation (fully synthetic — no CSV dependency) ────────────────

FEATURE_RANGES = features.ranges()
FEATURE_VALUES = features.values_map()


def generate_scenarios(n, seed=None):
    """n random A/B patient pairs, sampled uniformly within FEATURE_RANGES."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        a, b = {}, {}
        for p in FEATURES:
            if p in FEATURE_VALUES:
                a[p] = float(rng.choice(FEATURE_VALUES[p]))
                b[p] = float(rng.choice(FEATURE_VALUES[p]))
            else:
                lo, hi = FEATURE_RANGES[p]
                a[p] = float(rng.randint(lo, hi))
                b[p] = float(rng.randint(lo, hi))
        out.append({"A": a, "B": b})
    return out


def _pair_key(sc):
    return (
        tuple(round(float(sc["A"][p]), 3) for p in FEATURES)
        + tuple(round(float(sc["B"][p]), 3) for p in FEATURES)
    )


def _pair_distance(sc, other):
    return sum(
        abs(float(sc[side][p]) - float(other[side][p]))
        for side in ("A", "B") for p in FEATURES
    )


def generate_distinct_scenario(seen_keys, recent=(), tries=60, min_distance=8.0):
    """
    Produce a scenario the participant has not been shown before.
    Exact repeats rejected via seen_keys; near-repeats of recent pairs
    rejected via minimum total distance. Always returns something.
    """
    best, best_score = None, -1.0
    for _ in range(max(1, tries)):
        sc = generate_scenarios(1)[0]
        if _pair_key(sc) in seen_keys:
            continue
        if not recent:
            return sc
        score = min(_pair_distance(sc, r) for r in recent)
        if score >= min_distance:
            return sc
        if score > best_score:
            best, best_score = sc, score
    return best or generate_scenarios(1)[0]


def shuffle_scenarios(scenarios):
    """Randomise presentation order with a fresh entropy-seeded RNG."""
    order = list(scenarios)
    random.SystemRandom().shuffle(order)
    return order


# ── User helpers ──────────────────────────────────────────────────────────────

def get_user_record():
    username = session.get("username")
    if not username:
        return None, None
    users = db.load_all_users()
    return users, users.setdefault(username, {})


def record_decision(field, choice, idx, scenarios, csv_suffix):
    username = session["username"]
    os.makedirs(RESPONSES_DIR, exist_ok=True)
    sc = scenarios[idx]
    row = {
        "username": username,
        "scenario": idx + 1,
        "choice": choice,
        "timestamp": datetime.now().isoformat(),
    }
    for p in FEATURES:
        row[f"A_{p}"] = sc["A"][p]
        row[f"B_{p}"] = sc["B"][p]

    user_file = os.path.join(RESPONSES_DIR, f"{username}_{csv_suffix}.csv")
    new_df = pd.DataFrame([row])
    if os.path.exists(user_file):
        combined = pd.concat([pd.read_csv(user_file), new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(user_file, index=False)

    users, rec = get_user_record()
    rec.setdefault(field, []).append(row)
    rec[f"{field}_index"] = idx + 1
    db.save_user_record(username, rec)
    return rec


# ── Shared route helpers ──────────────────────────────────────────────────────

def _model_type():
    """Current user's model type — session first, DB fallback after restart."""
    if "model_type" in session:
        return session["model_type"]
    if "username" in session:
        mt = db.get_model_type(session["username"])
        session["model_type"] = mt
        return mt
    return "fft"


def _lock_model():
    """
    Freeze the participant's model (idempotent). Treatment arms freeze whatever
    they ended up with (their edits, if any); control freezes the unedited
    Part-1 model of its assigned display type. After this the model is never
    retrained or edited: Part 2 is a genuine hold-out test of THIS model.
    """
    username = session["username"]
    if db.get_frozen_model(username):
        return db.get_frozen_model(username)
    kind = _display_model_type()
    is_control = _model_type() == "control"
    handler = fl_handlers if kind in ("twostage", "twostage_v2") else dt_handlers
    frozen = handler.freeze_model(username, use_override=not is_control)
    if frozen is None:          # not enough Part-1 answers to build a model
        return None
    frozen["kind"] = kind
    frozen["arm"] = "control" if is_control else "treatment"
    frozen["edit_count"] = db.count_events(username, "model_edit_applied")
    db.set_frozen_model(username, frozen)
    db.log_event(username, "model_locked", {k: frozen[k] for k in
                 ("kind", "arm", "edited", "edit_count", "matches_before_edit", "matches_at_lock", "n_part1")})
    return frozen


def _model_locked():
    return bool(db.get_frozen_model(session["username"]))


def _feature_order():
    """This participant's factor order (same on every scenario and page); assigned if they somehow don't have one yet."""
    if "feature_order" not in session:
        session["feature_order"] = feature_order.for_user(session["username"])
    return session["feature_order"]


def _require_user():
    if "username" not in session:
        return redirect(url_for("intro"))
    return None


def _display_model_type():
    """Which model type to actually render at /final — the participant's own
    model_type for treatment arms, or control's randomly-assigned one."""
    mt = _model_type()
    if mt != "control":
        return mt
    if "display_model_type" in session:
        return session["display_model_type"]
    dmt = db.get_display_model_type(session["username"])
    session["display_model_type"] = dmt
    return dmt


# ── Shared base-pipeline routes ───────────────────────────────────────────────

def _requested_model():
    model_type = request.args.get("model")
    return model_type if model_type in MODEL_TYPES else None


@app.route("/")
def start():
    return render_template("consent.html", preselected_model=_requested_model())


@app.route("/consent", methods=["POST"])
def give_consent():
    age_ok = request.form.get("age_confirmed")
    consent_ok = request.form.get("consent_confirmed")
    if age_ok and consent_ok:
        session.clear()
    return redirect(url_for("intro", model=_requested_model()))


@app.route("/intro")
def intro():
    return render_template(
        "start.html",
        features=FEATURES,
        descriptions=PARAM_DESCRIPTIONS,
        feature_labels=FEATURE_LABELS,
        preselected_model=_requested_model(),
    )


@app.route("/start", methods=["POST"])
def do_start():
    name = (request.form.get("username") or "").strip()
    model_type = request.form.get("model_type")
    if model_type not in MODEL_TYPES:
        model_type = "fft"
    if len(name) < 2:
        return render_template(
            "start.html", features=FEATURES, descriptions=PARAM_DESCRIPTIONS,
            feature_labels=FEATURE_LABELS, preselected_model=model_type,
            error="Enter at least 2 characters to continue.",
        )
    session["username"] = name
    session["model_type"] = model_type
    db.set_model_type(name, model_type)
    # One counterbalanced factor order for this participant, used everywhere they see the factors.
    session["feature_order"] = feature_order.for_user(name)

    if model_type == "control":
        # Control never interacts with a model mid-study, but under the
        # symmetric end-of-study design it still sees one, read-only, at the
        # very end — assigned at random now so it's fixed for the session.
        display_type = random.choice(DISPLAYABLE_MODEL_TYPES)
        db.set_display_model_type(name, display_type)
        session["display_model_type"] = display_type

    users, rec = get_user_record()
    if "part1_scenarios" not in rec:
        rec["part1_scenarios"] = shuffle_scenarios(generate_scenarios(PART1_N))
        rec["part1"] = []
        rec["part1_index"] = 0
    if "part2_scenarios" not in rec:
        # Control arm never sees a model, so instead of the short "does the
        # model still match you" check-in, it gets extra scenarios here.
        n2 = CONTROL_EXTRA_N if model_type == "control" else PART2_N
        rec["part2_scenarios"] = shuffle_scenarios(generate_scenarios(n2))
        rec["part2"] = []
        rec["part2_index"] = 0
    db.save_user_record(name, rec)

    return redirect(url_for("questionnaire", part=1))


@app.route("/questionnaire/<int:part>")
def questionnaire(part):
    guard = _require_user()
    if guard:
        return guard
    users, rec = get_user_record()
    scenarios = rec.get(f"part{part}_scenarios", [])
    idx = rec.get(f"part{part}_index", 0)
    n_total = len(scenarios)

    is_control = _model_type() == "control"
    if part == 2:
        _lock_model()           # Part 2 tests a model that is already fixed

    if idx >= n_total:
        if part == 1:
            return redirect(url_for("control_transition") if is_control else url_for("results"))
        # Both arms reach /final now — control just hasn't edited anything,
        # so it always shows the freshly-trained, unedited model (see
        # DISPLAYABLE_MODEL_TYPES / _display_model_type()).
        return redirect(url_for("final"))

    heading = "Which patient should receive the kidney?"
    if part == 1:
        subheading = None
    elif is_control:
        subheading = "Here are some more pairs in the same format — just keep choosing as you have been."
    else:
        subheading = (
            "Your model is now locked in. These last pairs test how well it predicts "
            "your choices on cases it has never seen — just choose as you naturally would."
        )
    return render_template(
        "questionnaire.html",
        part=part, idx=idx, n_total=n_total,
        sc=scenarios[idx], features=_feature_order(),
        feature_labels=FEATURE_LABELS, descriptions=PARAM_DESCRIPTIONS, value_suffix=features.value_suffixes(),
        heading=heading, subheading=subheading,
    )


@app.route("/questionnaire/<int:part>/answer", methods=["POST"])
def answer(part):
    guard = _require_user()
    if guard:
        return guard
    users, rec = get_user_record()
    idx = rec.get(f"part{part}_index", 0)
    scenarios = rec.get(f"part{part}_scenarios", [])
    choice = request.form.get("choice")
    if choice in ("A", "B") and idx < len(scenarios):
        record_decision(f"part{part}", choice, idx, scenarios,
                        csv_suffix="responses" if part == 1 else "survey_responses")
    return redirect(url_for("questionnaire", part=part))


@app.route("/part2")
def part2_intro():
    guard = _require_user()
    if guard:
        return guard
    _lock_model()
    return redirect(url_for("questionnaire", part=2))


@app.route("/control/transition")
def control_transition():
    """
    Control's stand-in for the treatment arms' /results — no model is shown,
    just an acknowledgement, so the two experiences feel structurally similar
    even though nothing is displayed here.
    """
    guard = _require_user()
    if guard:
        return guard
    _lock_model()
    return render_template("control_transition.html")



@app.route("/survey", methods=["GET", "POST"])
def survey():
    """End-of-study questionnaire — same questions for every arm, except the
    'which experiences impacted you' block, which only lists what that arm
    actually had the chance to do."""
    guard = _require_user()
    if guard:
        return guard
    arm = "control" if _model_type() == "control" else "treatment"
    spec = survey_spec.build_survey(arm)
    if request.method == "POST":
        answers = survey_spec.collect_answers(spec, request.form)
        answers["_arm"] = arm
        answers["_model_type"] = _model_type()
        answers["_display_model_type"] = _display_model_type()
        frozen = db.get_frozen_model(session["username"]) or {}
        answers["_model_edited"] = frozen.get("edited")
        db.set_survey_answers(session["username"], answers)
        db.log_event(session["username"], "survey_submitted", {"arm": arm})
        return redirect(url_for("done"))
    return render_template("survey.html", spec=spec, existing=db.get_survey_answers(session["username"]) or {})

@app.route("/done")
def done():
    """Terminal page for the control arm — no model to show, just a thank-you."""
    guard = _require_user()
    if guard:
        return guard
    return render_template("done.html")


# ── Model-section dispatch routes ─────────────────────────────────────────────
# Each route inspects the session model type and delegates to the appropriate
# section handler. No section-specific logic lives in this file.

@app.route("/results")
def results():
    guard = _require_user()
    if guard:
        return guard
    mt = _model_type()
    if mt == "control":
        return redirect(url_for("control_transition"))
    if _model_locked():
        return redirect(url_for("questionnaire", part=2))
    if mt == "twostage_v2":
        return fl_handlers.results_view_v2()
    if mt == "twostage":
        return fl_handlers.results_view()
    return dt_handlers.results_view()


@app.route("/final")
def final():
    guard = _require_user()
    if guard:
        return guard
    # Everyone ends on the same kind of page: the model they were locked into
    # (edited for treatment arms, unedited for control) and how it did on the
    # Part-2 hold-out. Nothing is retrained here.
    _lock_model()
    dmt = _display_model_type()
    if dmt == "twostage_v2":
        return fl_handlers.final_view_v2()
    if dmt == "twostage":
        return fl_handlers.final_view()
    return dt_handlers.final_view()


@app.route("/results/rate", methods=["POST"])
def rate():
    guard = _require_user()
    if guard:
        return jsonify({"error": "not logged in"}), 401
    payload = request.get_json(force=True, silent=True) or {}
    score = payload.get("score")
    if not isinstance(score, int) or not (1 <= score <= 7):
        return jsonify({"error": "invalid score"}), 400
    users, rec = get_user_record()
    rec["alignment_score"] = score
    db.save_user_record(session["username"], rec)
    return jsonify({"ok": True})


@app.route("/edit/apply", methods=["POST"])
def apply_edit():
    guard = _require_user()
    if guard:
        return jsonify({"error": "not logged in"}), 401
    if _model_locked():
        return jsonify({"error": "Your model is locked in and can't be changed."}), 409
    payload = request.get_json(force=True, silent=True) or {}
    username = session["username"]
    if _model_type() in ("twostage", "twostage_v2"):
        data, code = fl_handlers.apply_edit_handler(username, payload)
    else:
        data, code = dt_handlers.apply_edit_handler(username, payload)
    return jsonify(data), code


@app.route("/edit/reset", methods=["POST"])
def reset_edit():
    guard = _require_user()
    if guard:
        return jsonify({"error": "not logged in"}), 401
    if _model_locked():
        return jsonify({"error": "Your model is locked in and can't be changed."}), 409
    username = session["username"]
    db.delete_fft_override(username)
    dt_handlers.clear_cache()
    version = db.bump_model_version(username)
    db.log_event(username, "model_reset", {"model_version": version})
    return jsonify({"ok": True, "redirect": url_for("results")})


@app.route("/scenario/new", methods=["POST"])
def new_scenario():
    """Generate a fresh scenario and open a trial row for it."""
    guard = _require_user()
    if guard:
        return jsonify({"error": "not logged in"}), 401

    username = session["username"]
    users, rec = get_user_record()

    seen = db.seen_pairs(username)
    for part in (1, 2):
        for sc in rec.get(f"part{part}_scenarios", []):
            seen.add(_pair_key(sc))

    previous = db.list_trials(username, answered_only=False)
    recent = [{"A": t["A"], "B": t["B"]} for t in previous[-3:]]

    scenario = generate_distinct_scenario(seen, recent=recent)
    version = db.get_model_version(username)
    trial_id = db.create_trial(username, version, scenario, datetime.now().isoformat())
    db.log_event(username, "scenario_generated",
                 {"trial_id": trial_id, "model_version": version})

    return jsonify({
        "ok": True,
        "trial_id": trial_id,
        "scenario": scenario,
        "model_version": version,
        "index": len(db.list_trials(username, model_version=version)) + 1,
    })


@app.route("/scenario/answer", methods=["POST"])
def answer_scenario():
    """Record the participant's pick, run the model, compare, return result."""
    guard = _require_user()
    if guard:
        return jsonify({"error": "not logged in"}), 401

    payload = request.get_json(force=True, silent=True) or {}
    username = session["username"]
    users, rec = get_user_record()

    data, code = dt_handlers.answer_scenario_handler(payload, username, rec)
    return jsonify(data), code


@app.route("/scenario/feedback", methods=["POST"])
def scenario_feedback():
    """Store participant's reasoning when they disagreed with the model."""
    guard = _require_user()
    if guard:
        return jsonify({"error": "not logged in"}), 401

    payload = request.get_json(force=True, silent=True) or {}
    trial_id = payload.get("trial_id")
    reasoning = (payload.get("reasoning") or "").strip()
    if trial_id is None:
        return jsonify({"error": "missing scenario"}), 400

    username = session["username"]
    if not db.get_trial(trial_id, username):
        return jsonify({"error": "unknown scenario"}), 404

    db.record_trial_reasoning(trial_id, username, reasoning[:4000])
    db.log_event(username, "reasoning_recorded",
                 {"trial_id": trial_id, "length": len(reasoning)})
    return jsonify({"ok": True})


@app.route("/review")
def review():
    guard = _require_user()
    if guard:
        return guard
    return dt_handlers.review_view()


@app.route("/log", methods=["POST"])
def log_event():
    if "username" not in session:
        return jsonify({"ok": False}), 401
    payload = request.get_json(force=True, silent=True) or {}
    name = payload.get("event")
    if name:
        db.log_event(session["username"], name, payload.get("detail"))
    return jsonify({"ok": True})


@app.route("/tutorial/seen", methods=["POST"])
def tutorial_seen():
    if "username" not in session:
        return jsonify({"ok": False}), 401
    payload = request.get_json(force=True, silent=True) or {}
    db.log_event(session["username"], "tutorial_dismissed",
                 {"mode": payload.get("mode"), "via": payload.get("via"),
                  "duration_ms": payload.get("duration_ms")})
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(debug=True)