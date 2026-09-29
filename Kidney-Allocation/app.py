"""
Preference Elicitation Portal — Flask edition
SURA 2026 · IIT Delhi

Run:  gunicorn app:app --bind 0.0.0.0:5001 --workers 1 --timeout 120
"""

import json
import os
import random
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session, jsonify

import db
import features
import tiebreaker
import twostage_model
from fft_model import train_fft, FastFrugalTree, build_difference_features, augment, feature_row
from fft_component import fft_svg_explained, DEFAULT_FFT_PALETTE, PARAM_DIRECTION_LABELS

APP_DIR       = os.path.dirname(os.path.abspath(__file__))
RESPONSES_DIR = os.environ.get("RESPONSES_DIR") or os.path.join(APP_DIR, "responses")

# The set of features is defined in features.json (loaded by features.py) —
# not here. Swap in different features by editing that file only.
FEATURES = features.FEATURE_KEYS

PARAM_DESCRIPTIONS = features.descriptions()

# key -> display label (e.g. "years_waiting" -> "Years Waiting"), from
# features.json — used anywhere a feature name is shown to the person,
# instead of mechanically title-casing the key (which breaks for anything
# that doesn't title-case nicely, e.g. a future "bmi" or "hla_match" key).
FEATURE_LABELS = {k: v["label"] for k, v in features.FEATURE_BY_KEY.items()}

PART1_N = 20
PART2_N = 10

# How many new checks a participant may add to the model in one editing pass.
# Enforced in the browser (so the limit is visible while editing) and again in
# apply_edit (so the stored tree can't drift past it).
MAX_NEW_FEATURES = 2

# Create the SQL schema (and import users.json once, if present) at startup.
db.init_db(FEATURES)

app = Flask(__name__)
app.secret_key = os.environ["FLASK_SECRET_KEY"]

_train_cache = {}


def train_fft_cached(decisions, params, override):
    key = (json.dumps(decisions, sort_keys=True), tuple(params), json.dumps(override or {}))
    if key not in _train_cache:
        _train_cache[key] = train_fft(
            json.dumps(decisions), list(params), json.dumps(override) if override else None
        )
    return _train_cache[key]


# ── Scenario generation (fully synthetic — no CSV dependency) ────────────────

# Reasonable real-world ranges per feature, used to sample random A/B patient
# pairs. All five features are integers in this study.
FEATURE_RANGES = features.ranges()
# Features with a fixed discrete value set (e.g. rejection_risk: [1,2,5,10]).
FEATURE_VALUES = features.values_map()


def generate_scenarios(n, seed=None):
    """n random A/B patient pairs, sampled uniformly within FEATURE_RANGES.
    Features with a 'values' list in features.json sample from that list instead."""
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
    """Stable identity for an A/B pair, used to avoid repeating scenarios."""
    return (
        tuple(round(float(sc["A"][p]), 3) for p in FEATURES)
        + tuple(round(float(sc["B"][p]), 3) for p in FEATURES)
    )


def _pair_distance(sc, other):
    """Total absolute difference between two pairs, across both patients."""
    return sum(
        abs(float(sc[side][p]) - float(other[side][p]))
        for side in ("A", "B") for p in FEATURES
    )


def generate_distinct_scenario(seen_keys, recent=(), tries=60, min_distance=8.0):
    """
    Produce a scenario the participant has not been shown before.

    Two things make it "genuinely different" rather than merely re-rolled:
    exact repeats are rejected outright via `seen_keys`, and near-repeats of the
    most recent pairs are rejected via a minimum total distance. The best
    candidate seen so far is kept as a fallback, so this always returns
    something even if the sampler gets unlucky.
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
    """
    Randomise the order in which a set of scenarios is presented.

    A fresh, entropy-seeded RNG is drawn on every call (rather than reusing a
    module-level `random` instance), so the presentation order is independently
    randomised each time this runs — per participant, per run of the app.

    Only the order changes: the scenario content and the generation logic that
    produced it are left untouched. The shuffled order is persisted with the
    participant's record, so the sequence stays stable if they resume a session
    and the recorded `scenario` number keeps pointing at the same pair.
    """
    order = list(scenarios)
    random.SystemRandom().shuffle(order)
    return order


# ── User persistence ──────────────────────────────────────────────────────────

def load_users():
    return db.load_all_users()


def save_users(u):
    db.save_all_users(u)


def get_user_record():
    username = session.get("username")
    if not username:
        return None, None
    users = load_users()
    return users, users.setdefault(username, {})


def save_user_record(record):
    db.save_user_record(session["username"], record)


def save_fft_override(username, tree_dict):
    db.set_fft_override(username, tree_dict)
    _train_cache.clear()


def load_fft_override(username):
    return db.get_fft_override(username)


def clear_fft_override(username):
    db.delete_fft_override(username)
    _train_cache.clear()


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
    save_user_record(rec)
    return rec


# ── Routes ────────────────────────────────────────────────────────────────────

MODEL_TYPES = ("fft", "twostage")


def _requested_model():
    """`?model=fft|twostage` on a study link preselects the model type;
    anything else leaves it unset so the start page shows the picker."""
    model_type = request.args.get("model")
    return model_type if model_type in MODEL_TYPES else None


@app.route("/")
def start():
    # Consent page always shown — no session check so every new visit requires consent.
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

    users, rec = get_user_record()
    if "part1_scenarios" not in rec:
        rec["part1_scenarios"] = shuffle_scenarios(generate_scenarios(PART1_N))
        rec["part1"] = []
        rec["part1_index"] = 0
    if "part2_scenarios" not in rec:
        rec["part2_scenarios"] = shuffle_scenarios(generate_scenarios(PART2_N))
        rec["part2"] = []
        rec["part2_index"] = 0
    save_user_record(rec)

    return redirect(url_for("questionnaire", part=1))


def _model_type():
    """The current user's model type — session first (fast path), falling
    back to the DB (e.g. after a server restart clears sessions)."""
    if "model_type" in session:
        return session["model_type"]
    if "username" in session:
        mt = db.get_model_type(session["username"])
        session["model_type"] = mt
        return mt
    return "fft"


def _require_user():
    if "username" not in session:
        return redirect(url_for("intro"))
    return None


@app.route("/questionnaire/<int:part>")
def questionnaire(part):
    guard = _require_user()
    if guard:
        return guard
    users, rec = get_user_record()
    scenarios = rec.get(f"part{part}_scenarios", [])
    idx = rec.get(f"part{part}_index", 0)
    n_total = len(scenarios)

    if idx >= n_total:
        return redirect(url_for("results") if part == 1 else url_for("final"))

    heading = "Which patient should receive the organ?" if part == 1 else "Part 2: A few more scenarios"
    subheading = None if part == 1 else (
        "Now that you've seen your model, here are a few more pairs in the same "
        "format. This checks whether the model still matches how you'd actually choose."
    )
    return render_template(
        "questionnaire.html",
        part=part, idx=idx, n_total=n_total,
        sc=scenarios[idx], features=FEATURES,
        feature_labels=FEATURE_LABELS,
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


def _train_for(rec, decisions_field, override):
    decisions = [d for d in rec.get(decisions_field, []) if d.get("choice") in ("A", "B")]
    if len(decisions) < 6:
        return None, decisions
    fft, nodes_df, stats, feat_names, error = train_fft_cached(decisions, FEATURES, override)
    if error or stats is None:
        return None, decisions
    return stats, decisions


def _live_tree(rec, decisions_field="part1"):
    """
    Rebuild the participant's current tree (edits included) for server-side
    prediction, together with the optional tie-breaker chain.

    Returns (tree, feat_names, chain) or (None, [], []) when there isn't enough
    data yet. The tree is reconstructed from the same dict the browser is
    editing, so the server and the page always agree on what the model says.
    """
    decisions = [d for d in rec.get(decisions_field, []) if d.get("choice") in ("A", "B")]
    if len(decisions) < 6:
        return None, [], []

    override = load_fft_override(session["username"])
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
    """
    Plain-language account of which step decided the case.

    Deliberately describes the model's own route only. It says nothing about
    whether the participant agreed — that comparison belongs to the review
    page, which states it once, in its own words.
    """
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
    """
    Compact A/B/choice-only view of past decisions, in the order they were
    answered, for the client-side "check the model against your answers"
    panel. Only what's needed to re-evaluate the tree in the browser.
    """
    out = []
    for d in decisions:
        out.append({
            "A": {p: d.get(f"A_{p}") for p in FEATURES},
            "B": {p: d.get(f"B_{p}") for p in FEATURES},
            "choice": d.get("choice"),
        })
    return out


@app.route("/results")
def results():
    guard = _require_user()
    if guard:
        return guard
    if _model_type() == "twostage":
        return _twostage_results()

    users, rec = get_user_record()
    override = load_fft_override(session["username"])
    stats, decisions = _train_for(rec, "part1", override)

    if stats is None:
        return render_template("not_ready.html", n_decisions=len(decisions))

    username = session["username"]
    version = db.get_model_version(username)

    # Tutorial timing is driven by how many times this page has been opened:
    # the walkthrough on the first visit, a short reminder a couple of visits
    # later, and nothing after that. Counting views (rather than dismissals)
    # is what lets the reminder actually arrive — a dismissal counter would
    # stop at one and the reminder would never fire.
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


def _twostage_results():
    """Two-stage model's version of /results — separate render path, reusing
    the same session/rating/tutorial plumbing as the tree."""
    users, rec = get_user_record()
    decisions = [d for d in rec.get("part1", []) if d.get("choice") in ("A", "B")]
    override = load_fft_override(session["username"])
    stats, err = twostage_model.train_twostage(decisions, override=override)

    if stats is None:
        return render_template("not_ready.html", n_decisions=len(decisions))

    # If the model has been edited, "started at X of 20" needs the freshly-
    # trained (un-edited) model's match count too — cheap enough to just
    # train it a second time rather than caching, given how few decisions
    # this ever runs against.
    baseline_matches = stats["n_matches"]
    if override:
        baseline_stats, _ = twostage_model.train_twostage(decisions, override=None)
        if baseline_stats:
            baseline_matches = baseline_stats["n_matches"]

    username = session["username"]
    views = db.get_tutorial_seen_count(username)
    tutorial_mode = "full" if views == 0 else ("reminder" if views == 2 else "")
    db.bump_tutorial_seen_count(username)

    return render_template(
        "twostage.html",
        final_mode=False,
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


def _feature_ranges():
    """Per-feature (min, max, step) from FEATURE_RANGES — bounds the sliders."""
    out = {}
    for p in FEATURES:
        lo, hi = FEATURE_RANGES[p]
        out[p] = {"min": lo, "max": hi, "step": 1}
    return out


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
    save_user_record(rec)
    return jsonify({"ok": True})


@app.route("/edit/apply", methods=["POST"])
def apply_edit():
    guard = _require_user()
    if guard:
        return jsonify({"error": "not logged in"}), 401
    payload = request.get_json(force=True, silent=True) or {}
    username = session["username"]

    if _model_type() == "twostage":
        model = payload.get("model")
        if not model:
            return jsonify({"error": "missing model"}), 400
        save_fft_override(username, model)  # same override column, holds either schema
        version = db.bump_model_version(username)
        db.log_event(username, "model_edit_applied", {"model_version": version, "type": "twostage"})
        return jsonify({"ok": True, "redirect": url_for("results")})

    tree = payload.get("tree")
    if not tree:
        return jsonify({"error": "missing tree"}), 400

    # Backstop for the browser-side limit. `baseline_node_count` is what the
    # page was opened with; anything beyond it plus the cap is rejected rather
    # than silently trimmed, so the participant is never told an edit saved
    # when part of it did not.
    baseline = payload.get("baseline_node_count")
    nodes = tree.get("nodes") or []
    if isinstance(baseline, int) and len(nodes) - baseline > MAX_NEW_FEATURES:
        return jsonify({
            "error": f"You can add at most {MAX_NEW_FEATURES} new checks at a time.",
            "limit": MAX_NEW_FEATURES,
        }), 400

    save_fft_override(username, tree)
    version = db.bump_model_version(username)
    db.log_event(username, "model_edit_applied",
                 {"nodes": len(nodes), "model_version": version})
    return jsonify({"ok": True, "redirect": url_for("results")})


@app.route("/edit/reset", methods=["POST"])
def reset_edit():
    guard = _require_user()
    if guard:
        return jsonify({"error": "not logged in"}), 401
    username = session["username"]
    clear_fft_override(username)
    version = db.bump_model_version(username)
    db.log_event(username, "model_reset", {"model_version": version})
    return jsonify({"ok": True, "redirect": url_for("results")})


# ── Review flow: new scenarios, agreements / disagreements, feedback ─────────

@app.route("/scenario/new", methods=["POST"])
def new_scenario():
    """
    Hand back a scenario this participant has not seen, and open a trial row
    for it. Each call produces its own trial, so every scenario carries its own
    agreement/disagreement record instead of overwriting the last one.
    """
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
    """Record the participant's pick, then the model's, and compare them."""
    guard = _require_user()
    if guard:
        return jsonify({"error": "not logged in"}), 401

    payload = request.get_json(force=True, silent=True) or {}
    trial_id = payload.get("trial_id")
    choice = payload.get("choice")
    if choice not in ("A", "B") or trial_id is None:
        return jsonify({"error": "invalid answer"}), 400

    username = session["username"]
    trial = db.get_trial(trial_id, username)
    if not trial:
        return jsonify({"error": "unknown scenario"}), 404

    users, rec = get_user_record()
    tree, feat_names, chain = _live_tree(rec)
    if tree is None:
        return jsonify({"error": "model not ready"}), 400

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

    return jsonify({
        "ok": True,
        "trial_id": trial_id,
        "user_choice": choice,
        "model_choice": result["choice"],
        "agreed": choice == result["choice"],
        "reason": result["reason"],
        "ambiguous": result.get("ambiguous", False),
        "tiebreak_used": result.get("tiebreak_used", False),
    })


@app.route("/scenario/feedback", methods=["POST"])
def scenario_feedback():
    """Store how the participant's reasoning differed from the model's."""
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
    """
    Agreements and disagreements, grouped by the model version that produced
    them. Older versions stay visible below the current one so a participant
    can see how the model's behaviour changed after they edited it.
    """
    guard = _require_user()
    if guard:
        return guard

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
        feature_labels=FEATURE_LABELS,
        back_url=url_for("results"),
        new_scenario_url=url_for("new_scenario"),
        current_version=current_version,
    )


@app.route("/log", methods=["POST"])
def log_event():
    """Best-effort interaction logging from the browser."""
    if "username" not in session:
        return jsonify({"ok": False}), 401
    payload = request.get_json(force=True, silent=True) or {}
    name = payload.get("event")
    if name:
        db.log_event(session["username"], name, payload.get("detail"))
    return jsonify({"ok": True})


@app.route("/tutorial/seen", methods=["POST"])
def tutorial_seen():
    """
    Record that the tutorial was dismissed. The show/hide decision itself is
    made from the page-view count in `results()`; this endpoint only captures
    how the participant got rid of it, which is useful when reviewing whether
    the onboarding landed.
    """
    if "username" not in session:
        return jsonify({"ok": False}), 401
    payload = request.get_json(force=True, silent=True) or {}
    db.log_event(session["username"], "tutorial_dismissed",
                 {"mode": payload.get("mode"), "via": payload.get("via"),
                  "duration_ms": payload.get("duration_ms")})
    return jsonify({"ok": True})


@app.route("/part2")
def part2_intro():
    guard = _require_user()
    if guard:
        return guard
    return redirect(url_for("questionnaire", part=2))


@app.route("/final")
def final():
    guard = _require_user()
    if guard:
        return guard
    if _model_type() == "twostage":
        return _twostage_final()

    users, rec = get_user_record()
    override = load_fft_override(session["username"])

    all_decisions = (
        [d for d in rec.get("part1", []) if d.get("choice") in ("A", "B")]
        + [d for d in rec.get("part2", []) if d.get("choice") in ("A", "B")]
    )
    if len(all_decisions) < 6:
        return render_template("not_ready.html", n_decisions=len(all_decisions))

    fft, nodes_df, stats, feat_names, error = train_fft_cached(all_decisions, FEATURES, override)
    if error or stats is None:
        return render_template("not_ready.html", n_decisions=len(all_decisions))

    svg = fft_svg_explained(
        tree=stats["tree"],
        palette=DEFAULT_FFT_PALETTE,
        node_explanations=stats.get("node_explanations"),
        summary_explanation=stats.get("summary_explanation"),
    )
    return render_template("final.html", svg=svg)


def _twostage_final():
    users, rec = get_user_record()
    override = load_fft_override(session["username"])
    all_decisions = (
        [d for d in rec.get("part1", []) if d.get("choice") in ("A", "B")]
        + [d for d in rec.get("part2", []) if d.get("choice") in ("A", "B")]
    )
    if len(all_decisions) < 6:
        return render_template("not_ready.html", n_decisions=len(all_decisions))
    stats, err = twostage_model.train_twostage(all_decisions, override=override)
    if stats is None:
        return render_template("not_ready.html", n_decisions=len(all_decisions))
    return render_template(
        "twostage.html",
        final_mode=True,
        apply_url="", reset_url="", rate_url="", continue_url="", tutorial_seen_url="",
        current_score=None, has_override=False,
        model_json=stats["model"],
        n_matches=stats["n_matches"],
        n_total=stats["n_total"],
        baseline_matches=stats["n_matches"],
        summary_text=stats["summary"],
        past_scenarios_json=_scenarios_for_audit(all_decisions),
        feature_ranges_json=_feature_ranges(),
        features_json=features.FEATURES_DATA,
        tutorial_mode="",
    )


if __name__ == "__main__":
    app.run(debug=True)