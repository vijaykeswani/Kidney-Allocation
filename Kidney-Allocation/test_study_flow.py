"""
End-to-end regression test of the study flow, for every arm, on a throw-away database.

Checks the things that must not break:
  * Part 1 -> model screen -> LOCK -> Part 2 -> final -> survey -> done, for fft / twostage / twostage_v2 / control
  * the model is frozen at lock time and never retrained or edited afterwards
  * the final page reports how that frozen model did on the Part-2 hold-out, matching an independent recomputation
  * control gets its transition page and an unedited model; going to Part 2 directly still locks
  * the events needed for analysis are logged

Run:  python test_study_flow.py
"""
import json
import os
import random
import re
import sys
import tempfile

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
tmp = tempfile.mkdtemp()
os.environ.update(FLASK_SECRET_KEY="flow-test", DB_PATH=os.path.join(tmp, "t.db"), RESPONSES_DIR=os.path.join(tmp, "resp"))

import app as a                                   # noqa: E402
import db                                         # noqa: E402
import features                                   # noqa: E402
from feature_list import twostage_model          # noqa: E402
import decision_tree.handlers as dt               # noqa: E402

random.seed(7)
failures = []


def check(cond, msg):
    if not cond:
        failures.append(msg)


def answer_all(c, part, n):
    for _ in range(n):
        c.post(f"/questionnaire/{part}/answer", data={"choice": random.choice("AB")})


def events(user):
    return {e["event"] for e in db.list_events(user, limit=2000)}


def edited_twostage_model(c):
    html = c.get("/results").data.decode()
    model, _ = json.JSONDecoder().raw_decode(html, html.index("{", html.index("var model = ")))
    model["aggregation"] = "tally"
    model["features"]["dependents"].update(matters=3, dir="favor", mode="steady")
    return model


def run(arm, name, edit=True, direct_part2=False):
    c = a.app.test_client()
    check(c.post("/start", data={"username": name, "model_type": arm}).status_code == 302, f"{arm}: start")
    answer_all(c, 1, 20)
    kind = arm
    if arm == "control":
        r = c.get("/questionnaire/1")
        check(r.status_code == 302 and "/control/transition" in r.headers["Location"], "control: should go to the transition page after Part 1")
        check(b"trained an AI model" in c.get("/control/transition").data, "control: transition page text")
        kind = db.get_display_model_type(name)
    else:
        r = c.get("/results"); check(r.status_code == 200, f"{arm}: /results")
        body_html = r.data.decode()
        check("The goal of this model is to predict the choices you would make in the future" in body_html and "click “Lock in my model and continue”" in body_html and "of your previous answers correctly" in body_html, f"{arm}: the view/edit page needs the full introduction")
        check(body_html.index('id="ts-match-panel"') > body_html.index('id="ts-summary-text"'), f"{arm}: the match panel belongs at the bottom, because the introduction says so")
        check("Advanced settings" not in body_html, f"{arm}: no 'Advanced settings' toggle")
        check('id="ts-lock-gate"' in body_html and "REQUIRE_REVIEW = true" in body_html and "StudyReview.init(" in body_html, f"{arm}: lock-in must be gated on the participant having reviewed the model")
        check("The current version predicts" in body_html and 'id="ts-intro-x"' in body_html, f"{arm}: the intro needs its live 'predicts X/N correctly' line")
        if kind == "twostage_v2":     # the first-visit tutorial must be started AFTER the variables it reads exist (it silently never opened before)
            check(body_html.index("var TUTORIAL_MODE") < body_html.index("startTutorial();"), f"{arm}: startTutorial() runs before TUTORIAL_MODE is defined, so the tutorial never opens")
        check("built an AI model from your answers so far" in body_html, f"{arm}: the view/edit page needs its header")
        if kind in ("twostage_v2", "fft"):        # (the older v1 layout has no step cards)
            check('class="ts-step-body open" id="ts-step1-body"' in body_html, f"{arm}: Step 1 must be open on arrival so the model is visible without clicking")
        if kind == "twostage_v2":
            check('id="ts-feat-list"' in body_html and 'id="ts-review-progress"' in body_html and 'class="ts-step-body open" id="ts-step2-body"' not in body_html,
                  f"{arm}: v2 needs the factor list with its review progress, and Step 2 collapsed")
        if edit:
            if arm == "fft":
                db.set_fft_override(name, {"nodes": [{"feature": "years_waiting_diff", "op": ">=", "threshold": 4.0, "use_abs": True,
                                                      "prefer_higher": True, "exit_class": 1, "support": .5, "purity": .7}], "default_class": 0})
            else:
                r = c.post("/edit/apply", json={"model": edited_twostage_model(c)})
                check(r.status_code == 200, f"{arm}: apply edit")

    # ---- lock
    check(db.get_frozen_model(name) is None or arm == "control", f"{arm}: nothing should be locked before leaving the model screen")
    if direct_part2:
        c.get("/questionnaire/2")                 # skipping /part2 must still lock
    elif arm != "control":
        r = c.get("/part2"); check(r.status_code == 302, f"{arm}: /part2 redirect")
    frozen = db.get_frozen_model(name)
    check(frozen is not None, f"{arm}: model not locked")
    if not frozen:
        return
    check(frozen["arm"] == ("control" if arm == "control" else "treatment"), f"{arm}: frozen arm label")
    check(frozen["edited"] == (edit and arm != "control"), f"{arm}: frozen 'edited' flag is {frozen['edited']}")
    if frozen["edited"]:
        check(frozen["matches_before_edit"] is not None and frozen["matches_at_lock"] is not None, f"{arm}: lock should record matches before/after the edit")

    # ---- after the lock: no edits, no going back
    if arm != "control":
        check(c.post("/edit/apply", json={"model": {"x": 1}}).status_code == 409, f"{arm}: edits must be refused after locking")
        check(c.post("/edit/reset").status_code == 409, f"{arm}: reset must be refused after locking")
        r = c.get("/results"); check(r.status_code == 302 and "/questionnaire/2" in r.headers["Location"], f"{arm}: /results after locking should go on to Part 2")

    answer_all(c, 2, 10)
    r = c.get("/questionnaire/2"); check(r.status_code == 302 and "/final" in r.headers["Location"], f"{arm}: Part 2 should end at /final")
    f = c.get("/final"); check(f.status_code == 200, f"{arm}: /final")
    check("built an AI model from your answers so far" not in f.data.decode(), f"{arm}: the final page should not show the view/edit header")
    after = db.get_frozen_model(name)
    check(after["model"] == frozen["model"], f"{arm}: the model changed after it was locked (it must never be retrained)")

    # ---- final page numbers vs independent recomputation on the hold-out
    rec = db.load_all_users()[name]
    part2 = [d for d in rec["part2"] if d.get("choice") in ("A", "B")]
    if kind in ("twostage", "twostage_v2"):
        m, n = 0, 0
        for aa, bb, y in twostage_model._extract_queries(part2):
            m += twostage_model.predict(frozen["model"], aa, bb) == y; n += 1
    else:
        cases = dt._tree_cases(frozen["model"], part2); m, n = sum(x["match"] for x in cases), len(cases)
    body = f.data.decode()
    # The final page is composed in the browser (same page as the model screen, read-only): check what the server embeds.
    cases, _ = json.JSONDecoder().raw_decode(body, body.index("[", body.index("past_scenarios_json" if False else "var pastScenarios = ")))
    check(len(cases) == len(part2) == 10, f"{arm}: final page should embed exactly the 10 hold-out cases (got {len(cases)})")
    check("var FINAL_MODE = true" in body, f"{arm}: final page should be the read-only version of the model page")
    if kind in ("twostage", "twostage_v2"):
        emb, _ = json.JSONDecoder().raw_decode(body, body.index("{", body.index("var model = ")))
        check(emb == json.loads(json.dumps(frozen["model"])), f"{arm}: final page must embed exactly the locked model")
    else:
        emb, _ = json.JSONDecoder().raw_decode(body, body.index("{", body.rindex("initEditor(")))
        want = [(n["feature"], n["threshold"]) for n in frozen["model"]["nodes"]]
        check([(n["feature"], n["threshold"]) for n in emb["nodes"]] == want, f"{arm}: final page must embed exactly the locked tree")
    check(after.get("holdout", {}).get("matches") == m, f"{arm}: hold-out result not recorded correctly")
    check(b"/survey" in f.data, f"{arm}: final page must link to the survey")
    # ---- events + survey
    ev = events(name)
    for needed in ("model_locked", "final_viewed") + (("results_viewed",) if arm != "control" else ()):
        check(needed in ev, f"{arm}: event '{needed}' not logged")
    check(c.get("/survey").status_code == 200, f"{arm}: /survey")
    form = {"similarity": "4", "trust_stressed": "4", "trust_behalf_override": "4", "chatbot_freq": "daily", "chatbot_trust": "4", "age": "30", "gender": "woman"}
    import survey_spec
    for qid, _ in survey_spec.IMPACT_QUESTIONS:
        form[f"{qid}__seen_predictions"] = "3"
    r = c.post("/survey", data=form); check(r.status_code == 302 and "/done" in r.headers["Location"], f"{arm}: survey submit")
    check(db.get_survey_answers(name).get("_model_edited") == frozen["edited"], f"{arm}: survey should record whether the model was edited")


for arm in ("fft", "twostage", "twostage_v2"):
    run(arm, f"{arm}_edited", edit=True)
    run(arm, f"{arm}_plain", edit=False)
run("twostage_v2", "direct", edit=False, direct_part2=True)
for i in range(3):
    run("control", f"control{i}")

if failures:
    print("STUDY FLOW TEST FAILED:")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("Study flow OK: lock -> hold-out -> final -> survey verified for fft, twostage, twostage_v2 (edited and unedited) and control.")
