"""
Guards the optional "check one factor first" rule of the feature-list model.

  1. A model with no gate, or a switched-off gate, predicts EXACTLY as the model did before this feature existed
     (checked against a frozen copy of the old predictor on random models, all three combining rules).
  2. Semantics: inclusive threshold, higher/lower, the gate's own factor is ignored by the second stage, unsettled pairs
     fall through to the second stage, a malformed gate is treated as off.
  3. The server refuses malformed gates, and keeps well-formed ones.
  4. The hold-out result stored at lock-in uses the gate (checked independently of the model code).
  5. The trainer never creates a gate.
"""
import json, os, random, sys, tempfile

os.environ.setdefault("FLASK_SECRET_KEY", "test")
os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(), "t.db")
os.environ["RESPONSES_DIR"] = tempfile.mkdtemp()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import app as a                                   # noqa: E402
import db, features                               # noqa: E402
from feature_list import twostage_model as tm     # noqa: E402

problems = []
def check(ok, msg):
    if not ok: problems.append(msg)

random.seed(7)
K = features.FEATURE_KEYS
VM = features.values_map()

def patient(): return {k: float(random.choice(VM[k])) for k in K}

def rand_model():
    m = {"aggregation": random.choice(["weighted", "tally", "prominent"]), "features": {}}
    for k in K:
        other = random.choice(K)
        m["features"][k] = {"matters": random.choice([0, 1, 1.5, 2, 3]), "dir": random.choice(["favor", "against"]),
                            "mode": random.choice(["steady", "first", "last", "anyvsnone"]), "cutoff": float(random.choice(VM[k])),
                            "cond": {"on": random.random() < .3, "when": other, "test": random.choice(["zero", "nonzero", "atleast"]), "n": 1}}
    return m

def old_predict_prob(model, A, B):               # the predictor exactly as it was before the gate existed
    agg = model.get("aggregation", "weighted"); keys = features.FEATURE_KEYS
    if agg == "tally":
        pa = pb = 0
        for k in keys:
            if model["features"][k]["matters"] == 0: continue
            sa, sb = tm.contrib(model, k, A), tm.contrib(model, k, B)
            if abs(sa - sb) < 1e-9: continue
            if sa > sb: pa += 1
            else: pb += 1
        return tm._sigmoid(tm.TALLY_GAIN * (pa - pb))
    if agg == "prominent":
        order = sorted(keys, key=lambda k: tm._amp_for(model["features"][k]["matters"]), reverse=True)
        for k in order:
            if model["features"][k]["matters"] == 0: continue
            sa, sb = tm.contrib(model, k, A), tm.contrib(model, k, B)
            if abs(sa - sb) > 1e-9: return 0.85 if sa > sb else 0.15
        return 0.5
    return tm._sigmoid(tm.GAIN * sum(tm.contrib(model, k, A) - tm.contrib(model, k, B) for k in keys))

# ---- 1. no gate / switched-off gate == old behaviour
bad = 0
for _ in range(400):
    m = rand_model(); A, B = patient(), patient()
    for variant in (None, dict(tm.DEFAULT_GATE), {"on": False, "feature": "years_waiting", "threshold": 4, "dir": "higher"}, {"on": True, "feature": "nonsense", "threshold": 4, "dir": "higher"}):
        mm = json.loads(json.dumps(m))
        if variant is not None: mm["gate"] = variant
        if abs(tm.predict_prob(mm, A, B) - old_predict_prob(m, A, B)) > 1e-12: bad += 1
check(bad == 0, f"a model without an active gate must predict exactly as before ({bad} differences)")

# ---- 2. semantics
base = rand_model()
for k in K: base["features"][k].update(matters=0)
def gm(feature, t, d, agg="weighted"):
    m = json.loads(json.dumps(base)); m["aggregation"] = agg; m["gate"] = {"on": True, "feature": feature, "threshold": t, "dir": d}; return m
A, B = patient(), patient(); A["years_waiting"], B["years_waiting"] = 9.0, 5.0       # differ by exactly 4
check(tm.predict_prob(gm("years_waiting", 4, "higher"), A, B) == tm.GATE_PROB, "threshold is inclusive: a difference equal to it settles the pair")
check(tm.predict_prob(gm("years_waiting", 4, "lower"), A, B) == 1 - tm.GATE_PROB, "direction 'lower' gives the kidney to the patient with less")
check(tm.predict_prob(gm("years_waiting", 4, "higher"), B, A) == 1 - tm.GATE_PROB, "swapping the patients swaps the answer")
check(tm.predict_prob(gm("years_waiting", 6, "higher"), A, B) == 0.5, "an unsettled pair falls through (all other factors off -> 50/50)")
check(tm.gate_verdict(gm("years_waiting", 4, "higher"), A, B) == "A" and tm.gate_verdict(gm("years_waiting", 6, "higher"), A, B) is None, "gate_verdict")
m = gm("years_waiting", 6, "higher"); m["features"]["dependents"].update(matters=3, dir="favor", mode="steady")
A2, B2 = dict(A), dict(B); A2["dependents"], B2["dependents"] = 3.0, 0.0
check(tm.predict_prob(m, A2, B2) > 0.5, "unsettled pairs are decided by the other factors")
m["features"]["years_waiting"].update(matters=3, dir="against")                  # the gate's own factor must be ignored by stage 2
A3, B3 = dict(A2), dict(B2); A3["years_waiting"], B3["years_waiting"] = 5.0, 7.0
check(tm.predict_prob(m, A3, B3) == tm.predict_prob(m, A2, B2), "the gate factor takes no part in the second stage")
for agg in ("weighted", "tally", "prominent"):
    mm = gm("years_waiting", 4, "higher", agg); mm["features"]["dependents"].update(matters=3, dir="against", mode="steady")
    check(tm.predict(mm, A2, B2) == 1, f"{agg}: a settled pair ignores the other factors")

# ---- validation
for g, ok in ((None, True), ({"on": False}, True), ({"on": True, "feature": "years_waiting", "threshold": 4, "dir": "higher"}, True),
              ({"on": True, "feature": "x", "threshold": 4, "dir": "higher"}, False), ({"on": True, "feature": "years_waiting", "threshold": 0, "dir": "higher"}, False),
              ({"on": True, "feature": "years_waiting", "threshold": "4", "dir": "higher"}, False), ({"on": True, "feature": "years_waiting", "threshold": 4, "dir": "up"}, False),
              ({"on": "yes"}, False), ("on", False)):
    check((tm.validate_gate(g) is None) == ok, f"validate_gate({g!r}) should be {'accepted' if ok else 'rejected'}")
check("gate" in tm.sanitize_gate({"on": True, "feature": "years_waiting", "threshold": 4, "dir": "higher", "evil": 1}) or True, "")
check(set(tm.sanitize_gate({"on": True, "feature": "years_waiting", "threshold": 4, "dir": "higher", "evil": 1})) == {"on", "feature", "threshold", "dir"}, "sanitize keeps only the four known keys")

# ---- 3 + 4 + 5: through the real app
def answer(c, part, n):
    for _ in range(n): c.post(f"/questionnaire/{part}/answer", data={"choice": random.choice("AB")})

c = a.app.test_client(); name = "gate_user"
db.set_rotation_override("twostage_v3")           # /start takes its arm from the rotation
c.post("/start", data={"username": name, "model_type": "twostage_v3"}); answer(c, 1, 20)
html = c.get("/results").data.decode()
served, _ = json.JSONDecoder().raw_decode(html, html.index("{", html.index("var model = ")))
check(served.get("gate", {}).get("on") is False, "the trainer must never switch a gate on by itself")
model = json.loads(json.dumps(served)); model["aggregation"] = "weighted"
for k in K: model["features"][k]["matters"] = 0
for bad_gate in ({"on": True, "feature": "x", "threshold": 4, "dir": "higher"}, {"on": True, "feature": "years_waiting", "threshold": -1, "dir": "higher"}, {"on": 1}):
    r = c.post("/edit/apply", json={"model": dict(model, gate=bad_gate)}); check(r.status_code == 400, f"a malformed gate must be refused ({bad_gate})")
check(c.post("/edit/apply", json={"model": {"x": 1}}).status_code == 400, "a model that is not a feature-list model must be refused")
model["gate"] = {"on": True, "feature": "years_waiting", "threshold": 4, "dir": "higher", "evil": 1}
check(c.post("/edit/apply", json={"model": model}).status_code == 200, "a well-formed gate is accepted")
saved = db.get_fft_override(name)
check(saved["gate"] == {"on": True, "feature": "years_waiting", "threshold": 4.0, "dir": "higher"}, f"the saved gate should be sanitized, got {saved.get('gate')}")
c.get("/part2"); answer(c, 2, 10); c.get("/final")
frozen = db.get_frozen_model(name); rec = db.load_all_users()[name]
part2 = [d for d in rec["part2"] if d.get("choice") in ("A", "B")]
def expected(d):                                  # worked out here, without the model code: the gate decides, everything else is off -> "B"
    diff = d["A_years_waiting"] - d["B_years_waiting"]
    return ("A" if diff > 0 else "B") if abs(diff) >= 4 else "B"
want = sum(1 for d in part2 if expected(d) == d["choice"])
check(frozen["holdout"]["matches"] == want and len(part2) == 10, f"hold-out must use the gate (stored {frozen['holdout']['matches']}, independent count {want})")
check(frozen["model"]["gate"]["on"] is True, "the locked model keeps its gate")

if problems:
    print("GATE TEST FAILED:"); [print("  -", p) for p in problems]; sys.exit(1)
print("Gate OK: identical to the old model when off, correct first-check semantics, validated on save, used by the stored hold-out result, never created by training.")
