"""
The strength scale has four levels: 1 a little, 2 somewhat, 3 a lot, 4 a great deal.
  * the model's weights sit at 1/2/3/4, interpolate linearly in between, and stop at 4;
  * trained models never use more than 3 (a fourth level is only something a participant can choose);
  * a stronger setting never makes a factor count LESS (monotonic), and 4 outweighs 3;
  * the page offers the fourth level (button, slider to 4, "of 4", chart gridline) and the browser's weights agree with Python's.
"""
import json, os, random, re, subprocess, sys, tempfile

os.environ.setdefault("FLASK_SECRET_KEY", "test")
os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(), "t.db")
os.environ["RESPONSES_DIR"] = tempfile.mkdtemp()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import app as a, db, features                      # noqa: E402
from feature_list import twostage_model as tm      # noqa: E402

problems = []
def check(ok, msg):
    if not ok: problems.append(msg)

for m, want in ((0, 0), (0.5, 0.5), (1, 1), (1.5, 1.5), (2, 2), (2.5, 2.5), (3, 3), (3.5, 3.5), (4, 4), (5, 4), (9, 4)):
    check(abs(tm._amp_for(m) - want) < 1e-9, f"_amp_for({m}) should be {want}, got {tm._amp_for(m)}")
xs = [i / 10 for i in range(0, 50)]
check(all(tm._amp_for(x) <= tm._amp_for(y) + 1e-12 for x, y in zip(xs, xs[1:])), "weights must never decrease as strength increases")

# a factor at strength 4 outweighs the same factor at 3 (one factor on, nothing else)
random.seed(3)
K = features.FEATURE_KEYS; VM = features.values_map()
def model(strength):
    m = {"aggregation": "weighted", "features": {k: {"matters": 0, "dir": "favor", "mode": "steady", "cutoff": float(VM[k][1]), "cond": {"on": False, "when": k, "test": "nonzero", "n": 1}} for k in K}}
    m["features"]["years_waiting"]["matters"] = strength; return m
A = {k: float(VM[k][0]) for k in K}; B = dict(A); A["years_waiting"] = float(VM["years_waiting"][-1])
check(tm.predict_prob(model(4), A, B) > tm.predict_prob(model(3), A, B) > tm.predict_prob(model(2), A, B), "a stronger setting should count for more")

# the page offers it, and the browser's weights match Python's
c = a.app.test_client(); db.set_rotation_override("twostage_v2"); c.post("/start", data={"username": "scale_user", "model_type": "twostage_v2"})
for _ in range(20): c.post("/questionnaire/1/answer", data={"choice": random.choice("AB")})
html = c.get("/results").data.decode()
for needle in ("'a great deal'", 'max="4"', "of 4</span>", "AMP = [0, 1.0, 2.0, 3.0, 4.0]", "<small>a great deal</small>"):
    check(needle in html, f"the page is missing {needle!r}")
check("ampFor(model.features[k].matters) / 3" not in html, "no leftover /3 scaling")
js = re.search(r"function ampFor\(matters\) \{.*?\n\}\n", html, re.S).group(0)
test = "var AMP = [0, 1.0, 2.0, 3.0, 4.0];\n" + js + "console.log(JSON.stringify([0,0.5,1,1.5,2,2.5,3,3.5,4,5].map(ampFor)));"
try:
    out = subprocess.run(["node", "-e", test], capture_output=True, text=True, timeout=20)
    got = json.loads(out.stdout.strip()); want = [tm._amp_for(x) for x in (0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5)]
    check(all(abs(g - w) < 1e-9 for g, w in zip(got, want)), f"browser weights {got} differ from Python's {want}")
except FileNotFoundError:
    print("(skipped the browser comparison: node not found)")

# trained models only use 1-3
html = c.get("/results").data.decode()
served, _ = json.JSONDecoder().raw_decode(html, html.index("{", html.index("var model = ")))
check(all(f["matters"] <= 3 for f in served["features"].values()), "the trainer must not produce strengths above 3")

if problems:
    print("STRENGTH SCALE TEST FAILED:"); [print("  -", p) for p in problems]; sys.exit(1)
print("Strength scale OK: four levels (1 a little ... 4 a great deal), same weights in Python and the browser, trained models unchanged.")
