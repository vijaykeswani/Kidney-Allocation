"""
Guards the per-case explanations ("why did the model pick Patient B?").

  If node + jsdom are available, the JavaScript explanations shown on the model page (live and final) are checked:
       - tree: always name the same winner as the tree's own predict() and the page's own evaluator;
       - two-stage: always name the same winner as the server's predict(), across random models, all three combine rules,
         every shape, and exceptions (plus internal checks: tally counts, "most important" decider, notes only on disagreement).

Run:  python test_explanations.py        (set NODE_PATH to your jsdom location if it isn't found)
"""
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
tmp = tempfile.mkdtemp()
os.environ.update(FLASK_SECRET_KEY="expl-test", DB_PATH=os.path.join(tmp, "t.db"), RESPONSES_DIR=os.path.join(tmp, "resp"))

import app as a                                              # noqa: E402
import features                                              # noqa: E402
from decision_tree import fft_model                          # noqa: E402
from feature_list import twostage_model as ts                # noqa: E402

random.seed(11)
KEYS, VALS = features.FEATURE_KEYS, features.values_map()
NAMES = [f"{k}_diff" for k in KEYS]
failures = []


def patient():
    return {k: random.choice(VALS[k]) for k in KEYS}


def fake_decisions(n=20):
    out = []
    for _ in range(n):
        pa, pb = patient(), patient()
        r = {"choice": random.choice("AB")}
        r.update({f"A_{k}": pa[k] for k in KEYS}); r.update({f"B_{k}": pb[k] for k in KEYS})
        out.append(r)
    return out


def normalised(tree_dict):          # what the page actually receives
    return fft_model.FastFrugalTree.from_dict(tree_dict, feature_names=NAMES).to_dict()


# ---- trees: trained + one with a follow-up check + empty
trees = [train[0].to_dict() for train in (fft_model.train_fft(fake_decisions(), KEYS) for _ in range(40)) if train[0] is not None]
trees.append(normalised({"nodes": [
    {"feature": "years_waiting_diff", "op": "<=", "threshold": 2.0, "use_abs": False, "prefer_higher": True, "exit_class": 1,
     "refine": {"feature": "life_years_gained_diff", "op": ">=", "threshold": 5.0, "use_abs": True, "prefer_higher": True, "true_class": 1, "false_class": 0}},
    {"feature": "dependents_diff", "op": ">=", "threshold": 1.0, "use_abs": True, "prefer_higher": True, "exit_class": 1}], "default_class": 0}))
trees.append(normalised({"nodes": [], "default_class": 1}))

tree_cases = []
for t in trees:
    ft = fft_model.FastFrugalTree.from_dict(t, feature_names=NAMES)
    for _ in range(25):
        pa, pb, choice = patient(), patient(), random.choice("AB")
        F, _ = fft_model.build_difference_features([dict({"choice": "A"}, **{f"A_{k}": pa[k] for k in KEYS}, **{f"B_{k}": pb[k] for k in KEYS})], KEYS)
        pred = "A" if int(ft.predict(F.values)[0]) == 1 else "B"
        tree_cases.append({"tree": t, "A": pa, "B": pb, "choice": choice, "py_pred": pred})
print(f"tree: {len(tree_cases)} cases over {len(trees)} trees prepared with the tree's own predict() as ground truth")

# ---- two-stage cases for the JS check
def rand_model():
    m = {"aggregation": random.choice(["weighted", "tally", "prominent"]), "features": {}}
    for k in KEYS:
        other = random.choice([x for x in KEYS if x != k])
        m["features"][k] = {"matters": random.choice([0, 1, 1.5, 2, 2.7, 3]), "dir": random.choice(["favor", "against"]),
                            "mode": random.choice(["steady", "first", "last", "anyvsnone"]), "cutoff": float(random.choice(VALS[k][1:])),
                            "cond": {"on": random.random() < .3, "when": other, "test": random.choice(["zero", "nonzero", "atleast"]), "n": random.choice([1, 2])}}
    return m


ts_cases = []
for _ in range(120):
    m = rand_model()
    for _ in range(4):
        pa, pb = patient(), patient()
        ts_cases.append({"model": m, "A": pa, "B": pb, "choice": random.choice("AB"), "py_winner": "A" if ts.predict(m, pa, pb) == 1 else "B"})

# ---- JS parity (optional: needs node + jsdom)
node = shutil.which("node")
env = dict(os.environ)
have_jsdom = False
if node:
    for cand in (env.get("NODE_PATH", ""), os.path.join(ROOT, "node_modules"), subprocess.run(["npm", "root", "-g"], capture_output=True, text=True).stdout.strip() if shutil.which("npm") else ""):
        if cand and os.path.isdir(os.path.join(cand, "jsdom")):
            env["NODE_PATH"] = cand; have_jsdom = True; break
if have_jsdom:
    static_js = open(os.path.join(ROOT, "decision_tree", "static", "fft_tree.js")).read()
    pages = {}
    for arm, name in (("twostage_v2", "ts"), ("fft", "tree")):
        c = a.app.test_client(); c.post("/start", data={"username": "p_" + name, "model_type": arm})
        for _ in range(20):
            c.post("/questionnaire/1/answer", data={"choice": random.choice("AB")})
        html = c.get("/results").data.decode()
        html = re.sub(r'<script src="[^"]*fft_tree\.js[^"]*"></script>', lambda m: "<script>" + static_js + "</script>", html)
        pages[name] = os.path.join(tmp, name + ".html"); open(pages[name], "w").write(html)
    cases_path = os.path.join(tmp, "cases.json"); json.dump({"ts": ts_cases, "tree": tree_cases}, open(cases_path, "w"))
    r = subprocess.run([node, os.path.join(ROOT, "tests_js", "explain_parity.js"), cases_path, pages["ts"], pages["tree"]], capture_output=True, text=True, env=env)
    print("\n".join(l for l in r.stdout.splitlines() if l.strip()))
    if r.returncode != 0:
        failures.append("JavaScript explanations failed the parity / consistency check (see output above)")
else:
    print("SKIPPED the JavaScript parity check: node and/or jsdom not found (npm i jsdom, or set NODE_PATH).")

if failures:
    print("EXPLANATION TEST FAILED:")
    for f in sorted(set(failures)):
        print("  -", f)
    sys.exit(1)
print("Explanations OK.")
