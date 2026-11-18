"""
The decision tree's rules are written as sentences; the blue parts are drop-downs (factor, size of the gap, which patient to choose).
  * no equation like |factor(A) - factor(B)| >= 4 is built anywhere, and the node edit / outcome / follow-up panels are gone;
  * the drop-down options match each factor's real values, and a change reaches the tree (checked in jsdom, on the real page);
  * follow-up checks can be removed and added again; a new rule starts with a real gap; the read-only page shows plain values.
Run:  python test_tree_rules.py     (needs node + jsdom for the page check: npm i jsdom, or set NODE_PATH)
"""
import os, re, shutil, subprocess, sys, tempfile, random

os.environ.setdefault("FLASK_SECRET_KEY", "test")
tmp = tempfile.mkdtemp()
os.environ["DB_PATH"] = os.path.join(tmp, "t.db"); os.environ["RESPONSES_DIR"] = os.path.join(tmp, "r")
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
import app as a, db   # noqa: E402

failures = []
src = open(os.path.join(ROOT, "decision_tree", "templates", "model.html"), encoding="utf-8").read()
static_js = open(os.path.join(ROOT, "decision_tree", "static", "fft_tree.js"), encoding="utf-8").read()
for pid in ('id="edit-panel"', 'id="leaf-panel"', 'id="refine-panel"'):
    if pid in src: failures.append(f"the page still contains {pid}")
if "'|' + H.prettyFeature" in static_js or "H.nodeCondText(" in static_js: failures.append("fft_tree.js still builds an equation line")
if "data-add-followup" in static_js or "addTieBreakerToNode" in src: failures.append("'add a follow-up check' should be gone for now")
if "function ruleHTML(" not in src: failures.append("the page has no rule-sentence builder")

node = shutil.which("node"); env = dict(os.environ); have = False
if node:
    for cand in (env.get("NODE_PATH", ""), os.path.join(ROOT, "node_modules"), subprocess.run(["npm", "root", "-g"], capture_output=True, text=True).stdout.strip() if shutil.which("npm") else ""):
        if cand and os.path.isdir(os.path.join(cand, "jsdom")): env["NODE_PATH"] = cand; have = True; break
if have:
    random.seed(1)
    c = a.app.test_client(); c.post("/consent"); a.ALLOW_MODEL_CHOICE = True
    c.post("/start", data={"username": "rules_user", "model_type": "fft"})
    for _ in range(20): c.post("/questionnaire/1/answer", data={"choice": random.choice("AB")})
    db.set_fft_override("rules_user", {"nodes": [
        {"feature": "years_waiting_diff", "op": "<=", "threshold": 2, "use_abs": True, "prefer_higher": True, "exit_class": 1, "support": .5, "purity": .7,
         "refine": {"feature": "life_years_gained_diff", "op": ">=", "threshold": 10, "use_abs": True, "prefer_higher": True, "true_class": 1, "false_class": 0,
                    "support": .5, "false_support": .5, "purity": .7, "false_purity": .6, "manual": True}},
        {"feature": "rejection_risk_diff", "op": ">=", "threshold": 5, "use_abs": True, "prefer_higher": False, "exit_class": 0, "support": .5, "purity": .7}], "default_class": 0})
    html = c.get("/results").data.decode()
    if 'id="edit-panel"' in html: failures.append("the served page still has the edit panel")
    html = re.sub(r'<script src="[^"]*fft_tree\.js[^"]*"></script>', lambda m: "<script>" + static_js + "</script>", html)
    page = os.path.join(tmp, "tree.html"); open(page, "w", encoding="utf-8").write(html)
    r = subprocess.run([node, os.path.join(ROOT, "tests_js", "tree_rules.js"), page], capture_output=True, text=True, env=env)
    print("\n".join(l for l in r.stdout.splitlines() if l.strip()))
    if r.returncode != 0: failures.append("the page check failed (see above)")
else:
    print("SKIPPED the page check: node and/or jsdom not found (npm i jsdom, or set NODE_PATH).")

if failures:
    print("TREE RULES TEST FAILED:"); [print("  -", f) for f in failures]; sys.exit(1)
print("Tree rules OK.")
