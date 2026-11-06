"""
The version picker on the intro page (testing aid):
  * OFF (the default): no picker is shown, and a posted choice is ignored — the rotation decides, as in the real study.
  * ON: the picker offers Automatic / Decision tree / Two-stage (feature list v2) / Control (no v1); a choice is used and does
    NOT advance the rotation; Automatic (or anything invalid) falls back to the rotation; ?model=... preselects it.
"""
import os, re, sys, tempfile

os.environ.setdefault("FLASK_SECRET_KEY", "test")
os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(), "t.db")
os.environ["RESPONSES_DIR"] = tempfile.mkdtemp()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import app as a   # noqa: E402
import db         # noqa: E402

problems = []
def check(ok, msg):
    if not ok: problems.append(msg)

def start(name, **form):
    c = a.app.test_client(); c.post("/consent")
    r = c.post("/start", data={"username": name, **form})
    return c, r

# ---------------- OFF (default) ----------------
a.ALLOW_MODEL_CHOICE = False
html = a.app.test_client().get("/intro?model=fft").data.decode()
check('name="model_type"' not in html, "the picker must not be shown unless ALLOW_MODEL_CHOICE is on")
idx0 = db.get_rotation_index(); expected = a.ROTATION_SEQUENCE[idx0 % len(a.ROTATION_SEQUENCE)]
other = next(m for m in ("fft", "twostage_v2", "control") if m != expected)
start("off_user", model_type=other)
check(db.get_model_type("off_user") == expected, f"with the picker off a posted choice must be ignored (got {db.get_model_type('off_user')}, rotation says {expected})")
check(db.get_rotation_index() == idx0 + 1, "the rotation should advance when it assigns the arm")

# ---------------- ON ----------------
a.ALLOW_MODEL_CHOICE = True
html = a.app.test_client().get("/intro").data.decode()
opts = re.findall(r'<option value="([^"]*)"', html.split('name="model_type"')[1].split("</select>")[0])
check(opts == ["", "fft", "twostage_v2", "control"], f"picker options should be Automatic + the three versions, no v1 (got {opts})")
check("Two-stage model" in html and "Decision tree" in html and "Control" in html, "picker labels")
check("twostage\"" not in html.split('name="model_type"')[1].split("</select>")[0], "v1 must not be offered")
check(re.search(r'<option value="twostage_v2"\s+selected', a.app.test_client().get("/intro?model=twostage_v2").data.decode()) is not None, "?model=twostage_v2 should preselect it")
check(re.search(r'<option value="fft"\s+selected', a.app.test_client().get("/intro").data.decode()) is None, "nothing preselected by default")
for choice in ("fft", "twostage_v2", "control"):
    before = db.get_rotation_index()
    c, r = start("pick_" + choice, model_type=choice)
    check(r.status_code == 302, f"{choice}: start should redirect")
    check(db.get_model_type("pick_" + choice) == choice, f"{choice}: the chosen version should be used (got {db.get_model_type('pick_' + choice)})")
    check(db.get_rotation_index() == before, f"{choice}: a manual choice must not advance the rotation")
    with c.session_transaction() as s:
        check(s.get("model_type") == choice, f"{choice}: session should carry the chosen version")
before = db.get_rotation_index(); expected = a.ROTATION_SEQUENCE[before % len(a.ROTATION_SEQUENCE)]
start("auto_user", model_type="")
check(db.get_model_type("auto_user") == expected and db.get_rotation_index() == before + 1, "Automatic should follow (and advance) the rotation")
before = db.get_rotation_index(); expected = a.ROTATION_SEQUENCE[before % len(a.ROTATION_SEQUENCE)]
start("bogus_user", model_type="bogus")
check(db.get_model_type("bogus_user") == expected, "an invalid choice falls back to the rotation")
start("v1_user", model_type="twostage")
check(db.get_model_type("v1_user") == "twostage", "v1 is not offered, but the server still accepts it (the existing tests rely on that)")
c, r = a.app.test_client(), None
c.post("/consent"); r = c.post("/start", data={"username": "x", "model_type": "fft"})
check(r.status_code == 200 and 'value="fft"' in r.data.decode() and re.search(r'<option value="fft"\s+selected', r.data.decode()), "a rejected (too short) ID keeps the selected version")

if problems:
    print("MODEL CHOICE TEST FAILED:"); [print("  -", p) for p in problems]; sys.exit(1)
print("Model choice OK: hidden by default; when enabled, Automatic / Decision tree / Two-stage / Control, no v1, rotation untouched.")