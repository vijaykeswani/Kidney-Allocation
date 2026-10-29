"""
Guards the two pages before the study starts:
  * consent page: the agreed Key Information text, consent by pressing "Next", no researcher name / institute
  * intro page: why it's a dilemma, what the participant will do, EVERY factor described (nothing collapsed), an example
    comparison, two choices, a random pre-filled (editable) participant ID, and an "I am ready to start" button
"""
import os, re, sys, tempfile

os.environ.setdefault("FLASK_SECRET_KEY", "test")
os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(), "t.db")
os.environ["RESPONSES_DIR"] = tempfile.mkdtemp()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import app as a      # noqa: E402
import db            # noqa: E402
import features      # noqa: E402

problems = []
def check(ok, msg):
    if not ok: problems.append(msg)

c = a.app.test_client()

# ---------------- consent page ----------------
html = c.get("/").data.decode()
for needle in ("Key Information", "Purpose", "Procedures", "Voluntariness", "Confidentiality", "Risks and benefits",
               "allocation of medical resources", "answer questions about scenarios involving the use of AI",
               "approximately 20-30 minutes or less", "closing your Internet browser window",
               "De-identified information collected in this study", "There are no\n  benefits", "press “Next” to continue"):
    check(needle.replace("\n  ", " ") in re.sub(r"\s+", " ", html) or needle in html, f"consent page is missing: {needle!r}")
check("Keswani" not in html and "Vijay" not in html and "IIT Delhi" not in html, "consent page must not name the researcher or the institute (for now)")
check('type="checkbox"' not in html, "consent is given by pressing Next: no checkboxes")
check(re.search(r">\s*Next", html) is not None, "consent button should say Next")
r = c.post("/consent", data={})
check(r.status_code == 302 and "/intro" in r.headers["Location"], "pressing Next should go to the intro page")

# ---------------- intro page ----------------
html = c.get("/intro?model=twostage_v2").data.decode()
flat = re.sub(r"\s+", " ", html)
for needle in ("moral dilemma", "far more patients waiting for a kidney than there are kidneys available",
               "You will see pairs of patients who each need a kidney transplant and choose who you think should receive it. A simple AI model — a computer program trained on your choices — will attempt to capture the pattern in your responses to predict your future choices.",
               "You will start with 20 pairs", "two choices", "Choose Patient A", "Choose Patient B", "I am ready to start"):
    check(needle in flat, f"intro page is missing: {needle!r}")
check("<details" not in html, "nothing on the intro page may be collapsed")
for k in features.FEATURE_KEYS:
    fk = features.FEATURE_BY_KEY[k]
    check(fk["label"] in html and fk["description"] in html, f"intro page must describe {k} in full")
# the example uses only values the study can really show, with the same % formatting
for side in ("A", "B"):
    for k in features.FEATURE_KEYS:
        check(a.EXAMPLE_PAIR[side][k] in features.values_map()[k], f"example value {side}/{k} is not a value the study uses")
check("%" in html.split('class="example-frame"')[1].split("Choose Patient A")[0], "the example should show the % on rejection risk")
m = re.search(r'name="username"[^>]*value="([^"]+)"', html)
check(m is not None and re.fullmatch(r"P-[A-Z2-9]{6}", m.group(1)) is not None, f"participant ID should be pre-filled like P-XXXXXX (got {m.group(1) if m else None})")
check("autofocus" not in html, "autofocus would scroll a long page straight to the bottom")
ids = {re.search(r'name="username"[^>]*value="([^"]+)"', c.get("/intro?model=fft").data.decode()).group(1) for _ in range(8)}
check(len(ids) == 8, "each visit should get a different random ID")

# the generator never hands out an ID that is already taken
calls = {"n": 0}
real = db.user_exists
def fake(u): calls["n"] += 1; return calls["n"] <= 3
db.user_exists = fake
pid = a._new_participant_id(); db.user_exists = real
check(calls["n"] == 4, f"generator should retry until it finds a free ID (looked {calls['n']} times)")

# ---------------- starting with the default ID, with an edited one, and with a bad one ----------------
c2 = a.app.test_client(); c2.post("/consent")
pid = re.search(r'name="username"[^>]*value="([^"]+)"', c2.get("/intro?model=twostage_v2").data.decode()).group(1)
r = c2.post("/start", data={"username": pid, "model_type": "twostage_v2"})
check(r.status_code == 302 and "/questionnaire/1" in r.headers["Location"], "starting with the pre-filled ID should work")
check(db.user_exists(pid), "the participant should exist under the pre-filled ID")
c3 = a.app.test_client(); r = c3.post("/start", data={"username": "my own id", "model_type": "fft"})
check(r.status_code == 302 and db.user_exists("my own id"), "the ID field must stay editable")
c4 = a.app.test_client(); r = c4.post("/start", data={"username": "x", "model_type": "fft"}); body = r.data.decode()
check(r.status_code == 200 and "at least 2 characters" in body and 'value="x"' in body, "a too-short ID should show an error and keep what was typed")

if problems:
    print("INTRO PAGES TEST FAILED:"); [print("  -", p) for p in problems]; sys.exit(1)
print("Intro pages OK: consent text, full factor list, example comparison, random editable participant ID.")
