"""
Palette guard. The study colour-codes Patient A teal and Patient B orange everywhere a participant looks at
a MODEL (feature-list pages, tree page, tree final diagram), and uses neutral colours while they are still
stating preferences. These colours have silently reverted to red/blue several times, so this fails loudly.

Run:  python test_palette.py
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
A, A_SOFT, B, B_SOFT = "#0f766e", "#d8ecea", "#b1622f", "#f6e7dd"
FORBIDDEN = {"#b91c1c", "#be123c", "#1d4ed8", "#2563eb"}   # the old red/blue patient colours
failures = []


def read(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return f.read()


def expect(cond, msg):
    if not cond:
        failures.append(msg)


# 1. Token definitions
base = read("templates/base.html")
for token, val in (("--a", A), ("--a-soft", A_SOFT), ("--b", B), ("--b-soft", B_SOFT)):
    expect(re.search(rf"{re.escape(token)}\s*:\s*{val}\b", base, re.I), f"base.html: {token} should be {val}")
expect(re.search(r"\.patient-card\.a\s*,\s*\.patient-card\.b\s*\{[^}]*--neutral-soft", base),
       "base.html: elicitation patient cards must use the same neutral colour for A and B")

model = read("decision_tree/templates/model.html")
for token, val in (("--patient-a", A), ("--patient-a-soft", A_SOFT), ("--patient-b", B), ("--patient-b-soft", B_SOFT)):
    expect(re.search(rf"{re.escape(token)}\s*:\s*{val}\b", model, re.I), f"model.html: {token} should be {val}")

for rel in ("feature_list/templates/twostage.html", "feature_list/templates/twostage_v2.html"):
    t = read(rel)
    for token, val in (("--patient-a", A), ("--patient-b", B)):
        expect(re.search(rf"{re.escape(token)}\s*:\s*{val}\b", t, re.I), f"{rel}: {token} should be pinned to {val}")

comp = read("decision_tree/fft_component.py")
expect(re.search(rf'"a":\s*"{A}"', comp, re.I) and re.search(rf'"b":\s*"{B}"', comp, re.I),
       f"fft_component.py: DEFAULT_FFT_PALETTE a/b should be {A}/{B}")

# 2. No old patient colours defined anywhere
pat = re.compile(r"(--(?:patient-)?[ab]\s*:\s*|\"[ab]\":\s*\")(#[0-9a-fA-F]{6})")
for dirpath, _, files in os.walk(ROOT):
    if any(skip in dirpath for skip in (".git", "node_modules", "export")):
        continue
    for name in files:
        if name.endswith((".html", ".py", ".js", ".css")) and name != "test_palette.py":
            rel = os.path.relpath(os.path.join(dirpath, name), ROOT)
            for m in pat.finditer(read(rel)):
                if m.group(2).lower() in FORBIDDEN:
                    failures.append(f"{rel}: patient colour defined as {m.group(2)} (old red/blue)")

# 3. What is actually SERVED
os.environ.setdefault("FLASK_SECRET_KEY", "palette-test")
os.environ["DB_PATH"] = os.path.join(ROOT, ".palette_test.db")
if os.path.exists(os.environ["DB_PATH"]):
    os.remove(os.environ["DB_PATH"])
sys.path.insert(0, ROOT)
try:
    import app as app_module
    c = app_module.app.test_client()
    c.post("/start", data={"username": "palette", "model_type": "fft"})
    html = c.get("/questionnaire/1").data.decode()
    expect(f"--a:{A}" in html.replace(" ", "") and f"--b:{B}" in html.replace(" ", ""), "served questionnaire page: --a/--b are not teal/orange")
finally:
    if os.path.exists(os.environ["DB_PATH"]):
        os.remove(os.environ["DB_PATH"])

if failures:
    print("PALETTE CHECK FAILED:")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("Palette OK: teal/orange wherever a model is shown, neutral while stating preferences, no red/blue patient colours anywhere.")
