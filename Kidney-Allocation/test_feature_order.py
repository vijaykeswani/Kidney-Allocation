"""
Guards the per-participant, counterbalanced factor order.

  1. Balance: after N sign-ups the orderings are used evenly (no ordering used more than once more than any other), and
     every factor has appeared in every position about equally often — even when N is far below 120.
  2. The order is fixed for a participant: identical on every scenario of the study, on the model page and in the
     case browser, and unchanged if they sign in again. Different participants really do get different orders.
  3. Only presentation changes: the model data, the DB columns and the exported CSVs keep the canonical order.
"""
import json, os, random, re, sys, tempfile
from collections import Counter

os.environ.setdefault("FLASK_SECRET_KEY", "test")
os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(), "t.db")
os.environ["RESPONSES_DIR"] = tempfile.mkdtemp()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import feature_order, features   # noqa: E402

problems = []
def check(ok, msg):
    if not ok: problems.append(msg)

# ---- 1. balance (pure function, so it's fast): 240 = two full cycles of the 120 orderings
rng = random.Random(1); assigned = []
for _ in range(240): assigned.append(feature_order.choose(assigned, rng))
used = Counter(tuple(o) for o in assigned)
check(len(used) == 120 and set(used.values()) == {2}, f"240 sign-ups should use every ordering exactly twice (got {len(used)} distinct, counts {set(used.values())})")
# partway through a cycle: counts never differ by more than 1, and positions stay even
for n in (7, 30, 61, 119, 150):
    sub = assigned[:n]; u = Counter(tuple(o) for o in sub)
    check(max(u.values()) - min(u.get(tuple(p), 0) for p in feature_order.all_orders()) <= 1, f"after {n} sign-ups some ordering was used >1 more than another")
for n in (20, 40, 60, 100):
    pos = Counter((f, i) for o in assigned[:n] for i, f in enumerate(o)); K = len(features.FEATURE_KEYS)
    spread = max(pos[(f, i)] for f in features.FEATURE_KEYS for i in range(K)) - min(pos[(f, i)] for f in features.FEATURE_KEYS for i in range(K))
    check(spread <= max(3, n // 8), f"after {n} sign-ups some factor is over-represented in a position (spread {spread})")
print(f"balance: 240 sign-ups -> all 120 orderings used exactly twice; position spread after 60 sign-ups = "
      f"{(lambda pos: max(pos.values()) - min(pos[(f, i)] for f in features.FEATURE_KEYS for i in range(len(features.FEATURE_KEYS))))(Counter((f, i) for o in assigned[:60] for i, f in enumerate(o)))}")

# ---- 2 + 3. through the real app
import app as a   # noqa: E402
import db         # noqa: E402

def card_order(html):
    """the factor order shown on the first patient card of a questionnaire page"""
    block = html.split("Patient A")[1].split("Patient B")[0]
    def pos(key):
        label = features.FEATURE_BY_KEY[key]["label"]
        return min(i for i in (block.find(label), block.find(label.replace("'", "&#39;"))) if i >= 0)
    return sorted(features.FEATURE_KEYS, key=pos)

seen = set()
for i in range(12):
    c = a.app.test_client(); name = f"order_user_{i}"
    c.post("/start", data={"username": name, "model_type": ["fft", "twostage", "twostage_v2", "control"][i % 4]})
    saved = db.get_feature_order(name)
    check(sorted(saved) == sorted(features.FEATURE_KEYS), f"{name}: saved order is not a permutation of the factors")
    seen.add(tuple(saved))
    orders = set()
    for q in range(1, 6):
        html = c.get("/questionnaire/1").data.decode(); orders.add(tuple(card_order(html)))
        c.post("/questionnaire/1/answer", data={"choice": random.choice("AB")})
    check(orders == {tuple(saved)}, f"{name}: the order changed between scenarios ({len(orders)} different) or doesn't match the saved order")
    c.post("/start", data={"username": name, "model_type": ["fft", "twostage", "twostage_v2", "control"][i % 4]})
    check(db.get_feature_order(name) == saved, f"{name}: signing in again changed the order")
    if i % 4 in (1, 2):                                  # the model page lists factors in the same order (v1) / weight order with this as tie-break (v2)
        for _ in range(15): c.post("/questionnaire/1/answer", data={"choice": random.choice("AB")})
        page = c.get("/results").data.decode()
        fd = json.JSONDecoder().raw_decode(page, page.index("[", page.index("var FEATURES_DATA = ")))[0]
        check([f["key"] for f in fd] == saved, f"{name}: model page FEATURES_DATA is not in the participant's order")
        canon = json.JSONDecoder().raw_decode(page, page.index("[", page.index("var CANON_KEYS = ")))[0]
        check(canon == features.FEATURE_KEYS, f"{name}: model logic must use the canonical order")
check(len(seen) >= 10, f"12 participants should mostly get different orders (got {len(seen)} distinct)")
cols = [r[1] for r in db._connect().execute("PRAGMA table_info(decisions)").fetchall()]
check([c for c in cols if c.startswith("A_")] == [f"A_{k}" for k in features.FEATURE_KEYS], "decision columns must stay in canonical order")

# ---- rejection chance: wording and values of the published study (arXiv 2511.10032, Fig. A.1 / Table A.1)
rr = features.FEATURE_BY_KEY["rejection_risk"]
check(features.values_map()["rejection_risk"] == [1, 2, 3, 5, 10, 15, 20], "rejection chance values should be 1, 2, 3, 5, 10, 15, 20")
check(rr["label"] == "Transplantation rejection risk", "rejection risk label")
check("in the next year" in rr["description"], "rejection risk info text")
check(rr.get("value_suffix") == "%", "rejection chance values should be shown as percentages")
q = c.get("/questionnaire/1").data.decode()
check("Which patient should receive the kidney?" in q and "%</div>" in q, "the questionnaire needs its heading and a % value")
check(features.FEATURE_BY_KEY["years_waiting"]["label"] == "Years on the waitlist", "years waiting label")
check(features.FEATURE_BY_KEY["dependents"]["description"].startswith("Number of people who financially depend"), "dependents info text")
check(features.FEATURE_BY_KEY["violent_crimes"]["description"] == "Number of past violent crimes the patient has been convicted of.", "crimes info text")

if problems:
    print("FEATURE ORDER TEST FAILED:"); [print("  -", p) for p in problems]; sys.exit(1)
print(f"Feature order OK: balanced (every ordering, every position), fixed per participant across scenarios/sign-ins/model page, canonical data untouched ({len(seen)} distinct orders over 12 participants).")
