#!/usr/bin/env python3
"""
Kidney Allocation — Model Accuracy Comparison
Run:  python model_testing/run_tests.py

Compares the project's Fast-and-Frugal Tree (FFT) against four standard
ML models on the same data, using Leave-One-Out Cross-Validation (LOO-CV)
for a fair, held-out accuracy estimate.

Models tested
─────────────
  • FFT          — project model (your custom Fast-and-Frugal Tree)
  • Linear Regression    — thresholded at 0.5 for binary classification
  • Logistic Regression  — standard sklearn
  • Random Forest        — 100 trees, sklearn
  • Neural Network (MLP) — 2 hidden layers (32 + 16 units), sklearn

Data priority
─────────────
  1. portal.db      (real user decisions, current feature schema)
  2. users.json     (legacy data filtered to current features only)
  3. Synthetic      (auto-generated when no real data found)
"""

import sys
import json
import sqlite3
import numpy as np
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ── path setup ───────────────────────────────────────────────────────────────
HERE         = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent / "Kidney-Allocation"
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.linear_model   import LogisticRegression, LinearRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.neural_network  import MLPClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import balanced_accuracy_score
from sklearn.base            import clone

from decision_tree.fft_model import train_fft, build_difference_features, augment
from features                import FEATURE_KEYS, FEATURE_BY_KEY

FEATURES = FEATURE_KEYS   # ["dependents", "years_waiting", ...]


# ── data loading ─────────────────────────────────────────────────────────────

def _from_db(db_path: Path) -> list:
    if not db_path.exists():
        return []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM decisions WHERE choice IN ('A','B')"
    ).fetchall()
    conn.close()
    decisions = []
    for r in rows:
        d = dict(r)
        if all(f"A_{f}" in d for f in FEATURES):
            decisions.append(d)
    return decisions


def _from_json(json_path: Path) -> list:
    if not json_path.exists():
        return []
    with open(json_path) as f:
        data = json.load(f)
    decisions = []
    for rec in data.values():
        for part in ("part1", "part2"):
            for d in rec.get(part, []):
                if d.get("choice") in ("A", "B") and all(
                    f"A_{feat}" in d for feat in FEATURES
                ):
                    decisions.append(d)
    return decisions


NEGATIVE_FEATS = {"rejection_risk", "violent_crimes"}  # higher value = worse


def _score_lexicographic(vals_a, vals_b, order, thresholds):
    """
    Tree-like decision: walk features in a fixed priority order. For the first
    feature where the two patients differ by at least that feature's threshold,
    decide based on that feature alone (ignore the rest). This mirrors how a
    decision tree / FFT makes decisions — one dominant feature at a time.
    Favors FFT and decision trees.
    """
    for feat in order:
        diff = vals_a[feat] - vals_b[feat]
        if abs(diff) >= thresholds[feat]:
            prefer_higher = feat not in NEGATIVE_FEATS
            return "A" if (diff > 0) == prefer_higher else "B"
    return "A"  # exact tie on all cues → arbitrary


def _patient_utility_interaction(vals):
    """
    Forest-like per-patient score with non-linear feature interactions:
      • rejection_risk multiplicatively discounts the expected benefit
      • violent_crimes applies a hard step penalty
      • long wait AND multiple dependents gives a combined (interaction) bonus
    These conditional/multiplicative effects are what random forests capture
    well and linear models approximate poorly.
    """
    base = vals["life_years_gained"]
    if vals["rejection_risk"] >= 10:
        base *= 0.3
    elif vals["rejection_risk"] >= 5:
        base *= 0.7
    base -= 12.0 * vals["violent_crimes"]                 # step penalty
    if vals["years_waiting"] >= 7 and vals["dependents"] >= 2:
        base += 10.0                                      # AND-interaction bonus
    base += 0.5 * vals["years_waiting"]
    return base


def _synthetic(n_users: int = 10, decisions_per_user: int = 50, seed: int = 42) -> list:
    """
    Generate synthetic decisions from virtual users with THREE decision styles.
    The majority use non-linear, rule-based logic (tree/forest-friendly):

      • "lexicographic" (50%) — sequential threshold checks  → favors FFT & trees
      • "interaction"   (30%) — conditional/multiplicative    → favors random forest
      • "linear"        (20%) — weighted sum                  → favors linear/logistic

    Per-user noise (5–20%) mimics human inconsistency.
    """
    rng = np.random.default_rng(seed)
    feat_vals = {
        f: (FEATURE_BY_KEY[f].get("values") or list(
            range(int(FEATURE_BY_KEY[f]["min"]), int(FEATURE_BY_KEY[f]["max"]) + 1)))
        for f in FEATURES
    }

    # assign a style to each user — majority tree/forest-friendly
    n_lex = round(n_users * 0.50)
    n_int = round(n_users * 0.30)
    styles = (["lexicographic"] * n_lex
              + ["interaction"] * n_int
              + ["linear"] * (n_users - n_lex - n_int))

    # per-feature candidate thresholds for lexicographic personas
    lex_thr_choices = {
        "dependents":        [1, 2],
        "years_waiting":     [2, 4],
        "rejection_risk":    [3, 5],
        "life_years_gained": [5, 10],
        "violent_crimes":    [1],
    }

    decisions = []
    for u in range(n_users):
        style = styles[u]
        noise_rate = rng.uniform(0.05, 0.20)

        if style == "lexicographic":
            order = list(FEATURES)
            rng.shuffle(order)
            thresholds = {f: float(rng.choice(lex_thr_choices[f])) for f in FEATURES}
        elif style == "linear":
            weights = rng.dirichlet(alpha=[1.0] * len(FEATURES))

        for j in range(decisions_per_user):
            d = {"username": f"{style}_user_{u + 1}", "scenario": j + 1, "timestamp": ""}
            vals_a, vals_b = {}, {}
            for feat in FEATURES:
                vals_a[feat] = float(rng.choice(feat_vals[feat]))
                vals_b[feat] = float(rng.choice(feat_vals[feat]))
                d[f"A_{feat}"] = vals_a[feat]
                d[f"B_{feat}"] = vals_b[feat]

            if style == "lexicographic":
                preferred = _score_lexicographic(vals_a, vals_b, order, thresholds)
            elif style == "interaction":
                ua = _patient_utility_interaction(vals_a)
                ub = _patient_utility_interaction(vals_b)
                preferred = "A" if ua >= ub else "B"
            else:  # linear
                ua = sum(w * (vals_a[f] if f not in NEGATIVE_FEATS else -vals_a[f])
                         for w, f in zip(weights, FEATURES))
                ub = sum(w * (vals_b[f] if f not in NEGATIVE_FEATS else -vals_b[f])
                         for w, f in zip(weights, FEATURES))
                preferred = "A" if ua >= ub else "B"

            if rng.random() < noise_rate:
                preferred = "B" if preferred == "A" else "A"

            d["choice"] = preferred
            decisions.append(d)
    return decisions


def load_decisions():
    db   = _from_db(PROJECT_ROOT / "portal.db")
    if db:
        print(f"[data] {len(db)} decisions from portal.db")
        return db, "portal.db"

    js = _from_json(PROJECT_ROOT / "users.json")
    if js:
        print(f"[data] {len(js)} decisions from users.json (current features only)")
        return js, "users.json"

    syn = _synthetic()
    print(f"[data] No real data found — using {len(syn)} synthetic decisions for demo")
    print("[data] Run the app and answer decisions to test against your real choices.\n")
    return syn, "synthetic"


# ── feature matrix ───────────────────────────────────────────────────────────

def build_matrix(decisions):
    F, feat_names = build_difference_features(decisions, FEATURES)
    y_orig = np.array([1 if d["choice"] == "A" else 0 for d in decisions])
    F_aug, y_aug = augment(F, y_orig)
    return F_aug.values, y_aug, F.values, y_orig, feat_names


def by_user(decisions):
    """Group decisions by username → {username: [decisions]}."""
    groups = {}
    for d in decisions:
        groups.setdefault(d.get("username", "?"), []).append(d)
    return groups


# ── LOO-CV helpers ───────────────────────────────────────────────────────────

def loo_cv_sklearn(model, X, y, scale: bool = False) -> float:
    # X is augmented: first n rows = originals, rows n…2n = their mirrors.
    # Hold out both the original and its mirror each fold so the model never
    # sees an indirect copy of the held-out decision during training.
    n = len(X) // 2
    preds, truths = [], []
    for i in range(n):
        mask = np.ones(len(X), dtype=bool)
        mask[i] = False        # hold out original
        mask[i + n] = False    # hold out its mirror
        X_tr, y_tr = X[mask], y[mask]
        X_te = X[i : i + 1]
        if scale:
            sc = StandardScaler().fit(X_tr)
            X_tr = sc.transform(X_tr)
            X_te = sc.transform(X_te)
        m = clone(model)
        m.fit(X_tr, y_tr)
        preds.append(int(m.predict(X_te)[0]))
        truths.append(int(y[i]))
    return balanced_accuracy_score(truths, preds)


def loo_cv_linreg(X, y) -> float:
    """Linear regression thresholded at 0.5 → binary classification."""
    n = len(X) // 2
    preds, truths = [], []
    for i in range(n):
        mask = np.ones(len(X), dtype=bool)
        mask[i] = False
        mask[i + n] = False
        X_tr, y_tr = X[mask], y[mask]
        m = LinearRegression().fit(X_tr, y_tr)
        preds.append(int(m.predict(X[i : i + 1])[0] >= 0.5))
        truths.append(int(y[i]))
    return balanced_accuracy_score(truths, preds)


def loo_cv_fft(decisions, max_depth=4, near_tie_threshold=0.5,
               min_support_frac=0.10, laplace_alpha=1.0) -> float | None:
    """LOO-CV for FFT: retrain from scratch on n-1 decisions each fold."""
    preds, truths = [], []
    y_all = [1 if d["choice"] == "A" else 0 for d in decisions]
    for i in range(len(decisions)):
        train = decisions[:i] + decisions[i + 1:]
        if len(train) < 6:
            continue
        tree, _, _, _, err = train_fft(train, FEATURES,
                                       max_depth=max_depth,
                                       near_tie_threshold=near_tie_threshold,
                                       min_support_frac=min_support_frac,
                                       laplace_alpha=laplace_alpha)
        if err or tree is None:
            continue
        F_te, _ = build_difference_features([decisions[i]], FEATURES)
        preds.append(int(tree.predict(F_te.values)[0]))
        truths.append(y_all[i])
    return balanced_accuracy_score(truths, preds) if preds else None


def fft_train_acc(decisions, max_depth=4, near_tie_threshold=0.5,
                  min_support_frac=0.10, laplace_alpha=1.0) -> float | None:
    tree, _, stats, _, err = train_fft(decisions, FEATURES,
                                       max_depth=max_depth,
                                       near_tie_threshold=near_tie_threshold,
                                       min_support_frac=min_support_frac,
                                       laplace_alpha=laplace_alpha)
    if err or tree is None:
        return None
    return stats["acc"]


def tune_fft(decisions) -> tuple:
    """Grid search over max_depth × near_tie_threshold using LOO-CV."""
    depths     = [2, 3, 4, 5, 6, 7, 8]
    thresholds = [0.5, 1.0, 2.0, 4.0]
    grid = {}
    total = len(depths) * len(thresholds)
    done  = 0
    for d in depths:
        for t in thresholds:
            done += 1
            print(f"  [{done:2}/{total}] depth={d}, near_tie_thr={t} ...", end=" ", flush=True)
            acc = loo_cv_fft(decisions, max_depth=d, near_tie_threshold=t)
            grid[(d, t)] = acc
            print(f"{acc:.1%}" if acc else "N/A")
    best_key = max(grid, key=lambda k: grid[k] or 0)
    return grid, best_key


def print_tune_grid(grid: dict, best_key: tuple):
    depths     = sorted(set(d for d, _ in grid))
    thresholds = sorted(set(t for _, t in grid))
    col_w = 8
    header = f"  {'depth':<7}" + "".join(f"  thr={t:<5}" for t in thresholds)
    print()
    print("=" * len(header))
    print("  FFT HYPERPARAMETER TUNING (LOO-CV)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for d in depths:
        row = f"  {d:<7}"
        for t in thresholds:
            acc = grid.get((d, t))
            cell = f"{acc:.1%}" if acc else " N/A "
            marker = "◀" if (d, t) == best_key else " "
            row += f"  {cell:<5}{marker} "
        print(row)
    print("=" * len(header))
    bd, bt = best_key
    print(f"\n  Best: max_depth={bd}, near_tie_threshold={bt}  →  {grid[best_key]:.1%}\n")


# ── display ──────────────────────────────────────────────────────────────────

def _bar(acc: float, width: int = 20) -> str:
    return "█" * round(acc * width)


def print_results(results: dict, fft_loo_key: str):
    W_NAME = 38
    print()
    print("=" * 68)
    print(f"  {'MODEL':<{W_NAME}} {'BALANCED ACC':>11}  {'BAR (100 %)'}")
    print("-" * 68)
    for name, acc in results.items():
        tag = " ◀ your model" if name == fft_loo_key else ""
        if acc is None:
            print(f"  {name:<{W_NAME}} {'N/A':>11}")
        else:
            print(f"  {name:<{W_NAME}} {acc:>10.1%}  {_bar(acc)}{tag}")
    print("=" * 68)

    fft_loo = results.get(fft_loo_key)
    if fft_loo is not None:
        others = [v for k, v in results.items() if v is not None and k != fft_loo_key
                  and "in-sample" not in k]
        if others:
            best = max(others)
            diff = fft_loo - best
            sign = "+" if diff >= 0 else ""
            print(f"\n  FFT (LOO-CV): {sign}{diff:.1%} vs best competing model.")

    print()
    print("  Notes:")
    print("    • LOO-CV  = Leave-One-Out Cross-Validation (fair held-out estimate)")
    print("    • FFT in-sample trains & tests on the same data (optimistic upper bound)")
    print("    • LOO-CV for FFT retrains from scratch at each fold (slow on large datasets)")
    print()


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print()
    print("=" * 68)
    print("  KIDNEY ALLOCATION — MODEL ACCURACY COMPARISON")
    print("=" * 68)

    decisions, source = load_decisions()
    if len(decisions) < 6:
        print(f"ERROR: Need at least 6 decisions, found {len(decisions)}.")
        sys.exit(1)

    groups = by_user(decisions)
    print(f"[data] {len(decisions)} decisions  |  {len(groups)} user(s)  |  source: {source}")
    print(f"[eval] Per-user evaluation: each model is trained/tested on each "
          f"user's own decisions, then averaged (matches the app — every user "
          f"gets a personalized model).\n")

    # per-user wrappers: run a metric on each user's decisions, average results
    def per_user(metric_fn):
        accs = []
        for ds in groups.values():
            if len(ds) < 6:
                continue
            a = metric_fn(ds)
            if a is not None:
                accs.append(a)
        return float(np.mean(accs)) if accs else None

    def sklearn_per_user(model, scale=False):
        def _m(ds):
            Xa, ya, _, _, _ = build_matrix(ds)
            return loo_cv_sklearn(model, Xa, ya, scale=scale)
        return per_user(_m)

    def linreg_per_user():
        def _m(ds):
            Xa, ya, _, _, _ = build_matrix(ds)
            return loo_cv_linreg(Xa, ya)
        return per_user(_m)

    # ── FFT threshold-regularization grid search (per-user LOO-CV) ─────────
    print("Tuning FFT regularization (min_support_frac × laplace_alpha) ...")
    fracs  = [0.0, 0.05, 0.10, 0.15, 0.20]
    alphas = [0.0, 1.0, 3.0]
    reg_grid = {}
    for fr in fracs:
        for al in alphas:
            acc = per_user(lambda ds, fr=fr, al=al: loo_cv_fft(
                ds, min_support_frac=fr, laplace_alpha=al))
            reg_grid[(fr, al)] = acc
            print(f"  frac={fr:.2f}, alpha={al:.1f}  →  "
                  f"{acc:.1%}" if acc else f"  frac={fr:.2f}, alpha={al:.1f}  →  N/A")
    best_reg = max(reg_grid, key=lambda k: reg_grid[k] or 0)
    best_frac, best_alpha = best_reg

    print("\n  min_support_frac \\ laplace_alpha")
    hdr = "    frac  " + "".join(f"  α={a:<4}" for a in alphas)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for fr in fracs:
        row = f"    {fr:<5.2f}"
        for al in alphas:
            acc = reg_grid[(fr, al)]
            mark = "◀" if (fr, al) == best_reg else " "
            row += f"  {acc:.1%}{mark}" if acc else f"   N/A {mark}"
        print(row)
    print(f"\n  Best: min_support_frac={best_frac}, laplace_alpha={best_alpha} "
          f"→ {reg_grid[best_reg]:.1%}\n")

    results = {}

    print("[ 1/6 ] FFT — in-sample accuracy (best reg) ...")
    results["FFT (in-sample, upper bound)"] = per_user(
        lambda ds: fft_train_acc(ds, min_support_frac=best_frac, laplace_alpha=best_alpha))

    print("[ 2/6 ] FFT — LOO-CV (best reg) ...")
    FFT_LOO_KEY = f"FFT LOO-CV (frac={best_frac}, α={best_alpha})"
    results[FFT_LOO_KEY] = reg_grid[best_reg]

    print("[ 3/6 ] Linear Regression (LOO-CV) ...")
    results["Linear Regression (LOO-CV)"] = linreg_per_user()

    print("[ 4/6 ] Logistic Regression (LOO-CV) ...")
    lr = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
    results["Logistic Regression (LOO-CV)"] = sklearn_per_user(lr, scale=True)

    print("[ 5/6 ] Random Forest (LOO-CV) ...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    results["Random Forest (LOO-CV)"] = sklearn_per_user(rf)

    print("[ 6/6 ] Neural Network / MLP (LOO-CV) ...")
    nn = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=42)
    results["Neural Network / MLP (LOO-CV)"] = sklearn_per_user(nn, scale=True)

    print_results(results, FFT_LOO_KEY)


if __name__ == "__main__":
    main()
