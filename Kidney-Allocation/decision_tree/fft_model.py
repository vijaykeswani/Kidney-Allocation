"""
Fast-and-Frugal Tree (FFT) model for the Preference Elicitation Portal.
SURA 2026 · IIT Delhi

This module is the *only* predictive model used by the application. It is kept
deliberately self-contained (pure numpy / pandas / scikit-learn metric) so it
can be retrained, unit-tested and extended independently of the Streamlit UI.

Design
------
* Features are ONLY the per-parameter differences between the two candidates:
      diff(param) = value(param, A) − value(param, B)
  Nothing else is engineered. Direction ("is higher better?") is learned by the
  tree itself through the split operator, so no hand-tuned transforms are needed.

* A Fast-and-Frugal Tree is an ordered checklist of single-cue tests. At every
  level exactly one cue (one difference feature) is tested; the TRUE branch is an
  immediate exit (a leaf that decides A or B) and the FALSE branch falls through
  to the next cue. After the last cue, a single default leaf catches everything
  that fell through. For k cues there are k+1 exits.

      node i :  if  (feature_i  op_i  threshold_i)   ->  exit to exit_class_i
                else                                  ->  continue to node i+1
      fall-through after last node                    ->  default_class

  This structure is maximally interpretable and trivially editable: every node is
  one feature, one operator, one threshold and one outcome.

Public API
----------
    build_difference_features(decisions, params) -> (DataFrame, feature_names)
    augment(F, y)                                -> (DataFrame, y)
    FastFrugalTree                               -> .fit / .predict / .predict_proba
                                                    .to_dict / .from_dict / .evaluate
    train_fft(decisions_json, params, override_json=None)
                                                 -> (tree, nodes_df, stats, feat_names, error)
    feature_row(...) / decision_prob(...) / explain_prediction(...)
"""

import json
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

import features

# Any node whose split threshold has a smaller absolute magnitude than this is
# treated as a "near-tie" cue — the two candidates are barely distinguishable
# on that dimension, so a single coarse cutoff is a weak way to decide. Every
# such node automatically gets one extra tie-breaker node appended after it
# (see FastFrugalTree.attach_near_tie_refinements / _best_refine_split below).
NEAR_TIE_ABS_THRESHOLD = 1.0


# ════════════════════════════════════════════════════════════════════════════
# FEATURES  —  only pairwise differences, nothing else
# ════════════════════════════════════════════════════════════════════════════

def build_difference_features(decisions, params):
    """
    Build the difference feature matrix.

    For every decision and every parameter p:
        <p>_diff = A_<p> - B_<p>

    Returns (F, feature_names) where F is a DataFrame with one column per param.
    These features are inherently antisymmetric: swapping A and B negates them.
    """
    a = np.array([[float(d.get(f"A_{p}", 0.0)) for p in params] for d in decisions],
                 dtype=float)
    b = np.array([[float(d.get(f"B_{p}", 0.0)) for p in params] for d in decisions],
                 dtype=float)
    if a.size == 0:
        cols = [f"{p}_diff" for p in params]
        return pd.DataFrame(columns=cols), cols
    diff = a - b
    cols = [f"{p}_diff" for p in params]
    return pd.DataFrame(diff, columns=cols), cols


def augment(F, y):
    """
    Double the dataset by swapping A and B. Because every feature is a difference
    (antisymmetric), the swap simply negates the whole matrix and flips the label.
    This forces the learned tree to be (near-)symmetric.
    """
    y = np.asarray(y).astype(int)
    F_swap = -F
    return (
        pd.concat([F, F_swap], ignore_index=True),
        np.concatenate([y, 1 - y]),
    )


# ════════════════════════════════════════════════════════════════════════════
# FAST-AND-FRUGAL TREE
# ════════════════════════════════════════════════════════════════════════════

class FastFrugalTree:
    """
    A depth-limited Fast-and-Frugal Tree over difference features.

    Each node is a dict:
        {feature, feature_idx, op ('>=' | '<='), threshold,
         exit_class (1=A, 0=B), support, purity}

    Class 1 == "prefer candidate A", class 0 == "prefer candidate B".
    """

    def __init__(self, max_depth=4, near_tie_threshold=NEAR_TIE_ABS_THRESHOLD):
        self.max_depth = int(max_depth)
        self.near_tie_threshold = float(near_tie_threshold)
        self.nodes = []
        self.default_class = 0
        self.default_support = 0.0
        self.default_purity = 0.5
        self.feature_names = []

    # ── threshold candidates ────────────────────────────────────────────────
    @staticmethod
    def _candidate_thresholds(x, cap=60):
        u = np.unique(x)
        if len(u) <= 1:
            return np.array([float(u[0]) if len(u) else 0.0])
        # Round midpoints to integers — all features are integer-valued
        mids = np.ceil((u[:-1] + u[1:]) / 2.0)
        cands = np.unique(np.concatenate([mids, [0.0]]))
        if len(cands) > cap:
            idx = np.linspace(0, len(cands) - 1, cap).astype(int)
            cands = cands[idx]
        return cands

    # ── fit ─────────────────────────────────────────────────────────────────
    def fit(self, F, y, feature_names=None):
        X = np.asarray(F, dtype=float)
        y = np.asarray(y).astype(int)
        n_total = len(y)
        if X.ndim != 2 or n_total == 0:
            return self

        self.feature_names = (list(feature_names) if feature_names is not None
                              else [f"f{i}" for i in range(X.shape[1])])

        remaining = np.ones(n_total, dtype=bool)
        used = set()
        self.nodes = []

        for _level in range(self.max_depth):
            idx = np.where(remaining)[0]
            if len(idx) == 0:
                break
            yr = y[idx]
            if len(np.unique(yr)) == 1:
                break  # already pure — the default leaf will cover it

            best = None  # (score, purity, n_exit, fidx, thr, prefer_higher, exit_class, mask)
            for fidx in range(X.shape[1]):
                if fidx in used:
                    continue
                xr     = X[idx, fidx]
                xr_abs = np.abs(xr)
                for thr in self._candidate_thresholds(xr_abs):
                    if thr <= 0:
                        continue  # skip zero — trivial split on abs values
                    M = xr_abs >= thr
                    n_exit = int(M.sum())
                    if n_exit == 0 or n_exit == len(idx):
                        continue
                    yexit = yr[M]
                    signs = np.sign(xr[M])  # sign of signed diff for exiting rows
                    # majority vote: prefer_higher = True if users preferred the
                    # patient with the higher raw value on this feature
                    votes_higher = int(np.sum((signs > 0) & (yexit == 1))
                                       + np.sum((signs < 0) & (yexit == 0)))
                    votes_lower  = int(np.sum((signs > 0) & (yexit == 0))
                                       + np.sum((signs < 0) & (yexit == 1)))
                    prefer_higher = votes_higher >= votes_lower
                    # predicted exit_class = what prefer_higher means per-row
                    pred_exit = np.where((signs > 0) == prefer_higher, 1, 0)
                    exit_class = int(round(yexit.mean()))  # kept for stats/compat
                    n_correct = int((yexit == pred_exit).sum())
                    score = n_correct - (n_exit - n_correct)
                    purity = n_correct / n_exit
                    key = (score, purity, n_exit)
                    if best is None or key > (best[0], best[1], best[2]):
                        best = (score, purity, n_exit, fidx,
                                float(round(thr)), prefer_higher, exit_class, M)
            if best is None:
                break

            score, purity, n_exit, fidx, thr, prefer_higher, exit_class, M = best
            global_exit = idx[M]
            self.nodes.append({
                "feature":       self.feature_names[fidx],
                "feature_idx":   fidx,
                "op":            ">=",
                "threshold":     thr,
                "use_abs":       True,
                "prefer_higher": prefer_higher,
                "exit_class":    exit_class,
                "support":       len(global_exit) / n_total,
                "purity":        purity,
            })
            used.add(fidx)
            remaining[global_exit] = False

            idx2 = np.where(remaining)[0]
            if len(idx2) == 0 or len(np.unique(y[idx2])) == 1:
                break

        # default leaf for whatever fell through
        idx = np.where(remaining)[0]
        if len(idx) > 0:
            yr = y[idx]
            self.default_class = int(round(yr.mean()))
            self.default_support = len(idx) / n_total
            self.default_purity = float((yr == self.default_class).mean())
        elif self.nodes:
            last = self.nodes[-1]
            self.default_class = 1 - (0 if last.get("prefer_higher") else 1) if last.get("use_abs") else 1 - last["exit_class"]
            self.default_support = 0.0
            self.default_purity = 0.5
        return self

    # ── near-tie tie-breaker nodes ──────────────────────────────────────────
    def attach_near_tie_refinements(self, F, y):
        """
        Walk every node in order. For any node whose split is a "near-tie" cue
        (abs(threshold) < self.near_tie_threshold — i.e. candidates are barely
        distinguishable on that dimension) attach one extra child node ("refine")
        that only fires on the rows which satisfied that node's condition. This
        gives close calls a real second check instead of a single coarse cutoff.

        A node's `refine` sub-node is a compact single-split test, evaluated
        strictly after the parent condition is true:
            refine.feature  refine.op  refine.threshold  -> true_class
            else                                          -> false_class

        Any node that no longer qualifies (near-tie condition broken because a
        threshold was edited) has its stale refine dropped, so this is safe to
        call repeatedly (e.g. after every retrain or user edit).

        Nodes carrying a refine with `"manual": True` — added on purpose via
        the editor's "add a node to the right" control, rather than detected
        automatically — are left untouched here entirely, so a manual addition
        is never silently removed just because its parent threshold isn't
        (or is no longer) a near-tie value.
        """
        X = np.asarray(F, dtype=float)
        y = np.asarray(y).astype(int)
        n_total = max(1, len(y))
        reached = np.ones(len(y), dtype=bool)

        for node in self.nodes:
            xs = X[:, node["feature_idx"]]
            if node.get("use_abs"):
                cond = np.abs(xs) >= node["threshold"]
            else:
                cond = (xs >= node["threshold"]) if node["op"] == ">=" else (xs <= node["threshold"])
            exit_here = reached & cond

            if node.get("refine", {}).get("manual"):
                reached = reached & ~cond
                continue  # user-added — leave exactly as configured

            node.pop("refine", None)
            is_near_tie = abs(node["threshold"]) < self.near_tie_threshold
            if is_near_tie and int(exit_here.sum()) >= 4:
                refine = self._best_refine_split(X, y, exit_here, node["feature_idx"], n_total)
                if refine is not None:
                    node["refine"] = refine

            reached = reached & ~cond
        return self

    def _best_refine_split(self, X, y, mask, exclude_idx, n_total):
        """
        Among the rows that satisfied a near-tie parent condition, find the
        single other cue that best separates them into A vs B.

        Uses the same abs-value split logic as the main fit() so the resulting
        node has the identical data structure (use_abs=True, prefer_higher,
        threshold ≥ 0) and can be displayed with the same |Δ| ≥ X format.

        Returns a refine-node dict, or None if no cue distinguishes anything.
        """
        idx = np.where(mask)[0]
        yr = y[idx]
        if len(idx) < 4 or len(np.unique(yr)) < 2:
            return None

        best = None  # (score, purity, n_exit, dict)
        for fidx in range(X.shape[1]):
            if fidx == exclude_idx:
                continue
            xr     = X[idx, fidx]
            xr_abs = np.abs(xr)
            for thr in self._candidate_thresholds(xr_abs):
                if thr <= 0:
                    continue  # skip zero — same rule as main fit()
                M = xr_abs >= thr
                n_t = int(M.sum())
                n_f = len(idx) - n_t
                if n_t == 0 or n_f == 0:
                    continue
                yr_t  = yr[M]
                signs = np.sign(xr[M])
                votes_higher = int(np.sum((signs > 0) & (yr_t == 1))
                                   + np.sum((signs < 0) & (yr_t == 0)))
                votes_lower  = int(np.sum((signs > 0) & (yr_t == 0))
                                   + np.sum((signs < 0) & (yr_t == 1)))
                prefer_higher = votes_higher >= votes_lower
                pred_exit = np.where((signs > 0) == prefer_higher, 1, 0)
                n_correct = int((yr_t == pred_exit).sum())
                # false branch: majority class among non-exiting rows
                yr_f = yr[~M]
                false_cls = int(round(yr_f.mean())) if n_f > 0 else (1 if prefer_higher else 0)
                purity_t = n_correct / n_t
                purity_f = float((yr_f == false_cls).mean()) if n_f > 0 else 0.5
                score = n_correct - (n_t - n_correct)
                key = (score, purity_t, n_t)
                if best is None or key > (best[0], best[1], best[2]):
                    # true_class: kept for backward compat; derived from prefer_higher
                    true_cls = 1 if prefer_higher else 0
                    best = (score, purity_t, n_t, {
                        "feature":       self.feature_names[fidx],
                        "feature_idx":   fidx,
                        "op":            ">=",
                        "threshold":     float(round(thr)),
                        "use_abs":       True,
                        "prefer_higher": prefer_higher,
                        "true_class":    true_cls,
                        "false_class":   false_cls,
                        "support":       n_t / n_total,
                        "false_support": n_f / n_total,
                        "purity":        purity_t,
                        "false_purity":  purity_f,
                    })
        return best[3] if best else None

    # ── single-row evaluation ───────────────────────────────────────────────
    def _row_predict(self, row):
        """
        Return (predicted_class, exit_node_index, purity, refine_branch).
        exit_node_index -1 = fell through to the default leaf.
        refine_branch is None (no tie-breaker involved), True, or False —
        which side of a near-tie node's tie-breaker fired, when one exists.
        """
        for i, node in enumerate(self.nodes):
            x = row[node["feature_idx"]]
            if node.get("use_abs"):
                cond = abs(x) >= node["threshold"]
                if cond:
                    refine = node.get("refine")
                    if refine:
                        rx = row[refine["feature_idx"]]
                        if refine.get("use_abs"):
                            rcond = abs(rx) >= refine["threshold"]
                            if rcond:
                                ph = refine["prefer_higher"]
                                cls = 1 if (rx > 0) == ph else 0
                            else:
                                cls = int(refine["false_class"])
                            purity = refine["purity"] if rcond else refine["false_purity"]
                        else:
                            # legacy signed refine — kept for backward compat
                            rcond = ((rx >= refine["threshold"]) if refine["op"] == ">="
                                     else (rx <= refine["threshold"]))
                            cls = refine["true_class"] if rcond else refine["false_class"]
                            purity = refine["purity"] if rcond else refine["false_purity"]
                        return cls, i, purity, bool(rcond)
                    # dynamic exit: prefer the patient with higher/lower raw value
                    ph = node["prefer_higher"]
                    cls = 1 if (x > 0) == ph else 0
                    return cls, i, node["purity"], None
            else:
                cond = (x >= node["threshold"]) if node["op"] == ">=" else (x <= node["threshold"])
                if cond:
                    refine = node.get("refine")
                    if refine:
                        rx = row[refine["feature_idx"]]
                        if refine.get("use_abs"):
                            rcond = abs(rx) >= refine["threshold"]
                            if rcond:
                                ph = refine["prefer_higher"]
                                cls = 1 if (rx > 0) == ph else 0
                            else:
                                cls = int(refine["false_class"])
                            purity = refine["purity"] if rcond else refine["false_purity"]
                        else:
                            rcond = ((rx >= refine["threshold"]) if refine["op"] == ">="
                                     else (rx <= refine["threshold"]))
                            cls = refine["true_class"] if rcond else refine["false_class"]
                            purity = refine["purity"] if rcond else refine["false_purity"]
                        return cls, i, purity, bool(rcond)
                    return node["exit_class"], i, node["purity"], None
        return self.default_class, -1, self.default_purity, None

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.array([self._row_predict(r)[0] for r in X])

    def predict_proba(self, X):
        """Confidence derived from the purity of the exit each row lands in."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        out = []
        for r in X:
            cls, _i, purity, _rb = self._row_predict(r)
            conf = min(max(float(purity), 0.5), 1.0)
            p1 = conf if cls == 1 else 1.0 - conf
            out.append([1.0 - p1, p1])
        return np.array(out)

    def exit_index(self, row):
        """Index of the node a single row exits at (-1 = default leaf)."""
        return self._row_predict(np.asarray(row, dtype=float))[1]

    def evaluate(self, F, y):
        preds = self.predict(np.asarray(F, dtype=float))
        return float(balanced_accuracy_score(np.asarray(y).astype(int), preds))

    def recompute_stats(self, F, y):
        """
        Recompute per-node support/purity and the default leaf stats against data.
        Used after a user edits an existing tree so the displayed reliability and
        coverage reflect the edited thresholds rather than the original ones.
        """
        X = np.asarray(F, dtype=float)
        y = np.asarray(y).astype(int)
        n_total = max(1, len(y))
        reached = np.ones(len(y), dtype=bool)
        for node in self.nodes:
            xs = X[:, node["feature_idx"]]
            if node.get("use_abs"):
                cond = np.abs(xs) >= node["threshold"]
            else:
                cond = (xs >= node["threshold"]) if node["op"] == ">=" else (xs <= node["threshold"])
            exit_here = reached & cond
            n_exit = int(exit_here.sum())
            node["support"] = n_exit / n_total
            if n_exit > 0:
                if node.get("use_abs"):
                    ph = node["prefer_higher"]
                    pred = np.where((xs[exit_here] > 0) == ph, 1, 0)
                    node["purity"] = float((y[exit_here] == pred).mean())
                else:
                    node["purity"] = float((y[exit_here] == node["exit_class"]).mean())
            else:
                node["purity"] = 0.5

            refine = node.get("refine")
            if refine:
                ridx = np.where(exit_here)[0]
                if len(ridx) > 0:
                    rx = X[ridx, refine["feature_idx"]]
                    ry = y[ridx]
                    if refine.get("use_abs"):
                        rcond = np.abs(rx) >= refine["threshold"]
                        if rcond.sum() > 0:
                            ph = refine["prefer_higher"]
                            pred_t = np.where((rx[rcond] > 0) == ph, 1, 0)
                            refine["purity"] = float((ry[rcond] == pred_t).mean())
                        else:
                            refine["purity"] = 0.5
                        refine["false_purity"] = (
                            float((ry[~rcond] == refine["false_class"]).mean())
                            if (~rcond).sum() > 0 else 0.5)
                    else:
                        rcond = ((rx >= refine["threshold"]) if refine["op"] == ">="
                                 else (rx <= refine["threshold"]))
                        n_t_r, n_f_r = int(rcond.sum()), int((~rcond).sum())
                        refine["purity"] = (float((ry[rcond] == refine["true_class"]).mean())
                                             if n_t_r > 0 else 0.5)
                        refine["false_purity"] = (float((ry[~rcond] == refine["false_class"]).mean())
                                                   if n_f_r > 0 else 0.5)
                    n_t, n_f = int(rcond.sum()), int((~rcond).sum())
                    refine["support"] = n_t / n_total
                    refine["false_support"] = n_f / n_total
                else:
                    refine["support"] = refine["false_support"] = 0.0
                    refine["purity"] = refine["false_purity"] = 0.5

            reached = reached & ~cond
        n_def = int(reached.sum())
        self.default_support = n_def / n_total
        self.default_purity = (float((y[reached] == self.default_class).mean())
                               if n_def > 0 else 0.5)
        return self

    # ── (de)serialisation ───────────────────────────────────────────────────
    @staticmethod
    def _refine_to_dict(r):
        d = {
            "feature": r["feature"], "op": r.get("op", ">="),
            "threshold": float(r["threshold"]),
            "true_class": int(r["true_class"]), "false_class": int(r["false_class"]),
            "support": float(r.get("support", 0.0)),
            "false_support": float(r.get("false_support", 0.0)),
            "purity": float(r.get("purity", 0.5)),
            "false_purity": float(r.get("false_purity", 0.5)),
        }
        if r.get("use_abs"):
            d["use_abs"] = True
            d["prefer_higher"] = bool(r["prefer_higher"])
        if r.get("manual"):
            d["manual"] = True
        return d

    def to_dict(self):
        nodes_out = []
        for n in self.nodes:
            nd = {"feature": n["feature"], "op": n["op"],
                  "threshold": float(n["threshold"]), "exit_class": int(n["exit_class"]),
                  "support": float(n.get("support", 0.0)), "purity": float(n.get("purity", 0.5))}
            if n.get("use_abs"):
                nd["use_abs"] = True
                nd["prefer_higher"] = bool(n["prefer_higher"])
            if n.get("refine"):
                nd["refine"] = self._refine_to_dict(n["refine"])
            nodes_out.append(nd)
        return {
            "nodes": nodes_out,
            "default_class":         int(self.default_class),
            "default_support":       float(self.default_support),
            "default_purity":        float(self.default_purity),
            "default_feature":       getattr(self, "default_feature", None),
            "default_prefer_higher": getattr(self, "default_prefer_higher", None),
            "feature_names":         list(self.feature_names),
            "max_depth":             self.max_depth,
            "near_tie_threshold":    self.near_tie_threshold,
        }

    @classmethod
    def from_dict(cls, d, feature_names=None):
        t = cls(max_depth=d.get("max_depth", 4),
                near_tie_threshold=d.get("near_tie_threshold", NEAR_TIE_ABS_THRESHOLD))
        t.feature_names = list(feature_names or d.get("feature_names") or [])
        name_to_idx = {n: i for i, n in enumerate(t.feature_names)}
        t.nodes = []
        for n in d.get("nodes", []):
            node = {
                "feature":     n["feature"],
                "feature_idx": name_to_idx.get(n["feature"], 0),
                "op":          n["op"],
                "threshold":   float(n["threshold"]),
                "exit_class":  int(n.get("exit_class", 1)),
                "support":     float(n.get("support", 0.0)),
                "purity":      float(n.get("purity", 0.5)),
            }
            if n.get("use_abs"):
                node["use_abs"] = True
                node["prefer_higher"] = bool(n["prefer_higher"])
            else:
                # Migrate legacy node: infer direction from op/threshold/exit_class
                op  = node["op"]
                thr = node["threshold"]
                ec  = node["exit_class"]
                if op == ">=" and thr > 0:
                    ph = (ec == 1)
                elif op == "<=" and thr < 0:
                    ph = (ec == 0)
                elif op == ">=" and thr <= 0:
                    ph = (ec == 1)
                else:
                    ph = (ec == 0)
                node["use_abs"]       = True
                node["prefer_higher"] = ph
                node["op"]            = ">="
                node["threshold"]     = abs(thr)
            r = n.get("refine")
            if r:
                refine = {
                    "feature":       r["feature"],
                    "feature_idx":   name_to_idx.get(r["feature"], 0),
                    "op":            r.get("op", ">="),
                    "threshold":     float(r["threshold"]),
                    "true_class":    int(r["true_class"]),
                    "false_class":   int(r["false_class"]),
                    "support":       float(r.get("support", 0.0)),
                    "false_support": float(r.get("false_support", 0.0)),
                    "purity":        float(r.get("purity", 0.5)),
                    "false_purity":  float(r.get("false_purity", 0.5)),
                }
                if r.get("use_abs"):
                    refine["use_abs"]       = True
                    refine["prefer_higher"] = bool(r["prefer_higher"])
                else:
                    # Migrate legacy signed refine to abs-value format
                    op  = refine["op"]
                    thr = refine["threshold"]
                    tc  = refine["true_class"]
                    # prefer_higher: when op=">=" the condition fires for positive diffs
                    # (A higher), so true_class=1 means prefer higher. Vice-versa for "<=".
                    if op == ">=":
                        ph = (tc == 1)
                    else:  # "<="
                        ph = (tc == 0)
                    refine["use_abs"]       = True
                    refine["prefer_higher"] = ph
                    refine["op"]            = ">="
                    refine["threshold"]     = abs(thr)
                if r.get("manual"):
                    refine["manual"] = True
                node["refine"] = refine
            t.nodes.append(node)
        t.default_class          = int(d.get("default_class", 0))
        t.default_support        = float(d.get("default_support", 0.0))
        t.default_purity         = float(d.get("default_purity", 0.5))
        t.default_feature        = d.get("default_feature")
        t.default_prefer_higher  = d.get("default_prefer_higher")
        return t


# ════════════════════════════════════════════════════════════════════════════
# INFERENCE HELPERS  (single pair)
# ════════════════════════════════════════════════════════════════════════════

def feature_row(params, a_dict, b_dict, feat_names):
    """Single-row difference DataFrame aligned to feat_names."""
    row = {f"{p}_diff": float(a_dict.get(p, 0)) - float(b_dict.get(p, 0)) for p in params}
    F = pd.DataFrame([row])
    return F.reindex(columns=feat_names, fill_value=0.0)


def decision_prob(tree, params, feat_names, d):
    """Return (pred_int, prob_A) for a decision dict."""
    a = {p: float(d.get(f"A_{p}", 0)) for p in params}
    b = {p: float(d.get(f"B_{p}", 0)) for p in params}
    F = feature_row(params, a, b, feat_names)
    pred = int(tree.predict(F.values)[0])
    prob = float(tree.predict_proba(F.values)[0][1])
    return pred, prob


def _pretty(feature):
    return feature.replace("_diff", "").replace("_", " ").title()


def _fmt_num(x, **_):
    """Round to nearest integer for display."""
    return str(int(round(float(x))))


def explain_prediction(tree, params, feat_names, d, pred):
    """Plain-English explanation of which cue decided this pair."""
    a = {p: float(d.get(f"A_{p}", 0)) for p in params}
    b = {p: float(d.get(f"B_{p}", 0)) for p in params}
    F = feature_row(params, a, b, feat_names)
    cls, exit_i, purity, refine_branch = tree._row_predict(F.values[0])
    pred_label = "A" if pred == 1 else "B"

    if exit_i >= 0:
        n = tree.nodes[exit_i]
        if refine_branch is not None and n.get("refine"):
            r = n["refine"]
            tie_radius = _fmt_num(abs(n["threshold"]))
            return (
                f"The tree decided **Option {pred_label}** at step {exit_i + 1}: this pair was "
                f"a close call on `{_pretty(n['feature'])}` (|A − B| ≤ {tie_radius}), "
                f"so it used the tie-breaker check on `{_pretty(r['feature'])}` "
                f"({r['op']} {_fmt_num(r['threshold'])}) to land on Patient {pred_label}. "
                f"This tie-breaker agreed with your choices {purity:.0%} of the time in training."
            )
        return (
            f"The tree decided **Option {pred_label}** at step {exit_i + 1}: "
            f"the condition `{_pretty(n['feature'])} ({n['feature']}) "
            f"{n['op']} {_fmt_num(n['threshold'])}` held, which exits straight to "
            f"Patient {pred_label}. This cue agreed with your choices "
            f"{purity:.0%} of the time in training."
        )
    return (
        f"No cue triggered for this pair, so the tree fell through to its "
        f"default preference for **Option {pred_label}** "
        f"({tree.default_purity:.0%} reliable on training)."
    )


# ════════════════════════════════════════════════════════════════════════════
# PLAIN-ENGLISH EXPLANATIONS  (deterministic templates)
# ════════════════════════════════════════════════════════════════════════════
#
# Two things are generated here, per the study's requirements:
#   1. explain_node_llm / explain_refine_llm — a short, non-technical caption
#      for every node (and every tie-breaker).
#   2. explain_tree_summary_llm — one short paragraph summarising what the
#      participant seems to value overall.
#
# All explanations are built from deterministic templates — no network calls.

# ── LLM infrastructure disabled ─────────────────────────────────────────────
# The Groq LLM client and associated prompt definitions are kept here for
# reference but are not used. To re-enable, uncomment this block and restore
# the _groq_chat_prompt() calls in the explain_* functions below.
#
# _groq_client = None
# _groq_unavailable = False
#
# def _get_groq_client():
#     global _groq_client, _groq_unavailable
#     if _groq_client is not None or _groq_unavailable:
#         return _groq_client
#     try:
#         from groq import Groq
#         api_key = os.environ.get("GROQ_API_KEY", "")
#         if not api_key:
#             _groq_unavailable = True
#             return None
#         _groq_client = Groq(api_key=api_key)
#         return _groq_client
#     except Exception as e:
#         _groq_unavailable = True
#         return None
#
# def _groq_chat(system, user, max_tokens=180):
#     client = _get_groq_client()
#     if client is None:
#         return None
#     try:
#         resp = client.chat.completions.create(
#             model="llama-3.1-8b-instant",
#             messages=[
#                 {"role": "system", "content": system},
#                 {"role": "user", "content": user},
#             ],
#             max_tokens=max_tokens,
#             temperature=0.4,
#         )
#         text = resp.choices[0].message.content
#         return text.strip() if text else None
#     except Exception as e:
#         return None
#
# PROMPTS = { ... }  # see git history for full prompt definitions
#
# def _groq_chat_prompt(prompt_key, **fields):
#     spec = PROMPTS[prompt_key]
#     return _groq_chat(spec["system"], spec["user"].format(**fields),
#                       max_tokens=spec["max_tokens"])
# ── end LLM block ────────────────────────────────────────────────────────────

_OP_PHRASE = {
    ">=": "greater than or equal to",
    "<=": "less than or equal to",
    ">":  "greater than",
    "<":  "less than",
}

# Compact, readable comparison phrases used in explanation templates.
_OP_READABLE = {
    ">=": lambda t: f"{t} or more",
    "<=": lambda t: f"{t} or less",
    ">":  lambda t: f"more than {t}",
    "<":  lambda t: f"less than {t}",
}

# Feature-specific phrasing for captions comes from features.json (see
# features.py) — not hardcoded here, so adding/swapping study features only
# means editing that one file.
_FEATURE_GAP = features.gap_phrases()


def _feature_meta(feature, prefer_higher):
    """Return (gap_phrase, unit, direction_label) for a feature name."""
    base = feature.replace("_diff", "")
    gap_phrase, unit = _FEATURE_GAP.get(base, (_pretty(feature).lower() + " gap", ""))
    # `direction` is intentionally unused by callers (captions describe only
    # the check, not the outcome — see _fallback_node_explanation) but kept
    # in the return signature in case a future caller wants it.
    entry = features.FEATURE_BY_KEY.get(base)
    if entry:
        direction = entry["direction_higher"] if prefer_higher else entry["direction_lower"]
    else:
        direction = None
    return gap_phrase, unit, direction


def _unit_str(threshold_str, unit):
    """Return the correct singular/plural unit string."""
    if not unit:
        return ""
    singular = unit.rstrip("s") if unit.endswith("s") else unit
    u = singular if threshold_str == "1" else unit
    return f" {u}"


def _threshold_with_unit(threshold_str, unit, op=">="):
    """Format threshold + unit into a readable comparison phrase."""
    readable_fn = _OP_READABLE.get(op, lambda t: t)
    qualifier = readable_fn(threshold_str).split(threshold_str, 1)[-1].strip()  # e.g. "or more"
    if unit:
        u = _unit_str(threshold_str, unit).strip()
        return f"{threshold_str} {u} {qualifier}"
    return f"{threshold_str} {qualifier}"


def _fallback_node_explanation(node, index=0):
    """
    Describe *only* the check itself, from the user's perspective — the
    outcome ("choose the older patient", etc.) is shown separately on the
    YES-exit leaf box, so it doesn't need to be repeated here.
    """
    gap_phrase, unit, _direction = _feature_meta(node["feature"], node.get("prefer_higher", True))
    tie_radius = _fmt_num(abs(node["threshold"]))
    lead = "You first check" if index == 0 else "Next, you check"

    if node.get("refine") and node.get("op", ">=") == "<=":
        unit_str = _unit_str(tie_radius, unit)
        base_name = _pretty(node["feature"].replace("_diff", "")).lower()
        return (
            f"{lead} if the {gap_phrase} is within {tie_radius}{unit_str} \u2014 "
            f"that close counts as a tie on {base_name}, so there's one more check below."
        )

    comparison = _threshold_with_unit(tie_radius, unit, node.get("op", ">="))
    return f"{lead} if the {gap_phrase} is {comparison}."


def _fallback_refine_explanation(node, refine, near_tie_threshold):
    """Describe only the follow-up check itself; outcomes shown on its mini-leaves."""
    r_gap, r_unit, _direction = _feature_meta(refine["feature"], refine.get("prefer_higher", True))
    r_comparison = _threshold_with_unit(_fmt_num(refine["threshold"]), r_unit, refine.get("op", ">="))
    return f"If that's a tie, you then check if the {r_gap} is {r_comparison}."


def _fallback_summary_explanation(tree_dict):
    nodes = tree_dict.get("nodes", [])
    if not nodes:
        return "Not enough decisions yet to summarise a clear pattern."
    lead = nodes[0]
    bits = [
        f"The factor that seems to matter most to you is **{_pretty(lead['feature'])}** — it's "
        f"the first thing the model checks, and on its own it explained about "
        f"{lead['support']:.0%} of your choices."
    ]
    if len(nodes) > 1:
        others = ", ".join(_pretty(n["feature"]) for n in nodes[1:3])
        bits.append(
            f"After that, you also seem to weigh **{others}** when the first factor doesn't "
            f"clearly decide it."
        )
    default_txt = "Patient A" if tree_dict.get("default_class", 0) == 1 else "Patient B"
    bits.append(f"When none of these checks clearly apply, your default leaning is toward "
                f"**{default_txt}**.")
    return " ".join(bits)


def explain_node_llm(node, index=0):
    """Plain-English caption for a single tree node (describes the check only)."""
    return _fallback_node_explanation(node, index)


def explain_refine_llm(node, refine, near_tie_threshold):
    """Plain-English caption for a near-tie tie-breaker node."""
    return _fallback_refine_explanation(node, refine, near_tie_threshold)


def explain_tree_summary_llm(tree_dict):
    """A short paragraph summarising what the participant seems to value overall."""
    # LLM summary disabled — using deterministic template.
    # To re-enable: uncomment the lines below and restore _groq_chat_prompt.
    # text = _groq_chat_prompt("tree_summary", checks=<formatted node list>)
    # return text or _fallback_summary_explanation(tree_dict)
    return _fallback_summary_explanation(tree_dict)


# ════════════════════════════════════════════════════════════════════════════
# MODEL REVIEW — comparing two trained trees (e.g. before/after Part 2)
# ════════════════════════════════════════════════════════════════════════════

def summarize_model_changes(old_tree, new_tree):
    """
    Plain-English review of what changed between two trained trees — used to
    show the participant how their model shifted after answering the Part 2
    follow-up scenarios. Purely descriptive (no LLM call needed: this is a
    factual diff, not an interpretation).
    """
    old_nodes = old_tree.get("nodes", [])
    new_nodes = new_tree.get("nodes", [])
    if not old_nodes and not new_nodes:
        return "Neither model had enough data to find a clear pattern."

    old_features = [n["feature"] for n in old_nodes]
    new_features = [n["feature"] for n in new_nodes]
    bits = []

    if old_features == new_features:
        bits.append(
            "The same factors, in the same order, still drive your model — the extra "
            "scenarios confirmed the pattern rather than changing it."
        )
    else:
        added = [f for f in new_features if f not in old_features]
        removed = [f for f in old_features if f not in new_features]
        if added:
            bits.append(
                f"{'A new factor now shows up' if len(added) == 1 else 'New factors now show up'} "
                f"in your model: {', '.join(_pretty(f) for f in added)}."
            )
        if removed:
            bits.append(
                f"{'This factor dropped out' if len(removed) == 1 else 'These factors dropped out'}: "
                f"{', '.join(_pretty(f) for f in removed)}."
            )
        if not added and not removed and old_features != new_features:
            bits.append("The same factors are used, but the order changed.")

    old_refines = sum(1 for n in old_nodes if n.get("refine"))
    new_refines = sum(1 for n in new_nodes if n.get("refine"))
    if new_refines > old_refines:
        n = new_refines - old_refines
        bits.append(f"{n} new close-call tie-breaker step{'s' if n != 1 else ''} appeared.")
    elif new_refines < old_refines:
        n = old_refines - new_refines
        bits.append(f"{n} tie-breaker step{'s' if n != 1 else ''} {'are' if n != 1 else 'is'} "
                    f"no longer needed.")

    old_default = "Patient A" if old_tree.get("default_class", 0) == 1 else "Patient B"
    new_default = "Patient A" if new_tree.get("default_class", 0) == 1 else "Patient B"
    if old_default != new_default:
        bits.append(f"Your default leaning also flipped, from {old_default} to {new_default}.")

    if not bits:
        bits.append("Your model looks essentially unchanged after the extra scenarios.")
    return " ".join(bits)


# ════════════════════════════════════════════════════════════════════════════
# THINKING EVOLUTION SUMMARY  (across study phases)
# ════════════════════════════════════════════════════════════════════════════

def explain_thinking_evolution_llm(part1_tree, part2_tree, part3_tree=None):
    """
    3-4 warm plain-English sentences describing how the participant's
    decision model evolved across the study phases.

    When part3_tree is None (2-phase study), describes the Part1→Part2 shift.
    When part3_tree is provided (3-phase study), describes the full arc.
    Falls back to the deterministic summarize_model_changes() text.
    """
    diff_1_2 = summarize_model_changes(part1_tree, part2_tree)

    if part3_tree is None:
        # LLM summary disabled — returning deterministic diff text.
        # To re-enable: return _groq_chat_prompt("evolution_2phase", diff_1_2=diff_1_2) or diff_1_2
        return diff_1_2

    diff_2_3 = summarize_model_changes(part2_tree, part3_tree)
    # LLM summary disabled — returning deterministic diff text.
    # To re-enable: return _groq_chat_prompt("evolution_3phase", diff_1_2=diff_1_2, diff_2_3=diff_2_3) or fallback
    return (
        f"After the Part 2 scenarios: {diff_1_2} "
        f"After the Part 3 scenarios: {diff_2_3}"
    )


# ════════════════════════════════════════════════════════════════════════════
# TRAINING ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def train_fft(decisions_json, params, override_json=None, max_depth=4):
    """
    Train (or load an edited) FFT.

    Returns (tree, nodes_df, stats, feat_names, error_msg).
    The tuple shape mirrors the old train_rulefit() so the UI swaps cleanly.

    If override_json is given, that edited tree is used verbatim (the user has
    committed manual edits) instead of learning a fresh one — but all statistics
    are still recomputed against the real training data.
    """
    decisions = json.loads(decisions_json) if isinstance(decisions_json, str) else decisions_json
    F, feat_names = build_difference_features(decisions, params)

    if len(decisions) < 6:
        return None, None, None, feat_names, "Need at least 6 answered decisions to train."

    y = np.array([1 if d["choice"] == "A" else 0 for d in decisions])
    F_aug, y_aug = augment(F, y)

    if override_json:
        tree = FastFrugalTree.from_dict(
            json.loads(override_json) if isinstance(override_json, str) else override_json,
            feature_names=feat_names,
        )
        tree.recompute_stats(F_aug, y_aug)   # refresh support/purity for edited thresholds
        edited = True
    else:
        tree = FastFrugalTree(max_depth=max_depth).fit(F_aug, y_aug, feature_names=feat_names)
        edited = False

    # Attach/refresh the near-tie tie-breaker nodes against the real training
    # data (works whether the tree was just fit or loaded from a user override).
    # tree.attach_near_tie_refinements(F_aug.values, y_aug)
    tree.recompute_stats(F_aug, y_aug)

    if not tree.nodes:
        return None, None, None, feat_names, (
            "Could not build a tree from these decisions — answer a few more, "
            "more varied scenarios."
        )

    # ── compute semantic label for the default (fall-through) leaf ───────────
    # Find rows in the ORIGINAL (non-augmented) data that fall through to default.
    tree.default_feature        = None
    tree.default_prefer_higher  = None
    reached_o = np.ones(len(y), dtype=bool)
    for _nd in tree.nodes:
        _xs = F.values[:, _nd["feature_idx"]]
        if _nd.get("use_abs"):
            reached_o &= ~(np.abs(_xs) >= _nd["threshold"])
        else:
            reached_o &= ~(((_xs >= _nd["threshold"]) if _nd["op"] == ">="
                             else (_xs <= _nd["threshold"])))
    _idx_def = np.where(reached_o)[0]
    if len(_idx_def) >= 2 and len(np.unique(y[_idx_def])) >= 2:
        _X_def = F.values[_idx_def]
        _y_def = y[_idx_def]
        _dc    = tree.default_class
        _best_sep = -1.0
        for _fi, _fn in enumerate(feat_names):
            _xf = _X_def[:, _fi]
            _mask1 = _y_def == _dc
            if not _mask1.any() or not (~_mask1).any():
                continue
            _m1, _m0 = float(_xf[_mask1].mean()), float(_xf[~_mask1].mean())
            _sep = abs(_m1 - _m0)
            if _sep > _best_sep:
                _best_sep = _sep
                tree.default_feature       = _fn
                tree.default_prefer_higher = (_m1 > _m0)

    # ── metrics ─────────────────────────────────────────────────────────────
    acc = float(balanced_accuracy_score(y_aug, tree.predict(F_aug.values)))
    p_orig = tree.predict(F.values)
    p_swap = tree.predict((-F).values)
    sym = float(((p_orig + p_swap) == 1).mean())

    # ── nodes_df (drives the readable Rules tab + LLM, mirrors old rules_df) ──
    rows = []
    for n in tree.nodes:
        direction = 1 if n["exit_class"] == 1 else -1   # +A / -B
        strength = max(0.0, (n["purity"] - 0.5) * 2.0)  # 0..1
        rule_text = (f'|{n["feature"]}| <= {_fmt_num(abs(n["threshold"]))}' if n.get("refine")
                     else f'{n["feature"]} {n["op"]} {_fmt_num(n["threshold"])}')
        row = {
            "feature":    n["feature"],
            "rule":       rule_text,
            "op":         n["op"],
            "threshold":  float(n["threshold"]),
            "exit_class": int(n["exit_class"]),
            "support":    float(n["support"]),
            "purity":     float(n["purity"]),
            "coef":       float(direction * strength),
            "importance": float(n["support"] * strength),
        }
        r = n.get("refine")
        if r:
            row["refine_rule"] = f'{r["feature"]} {r["op"]} {_fmt_num(r["threshold"])}'
            row["refine_true_class"] = int(r["true_class"])
            row["refine_false_class"] = int(r["false_class"])
        rows.append(row)
    nodes_df = pd.DataFrame(rows)

    # ── plain-English explanations (LLM, deterministic fallback) ─────────────
    node_explanations = []
    for i, n in enumerate(tree.nodes):
        entry = {"explanation": explain_node_llm(n, i)}
        if n.get("refine"):
            entry["refine_explanation"] = explain_refine_llm(n, n["refine"], tree.near_tie_threshold)
        node_explanations.append(entry)
    summary_explanation = explain_tree_summary_llm(tree.to_dict())

    # ── per-decision agreement (drives Examples tab) ─────────────────────────
    preds_o = tree.predict(F.values)
    per_decision = []
    for i, (d, actual, predicted) in enumerate(zip(decisions, y, preds_o)):
        per_decision.append({
            "scenario":  d.get("scenario", i + 1),
            "actual":    "A" if actual == 1 else "B",
            "predicted": "A" if predicted == 1 else "B",
            "match":     bool(actual == predicted),
            "decision":  d,
        })

    stats = {
        "acc":          acc,
        "sym":          sym,
        "n_rules":      len(tree.nodes),
        "n_decisions":  len(decisions),
        "top_param":    _pretty(tree.nodes[0]["feature"]),
        "edited":       edited,
        "per_decision": per_decision,
        "tree":         tree.to_dict(),
        "node_explanations":   node_explanations,
        "summary_explanation": summary_explanation,
    }

    return tree, nodes_df, stats, feat_names, None