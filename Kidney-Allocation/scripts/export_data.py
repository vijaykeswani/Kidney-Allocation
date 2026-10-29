"""
Export every table from a SQLite database to CSV files.

Usage:
    python scripts/export_data.py [path/to/portal.db]

Defaults to ../portal.db relative to this script if no path is given.
Output CSVs land in ./export/ (created if absent).
The source database is opened read-only; nothing is written to it.
"""

import csv
import json
import os
import sqlite3
import sys
from datetime import datetime


def _secs(a, b):
    try:
        return round((datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds(), 1)
    except Exception:
        return None


def export_participant_summary(conn, out_dir: str, stamp: str) -> str:
    """
    One row per participant: arm, the locked model, how it did on the Part-2 hold-out
    (and how the unedited model would have), interaction counts, timings, and the
    survey answers flattened into columns. Everything here is derived from the raw
    tables exported alongside it, so nothing is lost by using this file.
    """
    events = {}
    for r in conn.execute("SELECT username, timestamp, event FROM ui_events ORDER BY id"):
        events.setdefault(r["username"], []).append((r["timestamp"], r["event"]))

    rows, survey_cols = [], []
    for u in conn.execute("SELECT * FROM users ORDER BY username"):
        keys = u.keys()
        get = lambda k: u[k] if k in keys else None
        frozen = json.loads(get("frozen_model_json")) if get("frozen_model_json") else {}
        survey = json.loads(get("survey_json")) if get("survey_json") else {}
        ev = events.get(u["username"], [])
        count = lambda e: sum(1 for _, x in ev if x == e)
        first = lambda e: next((t for t, x in ev if x == e), None)
        hold = frozen.get("holdout", {})
        row = {
            "username": u["username"],
            "model_type": get("model_type"),
            "display_model_type": get("display_model_type"),
            "feature_order": " > ".join(json.loads(get("feature_order"))) if get("feature_order") else None,
            "arm": frozen.get("arm"),
            "alignment_score": get("alignment_score"),
            "model_edited": frozen.get("edited"),
            "edits_applied": frozen.get("edit_count"),
            "n_part1": frozen.get("n_part1"),
            "part1_matches_before_edit": frozen.get("matches_before_edit"),
            "part1_matches_at_lock": frozen.get("matches_at_lock"),
            "holdout_n": hold.get("n"),
            "holdout_matches": hold.get("matches"),
            "holdout_matches_if_unedited": hold.get("matches_if_unedited"),
            "model_edit_events": count("model_edit"),
            "test_model_opened": count("test_model_opened"),
            "cases_viewed": count("case_viewed"),
            "custom_scenarios_tried": count("custom_scenario_tried"),
            "results_viewed_at": first("results_viewed"),
            "model_locked_at": first("model_locked"),
            "final_viewed_at": first("final_viewed"),
            "survey_submitted_at": first("survey_submitted"),
        }
        row["secs_on_model_screen"] = _secs(row["results_viewed_at"], row["model_locked_at"]) if row["results_viewed_at"] else None
        row["secs_locked_to_final"] = _secs(row["model_locked_at"], row["final_viewed_at"])
        for k, v in survey.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    row[f"survey_{k}__{kk}"] = vv; survey_cols.append(f"survey_{k}__{kk}")
            else:
                row[f"survey_{k}"] = v; survey_cols.append(f"survey_{k}")
        rows.append(row)

    base_cols = [k for k in rows[0] if not k.startswith("survey_")] if rows else []
    cols = base_cols + sorted(set(survey_cols))
    out_path = os.path.join(out_dir, f"participant_summary_{stamp}.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  {'participant_summary':20s} → {out_path}  ({len(rows)} rows)")
    return out_path


def export(db_path: str, out_dir: str = "export") -> None:
    if not os.path.exists(db_path):
        sys.exit(f"Error: {db_path!r} not found.")

    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row

    tables = [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
    ]

    if not tables:
        print("No tables found.")
        conn.close()
        return

    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for table in tables:
        rows = conn.execute(f"SELECT * FROM [{table}]").fetchall()
        out_path = os.path.join(out_dir, f"{table}_{stamp}.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if rows:
                writer.writerow(rows[0].keys())
                writer.writerows(rows)
            else:
                writer.writerow([])
        print(f"  {table:20s} → {out_path}  ({len(rows)} rows)")

    export_participant_summary(conn, out_dir, stamp)
    conn.close()
    print(f"\nDone. {len(tables)} table(s) exported from {db_path!r}")


if __name__ == "__main__":
    default_db = os.path.join(os.path.dirname(__file__), "..", "portal.db")
    db_path = sys.argv[1] if len(sys.argv) > 1 else default_db
    export(os.path.abspath(db_path))
