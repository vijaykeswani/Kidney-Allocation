"""
SQL storage layer for the Preference Elicitation Portal.
SURA 2026 · IIT Delhi

Replaces the previous `users.json` flat-file store with a SQLite database
(`portal.db`). The schema is a direct relational translation of the JSON
document that used to live in users.json:

    users.json                              SQL
    ─────────────────────────────────────   ─────────────────────────────────
    <username>                              users.username           (PK)
    <username>.part1_index                  users.part1_index
    <username>.part2_index                  users.part2_index
    <username>.alignment_score              users.alignment_score
    <username>.fft_override                 users.fft_override       (JSON text)
    <username>.part1_scenarios[i]           scenarios(part=1, position=i)
    <username>.part2_scenarios[i]           scenarios(part=2, position=i)
    <username>.part1[i]                     decisions(part=1, seq=i)
    <username>.part2[i]                     decisions(part=2, seq=i)

`load_all_users()` reconstructs *exactly* the same nested dict the JSON file
used to yield, and `save_user_record()` / `save_all_users()` write it back, so
every caller in app.py keeps working against an unchanged data shape.

Only the stdlib `sqlite3` module is used — no new dependency.
"""

import json
import os
import sqlite3

APP_DIR      = os.path.dirname(os.path.abspath(__file__))
DB_PATH      = os.environ.get("DB_PATH") or os.path.join(APP_DIR, "portal.db")
LEGACY_JSON  = os.path.join(APP_DIR, "users.json")

# Set by init_db() — the parameter list the A_/B_ columns are built from.
_FEATURES = []


# ── feature-column migration ──────────────────────────────────────────────────
# Called before CREATE TABLE IF NOT EXISTS. If the scenarios table already
# exists but doesn't have the expected first feature column, the study's
# feature set has changed and the dynamic tables must be rebuilt (the stored
# data is incompatible with the new column layout).

def _drop_stale_feature_tables():
    if not _FEATURES:
        return
    with _connect() as conn:
        tbl = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='scenarios'"
        ).fetchone()
        if not tbl:
            return
        existing = {r["name"] for r in conn.execute("PRAGMA table_info(scenarios)")}
        expected = {f"A_{k}" for k in _FEATURES} | {f"B_{k}" for k in _FEATURES}
        if not expected.issubset(existing):
            conn.executescript("""
                DROP TABLE IF EXISTS trials;
                DROP TABLE IF EXISTS decisions;
                DROP TABLE IF EXISTS scenarios;
            """)
            print("[db] Feature set changed — dropped stale dynamic tables.")


# ── connection ────────────────────────────────────────────────────────────────

def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _cols(prefix):
    """Column name list for one side of a pair, e.g. A_age, A_dependents, ..."""
    return [f"{prefix}_{p}" for p in _FEATURES]


# ── schema ────────────────────────────────────────────────────────────────────

def init_db(features):
    """
    Create the schema if it does not exist, then import users.json once if the
    database is still empty (so existing participant data is preserved).

    `features` is the app's FEATURES list; the per-parameter A_/B_ columns are
    generated from it so the schema always matches the study's parameter set.
    """
    global _FEATURES
    _FEATURES = list(features)

    pair_cols = ",\n            ".join(
        f"{c} REAL" for c in (_cols("A") + _cols("B"))
    )

    _drop_stale_feature_tables()

    with _connect() as conn:
        conn.executescript(f"""
        CREATE TABLE IF NOT EXISTS users (
            username        TEXT PRIMARY KEY,
            part1_index     INTEGER NOT NULL DEFAULT 0,
            part2_index     INTEGER NOT NULL DEFAULT 0,
            alignment_score INTEGER,
            fft_override    TEXT
        );

        CREATE TABLE IF NOT EXISTS scenarios (
            username  TEXT    NOT NULL,
            part      INTEGER NOT NULL,
            position  INTEGER NOT NULL,
            {pair_cols},
            PRIMARY KEY (username, part, position),
            FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS decisions (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            username  TEXT    NOT NULL,
            part      INTEGER NOT NULL,
            seq       INTEGER NOT NULL,
            scenario  INTEGER NOT NULL,
            choice    TEXT    NOT NULL,
            timestamp TEXT    NOT NULL,
            {pair_cols},
            FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_decisions_user_part
            ON decisions(username, part, seq);

        /* ── Review flow ────────────────────────────────────────────────────
           One row per "try a new scenario" attempt. Each row is a complete,
           self-contained agreement/disagreement record: the pair that was
           shown, what the participant picked, what the model predicted, and
           the model version that made the prediction. Grouping by
           model_version is what keeps each model's history independent of the
           one before it — editing the tree starts a fresh set rather than
           re-labelling the old one.                                          */
        CREATE TABLE IF NOT EXISTS trials (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT    NOT NULL,
            model_version INTEGER NOT NULL DEFAULT 0,
            created_at    TEXT    NOT NULL,
            {pair_cols},
            user_choice   TEXT,
            model_choice  TEXT,
            agreed        INTEGER,
            exit_index    INTEGER,
            exit_reason   TEXT,
            reasoning     TEXT,
            FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_trials_user_version
            ON trials(username, model_version, id);

        /* Lightweight interaction log — one row per notable UI event, so a
           session can be reconstructed during analysis without any of the
           study tables having to carry UI concerns.                          */
        CREATE TABLE IF NOT EXISTS ui_events (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            username  TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            event     TEXT NOT NULL,
            detail    TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_ui_events_user
            ON ui_events(username, id);
        """)

    _add_missing_columns()
    _migrate_legacy_json()


# Columns added after the original schema shipped. Existing portal.db files
# predate them, so they are applied additively at startup rather than by
# recreating the table — no data is touched and re-running is a no-op.
_ADDED_USER_COLUMNS = [
    ("model_version",       "INTEGER NOT NULL DEFAULT 0"),
    ("tutorial_seen_count", "INTEGER NOT NULL DEFAULT 0"),
    # Which model type this participant sees for the whole study — set once
    # at signup, never changed after. Existing rows default to "fft" so
    # nothing about already-running sessions changes.
    ("model_type",          "TEXT NOT NULL DEFAULT 'fft'"),
    # Control-arm participants never pick a model_type (it's "control"), but
    # under the symmetric end-of-study design they still see *a* trained
    # model once, read-only, at the very end — this says which kind. Chosen
    # at random when their session starts (see app.py do_start). Unused for
    # treatment arms, which just use model_type directly.
    ("display_model_type",  "TEXT"),
    # End-of-study trust/alignment/demographics questionnaire — one JSON blob
    # per participant, same storage pattern as fft_override.
    ("survey_json",         "TEXT"),
    # The model the participant is locked into when they leave the model screen:
    # their edited model (treatment arms) or the unedited Part-1 model (control).
    # Part 2 then tests THIS model on fresh answers; it is never retrained.
    ("frozen_model_json",   "TEXT"),
]


def _add_missing_columns():
    with _connect() as conn:
        existing = {r["name"] for r in conn.execute("PRAGMA table_info(users)")}
        for col, decl in _ADDED_USER_COLUMNS:
            if col not in existing:
                conn.execute(f"ALTER TABLE users ADD COLUMN {col} {decl}")


def _migrate_legacy_json():
    """One-time import of users.json, only when the users table is still empty."""
    with _connect() as conn:
        already = conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"]
    if already or not os.path.exists(LEGACY_JSON):
        return
    try:
        with open(LEGACY_JSON) as f:
            legacy = json.load(f)
    except Exception as e:
        print(f"[db] Could not read {LEGACY_JSON} for migration: {e}")
        return
    if not isinstance(legacy, dict) or not legacy:
        return

    # Skip migration if the legacy data uses a different feature set — happens
    # when features.json is updated but users.json is from a prior study run.
    for rec in legacy.values():
        scens = rec.get("part1_scenarios") or rec.get("part2_scenarios") or []
        if scens:
            sample = scens[0].get("A", {})
            if any(p not in sample for p in _FEATURES):
                print("[db] Skipping users.json migration — feature set has changed.")
                return
            break

    save_all_users(legacy)
    print(f"[db] Migrated {len(legacy)} user record(s) from users.json into {DB_PATH}")


# ── read ──────────────────────────────────────────────────────────────────────

def _scenario_from_row(row):
    return {
        "A": {p: row[f"A_{p}"] for p in _FEATURES},
        "B": {p: row[f"B_{p}"] for p in _FEATURES},
    }


def _decision_from_row(row):
    """Rebuild the decision dict in the same key order the JSON store used."""
    d = {
        "username":  row["username"],
        "scenario":  row["scenario"],
        "choice":    row["choice"],
        "timestamp": row["timestamp"],
    }
    for p in _FEATURES:
        d[f"A_{p}"] = row[f"A_{p}"]
        d[f"B_{p}"] = row[f"B_{p}"]
    return d


def load_all_users():
    """Return the full store in the exact shape users.json used to hold."""
    out = {}
    with _connect() as conn:
        for u in conn.execute("SELECT * FROM users ORDER BY username"):
            name = u["username"]
            rec = {}

            for part in (1, 2):
                scen = [
                    _scenario_from_row(r)
                    for r in conn.execute(
                        "SELECT * FROM scenarios WHERE username=? AND part=? "
                        "ORDER BY position", (name, part))
                ]
                dec = [
                    _decision_from_row(r)
                    for r in conn.execute(
                        "SELECT * FROM decisions WHERE username=? AND part=? "
                        "ORDER BY seq", (name, part))
                ]
                if scen:
                    rec[f"part{part}_scenarios"] = scen
                    rec[f"part{part}"] = dec
                    rec[f"part{part}_index"] = u[f"part{part}_index"]

            if u["alignment_score"] is not None:
                rec["alignment_score"] = u["alignment_score"]
            if u["fft_override"]:
                rec["fft_override"] = json.loads(u["fft_override"])

            out[name] = rec
    return out


# ── write ─────────────────────────────────────────────────────────────────────

def save_user_record(username, record):
    """
    Persist one participant's record. The record dict is the source of truth —
    the user's scenarios and decisions are rewritten to match it, mirroring the
    whole-document overwrite the JSON store performed on every save.
    """
    with _connect() as conn:
        _write_user(conn, username, record)


def save_all_users(users):
    """Persist the whole store (kept so app.save_users() keeps its signature)."""
    with _connect() as conn:
        for username, record in users.items():
            _write_user(conn, username, record or {})


def _write_user(conn, username, record):
    override = record.get("fft_override")
    conn.execute(
        """
        INSERT INTO users (username, part1_index, part2_index,
                           alignment_score, fft_override)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(username) DO UPDATE SET
            part1_index     = excluded.part1_index,
            part2_index     = excluded.part2_index,
            alignment_score = excluded.alignment_score,
            fft_override    = excluded.fft_override
        """,
        (
            username,
            int(record.get("part1_index", 0) or 0),
            int(record.get("part2_index", 0) or 0),
            record.get("alignment_score"),
            json.dumps(override) if override else None,
        ),
    )

    scen_cols = _cols("A") + _cols("B")
    for part in (1, 2):
        scenarios = record.get(f"part{part}_scenarios")
        if scenarios is not None:
            conn.execute("DELETE FROM scenarios WHERE username=? AND part=?",
                         (username, part))
            conn.executemany(
                f"INSERT INTO scenarios (username, part, position, "
                f"{', '.join(scen_cols)}) VALUES "
                f"(?, ?, ?, {', '.join('?' * len(scen_cols))})",
                [
                    (username, part, i)
                    + tuple(float(sc["A"][p]) for p in _FEATURES)
                    + tuple(float(sc["B"][p]) for p in _FEATURES)
                    for i, sc in enumerate(scenarios)
                ],
            )

        decisions = record.get(f"part{part}")
        if decisions is not None:
            conn.execute("DELETE FROM decisions WHERE username=? AND part=?",
                         (username, part))
            conn.executemany(
                f"INSERT INTO decisions (username, part, seq, scenario, choice, "
                f"timestamp, {', '.join(scen_cols)}) VALUES "
                f"(?, ?, ?, ?, ?, ?, {', '.join('?' * len(scen_cols))})",
                [
                    (username, part, i, int(d.get("scenario", i + 1)),
                     d.get("choice"), d.get("timestamp"))
                    + tuple(float(d.get(f"A_{p}", 0.0)) for p in _FEATURES)
                    + tuple(float(d.get(f"B_{p}", 0.0)) for p in _FEATURES)
                    for i, d in enumerate(decisions)
                ],
            )


# ── targeted helpers (fft_override) ───────────────────────────────────────────

def set_fft_override(username, tree_dict):
    with _connect() as conn:
        conn.execute(
            "INSERT INTO users (username, fft_override) VALUES (?, ?) "
            "ON CONFLICT(username) DO UPDATE SET fft_override = excluded.fft_override",
            (username, json.dumps(tree_dict)),
        )


def get_fft_override(username):
    with _connect() as conn:
        row = conn.execute(
            "SELECT fft_override FROM users WHERE username=?", (username,)
        ).fetchone()
    return json.loads(row["fft_override"]) if row and row["fft_override"] else None


def delete_fft_override(username):
    with _connect() as conn:
        conn.execute("UPDATE users SET fft_override=NULL WHERE username=?", (username,))


def set_frozen_model(username, frozen):
    with _connect() as conn:
        conn.execute(
            "INSERT INTO users (username, frozen_model_json) VALUES (?, ?) "
            "ON CONFLICT(username) DO UPDATE SET frozen_model_json = excluded.frozen_model_json",
            (username, json.dumps(frozen)),
        )


def get_frozen_model(username):
    with _connect() as conn:
        row = conn.execute(
            "SELECT frozen_model_json FROM users WHERE username=?", (username,)
        ).fetchone()
    return json.loads(row["frozen_model_json"]) if row and row["frozen_model_json"] else None


def count_events(username, event):
    with _connect() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM ui_events WHERE username=? AND event=?", (username, event)
        ).fetchone()[0]


def set_survey_answers(username, answers):
    with _connect() as conn:
        conn.execute(
            "INSERT INTO users (username, survey_json) VALUES (?, ?) "
            "ON CONFLICT(username) DO UPDATE SET survey_json = excluded.survey_json",
            (username, json.dumps(answers)),
        )


def get_survey_answers(username):
    with _connect() as conn:
        row = conn.execute(
            "SELECT survey_json FROM users WHERE username=?", (username,)
        ).fetchone()
    return json.loads(row["survey_json"]) if row and row["survey_json"] else None


# ── model version ─────────────────────────────────────────────────────────────
#
# Bumped every time the participant applies or resets an edit. Trials record the
# version that predicted them, which is what makes each model's
# agreement/disagreement history independent of the previous model's.

def get_model_version(username):
    with _connect() as conn:
        row = conn.execute(
            "SELECT model_version FROM users WHERE username=?", (username,)
        ).fetchone()
    return int(row["model_version"]) if row and row["model_version"] is not None else 0


def bump_model_version(username):
    with _connect() as conn:
        conn.execute(
            "INSERT INTO users (username, model_version) VALUES (?, 1) "
            "ON CONFLICT(username) DO UPDATE SET "
            "model_version = COALESCE(users.model_version, 0) + 1",
            (username,),
        )
        row = conn.execute(
            "SELECT model_version FROM users WHERE username=?", (username,)
        ).fetchone()
    return int(row["model_version"]) if row else 0


# ── model type ────────────────────────────────────────────────────────────────
# Chosen once at signup (see app.py's do_start), never changed after — a
# participant sees exactly one of "fft" / "twostage" for their whole session.

def set_model_type(username, model_type):
    with _connect() as conn:
        conn.execute(
            "INSERT INTO users (username, model_type) VALUES (?, ?) "
            "ON CONFLICT(username) DO UPDATE SET model_type = excluded.model_type",
            (username, model_type),
        )


def get_model_type(username):
    with _connect() as conn:
        row = conn.execute(
            "SELECT model_type FROM users WHERE username=?", (username,)
        ).fetchone()
    return row["model_type"] if row and row["model_type"] else "fft"


def set_display_model_type(username, display_model_type):
    with _connect() as conn:
        conn.execute(
            "INSERT INTO users (username, display_model_type) VALUES (?, ?) "
            "ON CONFLICT(username) DO UPDATE SET display_model_type = excluded.display_model_type",
            (username, display_model_type),
        )


def get_display_model_type(username):
    with _connect() as conn:
        row = conn.execute(
            "SELECT display_model_type, model_type FROM users WHERE username=?", (username,)
        ).fetchone()
    if not row:
        return "fft"
    # Treatment arms just use their own model_type; only control needed a
    # separately-assigned one.
    return row["display_model_type"] or row["model_type"] or "fft"


# ── tutorial ──────────────────────────────────────────────────────────────────

def get_tutorial_seen_count(username):
    with _connect() as conn:
        row = conn.execute(
            "SELECT tutorial_seen_count FROM users WHERE username=?", (username,)
        ).fetchone()
    return int(row["tutorial_seen_count"]) if row and row["tutorial_seen_count"] is not None else 0


def bump_tutorial_seen_count(username):
    with _connect() as conn:
        conn.execute(
            "INSERT INTO users (username, tutorial_seen_count) VALUES (?, 1) "
            "ON CONFLICT(username) DO UPDATE SET "
            "tutorial_seen_count = COALESCE(users.tutorial_seen_count, 0) + 1",
            (username,),
        )


# ── trials (agreements / disagreements) ───────────────────────────────────────

def _trial_from_row(row):
    return {
        "id":            row["id"],
        "model_version": row["model_version"],
        "created_at":    row["created_at"],
        "A":             {p: row[f"A_{p}"] for p in _FEATURES},
        "B":             {p: row[f"B_{p}"] for p in _FEATURES},
        "user_choice":   row["user_choice"],
        "model_choice":  row["model_choice"],
        "agreed":        None if row["agreed"] is None else bool(row["agreed"]),
        "exit_index":    row["exit_index"],
        "exit_reason":   row["exit_reason"],
        "reasoning":     row["reasoning"],
    }


def create_trial(username, model_version, scenario, created_at):
    """
    Open a new trial for a freshly generated pair. The outcome columns stay
    empty until the participant answers, so an abandoned scenario is
    distinguishable from an answered one rather than silently counting as a
    disagreement.
    """
    cols = _cols("A") + _cols("B")
    values = (
        tuple(float(scenario["A"][p]) for p in _FEATURES)
        + tuple(float(scenario["B"][p]) for p in _FEATURES)
    )
    with _connect() as conn:
        cur = conn.execute(
            f"INSERT INTO trials (username, model_version, created_at, "
            f"{', '.join(cols)}) VALUES (?, ?, ?, "
            f"{', '.join('?' * len(cols))})",
            (username, int(model_version), created_at) + values,
        )
        return cur.lastrowid


def record_trial_outcome(trial_id, username, user_choice, model_choice,
                         exit_index=None, exit_reason=None):
    with _connect() as conn:
        conn.execute(
            "UPDATE trials SET user_choice=?, model_choice=?, agreed=?, "
            "exit_index=?, exit_reason=? WHERE id=? AND username=?",
            (user_choice, model_choice,
             1 if user_choice == model_choice else 0,
             exit_index, exit_reason, int(trial_id), username),
        )


def record_trial_reasoning(trial_id, username, reasoning):
    with _connect() as conn:
        conn.execute(
            "UPDATE trials SET reasoning=? WHERE id=? AND username=?",
            (reasoning, int(trial_id), username),
        )


def get_trial(trial_id, username):
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM trials WHERE id=? AND username=?", (int(trial_id), username)
        ).fetchone()
    return _trial_from_row(row) if row else None


def list_trials(username, model_version=None, answered_only=True):
    sql = "SELECT * FROM trials WHERE username=?"
    args = [username]
    if model_version is not None:
        sql += " AND model_version=?"
        args.append(int(model_version))
    if answered_only:
        sql += " AND user_choice IS NOT NULL"
    sql += " ORDER BY id"
    with _connect() as conn:
        return [_trial_from_row(r) for r in conn.execute(sql, args)]


def list_trial_versions(username):
    """Model versions that have at least one answered trial, newest first."""
    with _connect() as conn:
        return [
            int(r["model_version"])
            for r in conn.execute(
                "SELECT DISTINCT model_version FROM trials "
                "WHERE username=? AND user_choice IS NOT NULL "
                "ORDER BY model_version DESC", (username,))
        ]


def seen_pairs(username):
    """
    Every A/B pair the participant has already been shown as a trial, as a set
    of rounded tuples. Used to keep newly generated scenarios genuinely new.
    """
    out = set()
    with _connect() as conn:
        for r in conn.execute("SELECT * FROM trials WHERE username=?", (username,)):
            out.add(
                tuple(round(float(r[f"A_{p}"]), 3) for p in _FEATURES)
                + tuple(round(float(r[f"B_{p}"]), 3) for p in _FEATURES)
            )
    return out


# ── interaction log ───────────────────────────────────────────────────────────

def log_event(username, event, detail=None, timestamp=None):
    """
    Append one UI event. Deliberately fire-and-forget: logging must never be
    able to break the interaction it is recording.
    """
    from datetime import datetime
    try:
        with _connect() as conn:
            conn.execute(
                "INSERT INTO ui_events (username, timestamp, event, detail) "
                "VALUES (?, ?, ?, ?)",
                (username, timestamp or datetime.now().isoformat(), str(event)[:120],
                 json.dumps(detail) if detail is not None else None),
            )
    except Exception as e:      # pragma: no cover - logging is best-effort
        print(f"[db] log_event failed: {e}")


def list_events(username, limit=500):
    with _connect() as conn:
        return [
            {"timestamp": r["timestamp"], "event": r["event"],
             "detail": json.loads(r["detail"]) if r["detail"] else None}
            for r in conn.execute(
                "SELECT * FROM ui_events WHERE username=? ORDER BY id DESC LIMIT ?",
                (username, int(limit)))
        ]