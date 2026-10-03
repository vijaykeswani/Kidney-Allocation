# Organ Allocation Preference Study — Flask edition

## Run
    pip install -r requirements.txt
    export FLASK_SECRET_KEY=some-random-value   # required, no default — app.py will KeyError without it
    flask --app app run --debug
Then open http://127.0.0.1:5000

Notes on running it:
- `FLASK_SECRET_KEY` must be set (via env var or a `.env` file — `python-dotenv` is loaded
  at startup). There's no fallback default anymore.
- `fft/static/fft_tree.js` is served at `/fft/static/fft_tree.js` by the `fft` blueprint -
  no extra config needed.
- Optional: set `FFT_TIEBREAKER=1` to turn on the experimental multi-step tie-breaker
  (`fft/tiebreaker.py`) for ambiguous predictions in the trial/review flow; off by default.

## Project layout
Each model has its own folder, so changing one cannot touch the other. Everything a
participant sees is the same as before the split; only file locations changed.

    app.py, db.py, features.py/.json   shared: routes, database, feature definitions
    templates/                         shared pages: base, consent, start,
                                       questionnaire, not_ready
    fft/                               Fast-and-Frugal Tree (decision tree)
        fft_model.py, fft_component.py, tiebreaker.py
        templates/  model.html, review.html, final.html
        static/     fft_tree.js
    twostage/                          two-stage (feature list) model
        twostage_model.py
        templates/  twostage.html

Each folder is a Flask blueprint (see its `__init__.py`), registered in `app.py`. That only
makes the folder's `templates/` and `static/` visible: template names are unchanged, so
`render_template("model.html")` still works. Template names must therefore stay unique
across folders. Participants pick a model on the start page, or a study link with
`?model=fft` / `?model=twostage` picks it for them; the choice is stored in
`users.model_type`.

Older notes below name files by their bare names (`fft_model.py`, `model.html`); they now
live in the folders above.

## Latest UI changes (this pass)
- **Insert-step bug, fixed properly**: "Add a rule after this step" (leaf panel) now inserts a
  plain, general sequential rule right after the clicked step — not a tie-breaker, and not
  silently appended to the end of the whole list. It auto-opens the new rule so you can
  configure it immediately. "Add tie-breaker" (inside the main edit panel) is unchanged and
  still available as a separate, more advanced option for near-tie cases.
- **Descriptive direction labels**: edit-panel outcome buttons now read as full instructions
  ("Choose the older patient", "Choose the one who's waited longer") instead of bare
  fragments ("older patient", "longer wait"). See `PARAM_DIRECTION_LABELS` in
  `fft_component.py` — the compact SVG leaf-box labels are a separate, intentionally-short
  dict and were left alone (box width is fixed).
- **"Difference in X" dropdowns**: the "Factor" dropdowns now say "Difference in Age" etc.
  instead of just "Age", since every value in this model actually is A-minus-B.
- **Per-rule stats surfaced during editing**: each step now shows a small colored line —
  e.g. "82% match · 34% of scenarios apply" (green/amber/red by match quality), or "Not
  checked against your answers yet" for a rule you just added. Ties the abstract rule
  back to the person's own data while they're deciding whether to trust/change it.
- **Reset to original model**: once any edit has been applied, a "↺ Reset to original model"
  link appears next to "Edit the model", clearing the override and reloading the freshly-
  trained (un-edited) tree. Removes the fear of "permanently breaking" the model, which
  should make people more willing to actually experiment with edits.

## Further interaction ideas (not implemented — for the trust/alignment research question)
- **Live what-if tester**: two side-by-side patient sliders (or a "surprise me" random
  pair) that show which side the *current* model picks, updating live as thresholds are
  edited — turns abstract threshold edits into a concrete, checkable prediction.
- **"Explain this decision" trace**: pick any past scenario (from the person's own 20+10)
  and highlight exactly which rule fired and why, so the model's behavior is traceable to
  a choice the person actually remembers making.
- **Edit diff summary before applying**: a plain-English "what changed" line ("Step 2:
  threshold moved from ≥3 to ≥5 years") shown before committing, so people can see the
  effect of their edit before it's permanent for that round.
- **Per-rule "does this match how I think?" micro-rating**: a quick 👍/👎 on each individual
  rule (separate from the overall 7-point score) to localize *where* trust breaks down,
  rather than only measuring it in aggregate.
- **Compare original vs. edited model side-by-side**: after editing, show both trees (or
  just their differing rules) rather than only the new one, so the edit's impact is visible
  rather than assumed.

## Latest changes (this pass) — features.json: swap the study's features in one place
- **`features.json`** is now the single source of truth for every feature in the study —
  key, display label, description, min/max range, unit, the caption gap-phrase, and the
  full and short "choose the X patient" direction labels. To use a different set of
  features later, edit this file only.
- **`features.py`** is a small shared loader (`FEATURES_DATA`, `FEATURE_BY_KEY`, plus
  `descriptions()`/`ranges()`/`direction_labels()`/`gap_phrases()` helpers) used by
  `app.py`, `fft_model.py` and `fft_component.py` — one load, not three copies.
- Removed real duplication found along the way: `fft_component.py`'s
  `PARAM_DIRECTION_LABELS` and `_PARAM_LABELS_SVG` were two separately-maintained dicts
  with identical content (now one); `fft_model.py`'s `_FEATURE_DIRECTION` was computed
  every time but never actually used (removed).
- `model.html`'s JS builds `FEATURE_GAP` and `SHORT_LABELS` from an injected
  `FEATURES_DATA` array (from `features_json` in the `/results` route) instead of a
  hardcoded JS object mirroring the Python one.
- **Verified with an actual feature swap, not just code review**: temporarily replaced
  "dependents" with a fictional "distance_km" feature in `features.json` and confirmed
  the new label/description/range flowed correctly through the start page, the
  questionnaire, and the trained model's results page — including catching and fixing a
  real gap where `questionnaire.html`, `review.html` and `start.html` were mechanically
  title-casing the feature key instead of reading the actual label (harmless for the
  current features, since e.g. "age" title-cases fine, but would've shown "Distance Km"
  instead of "Distance (km)" for a key that doesn't title-case cleanly).

## Latest changes (previous pass) — branching tree layout + feature merge from GitHub repo
- **Tree is now a real branching diagram, not a vertical checklist.** Rendering moved
  into `static/fft_tree.js`, a proper tidy-tree layout engine (two-pass measure/place):
  first check at the top, YES branches down-left to its outcome, NO branches down-right
  to the next check, with clean elbow connectors. This came from
  `github.com/Web-Hosting7/Kidney-Allocation`, which had done exactly this rewrite —
  `model.html` now calls `FFTTree.build(tree, {...helpers})` instead of hand-drawing a
  single-column SVG.
- **Merged in from that same source, verified working**: zoom controls (−/level/+/Fit),
  a first-visit tutorial walkthrough (with a lighter reminder on the 3rd visit, driven by
  `db.get_tutorial_seen_count`), the top legend banner, a "Review agreements &
  disagreements →" link, a "?" help button that reopens the tutorial, and a live
  "You can add N more checks in this pass" counter enforcing `MAX_NEW_FEATURES`
  (2 by default) both in the UI and as a `baseline_node_count` check on `/edit/apply`.
- **What did NOT come along from that repo, on purpose**: its `app.py`/`fft_component.py`
  had the same CSV-dependency and wording reverts fixed two passes ago — kept our fixed
  versions instead. Its "test the model" feature also predated the modal redesign from
  last pass (used an inline panel + the formal server-side trial system for new
  scenarios); replaced with our two-option modal (editable random-prefill sandbox +
  past-answers chip grid), reusing its `pastScenarios`/`evalTreeRow` client-side eval
  logic since that part was identical. Also caught and re-fixed "tie-breaker" branding
  that had reverted alongside the rest of that file, in both `model.html` and
  `fft_tree.js`.
- **Verified with a real DOM run, not just a syntax check**: rendered `/results` through
  jsdom (inlining `fft_tree.js` since jsdom can't fetch external scripts from a static
  HTML dump) and confirmed the tree has genuinely branching box positions (not a single
  column), the default leaf reads "Too similar to differentiate" in neutral gray, zoom/
  tutorial/legend all render, the modal opens and both its views work, and edit mode's
  inline "+" badges and the add-limit counter both function correctly.

## Earlier changes
- **No more CSV dependency**: scenarios are fully synthetic now. `FEATURE_RANGES` in
  `app.py` (age 18–75, years_waiting 0–10, health_score 1–10, dependents 0–5,
  urgency_score 1–10) replaces the old CSV-derived ranges; `organ_allocation_scenarios.csv`
  is no longer read (or shipped).
- **Rule captions rewritten**: a step's caption now describes only the check itself
  ("You first check if the waiting time gap is 2 years or more" / "Next, you check
  if..."), not the outcome. The outcome ("Choose patient waiting longer" / "Choose
  patient waiting shorter", etc.) lives only on the YES-exit leaf box, matching the
  requested phrasing exactly. Updated in `fft_model.py` (caption templates) and both
  `fft_component.py` + `model.html` (leaf-box labels, in `PARAM_DIRECTION_LABELS` /
  `_PARAM_LABELS_SVG` / `SHORT_LABELS`).
- **"Add a rule" fixed for both YES and NO, tie-breaker branding removed**:
  - The YES-exit leaf's "+ Add a follow-up check" now correctly nests the new check
    under that step's YES branch (previously it silently landed on the NO/fall-through
    chain instead — the reported bug).
  - The main step's own edit panel now has its own separate "+ Add a rule after this
    step" for the NO branch, so both paths have a correct, distinct entry point.
  - "Add tie-breaker" as a separate manually-triggered option is gone; the underlying
    nested-check mechanism is unified under "follow-up check" language everywhere
    (still used to represent auto-trained near-tie checks, and still editable/removable
    from its own panel — just not exposed as a second, confusingly-named button).
- **Found and fixed a real pre-existing bug while testing the above**: the "Factor"
  dropdowns and all "add a rule" code paths were appending `_diff` a second time to
  feature names that already had it (since `params` is sent from the server already
  suffixed, e.g. `"age_diff"`), producing invalid feature keys like `"age_diff_diff"`.
  This meant the Factor dropdown never actually pre-selected a node's real feature when
  you opened its edit panel, and picking a value could silently corrupt the tree. Fixed
  at all 6 call sites in `model.html`.

## Latest addition: scenario audit panel (results page)
A collapsible panel below the tree/rating/actions row, titled "Check when the
model makes the same choice as you":
- A compact chip grid, one per past-answered scenario, colored green (model
  agrees) or red (model disagrees). Click a chip to see that scenario's
  Patient A/B values, your choice, the model's prediction, a match/mismatch
  banner, and a brief note on which step decided it.
- A separate "+ Try a new scenario" entry above the grid opens an editable
  form (five inputs per patient) and shows a live prediction + which step
  fired, without needing to submit anything to the server.
- Runs **entirely client-side** — `evalTreeRow()` in `model.html` is a JS
  port of `FastFrugalTree._row_predict()` in `fft_model.py`, verified to
  match Python bit-for-bit across 160+ randomized test cases (7 different
  trained trees, including one with a manually-added follow-up check) before
  shipping.
- **Disabled while editing** (grayed out with a note), re-enabled once you
  apply your changes — intentionally not live during editing, since a
  manually-added follow-up check only gets normalized into the format this
  evaluator expects after a real server round-trip (`/edit/apply` →
  `train_fft` → `FastFrugalTree.from_dict`), not in its raw just-clicked
  client-side shape.
- No aggregate percentage is shown, per instruction — just the header text
  and the color-coded grid itself.
- Needed one new field passed from the server: `past_scenarios_json` (each
  past decision's A/B feature values + choice, in answered order) and
  `feature_ranges_json`, added to the `/results` route in `app.py`.

## Latest changes (this pass)
- **Restored two unintentional reverts** from an upstream merge: scenario
  generation is CSV-free again (`FEATURE_RANGES` in `app.py`), and the
  "Choose patient X" / "You first check if..." wording is back. Kept all the
  genuinely-new logic from that upload (distinct-scenario sampling, the
  trials/review system) — just pointed it at the right data source.
- **Default (fall-through) leaf reworded**: now reads "Too similar to
  differentiate" in neutral gray, instead of asserting a directional "Choose
  the younger patient" — that leaf firing means no rule found a clear enough
  gap to act on, so the old phrasing overstated the model's confidence. Fixed
  in both renderers (`fft_component.py` for `/final`, `model.html`'s JS for
  `/results`).
- **"Test this model" is now a modal, not an inline section** — the page was
  getting dense, so this moved off the page entirely into an overlay
  triggered by a small button next to Edit/Continue. Two options: **"Check a
  new scenario"** (prefilled with random values, fully editable, "🎲 Fill with
  random values" to re-roll) and **"Check your past answers"** (the existing
  chip grid + detail view, now inside the modal). The picker also shows a live
  "Matches 17 of your 20 answers" preview before you even open that view.
- **Found and fixed a real bug while building the above**: the HTML/CSS for
  this modal had already been scaffolded in a prior pass, but the JavaScript
  wiring it up was never written — it was still pointing at old element IDs
  that no longer existed, which meant the results page was silently crashing
  on load (a null-reference error killed the whole script, including the tree
  renderer). Verified the fix for real by running the rendered page through a
  headless DOM (jsdom) and clicking through every path in the modal, not just
  a syntax check.

## Flow implemented
1. `/`                     — start screen (name entry, feature descriptions)
2. `/questionnaire/1`      — 20 randomly generated pairwise scenarios (dependents, age,
                              years_waiting, urgency_score, health_score)
3. `/results`              — the model, unified view/edit page (see below)
4. `/questionnaire/2`      — 10 more scenarios, then `/final` shows the retrained model

## `/results`: unified view/edit page (`fft/templates/model.html`)
The tree is rendered entirely client-side (ported from the old `index.html` component),
so viewing and editing are the same page/DOM — clicking "✎ Edit the model" just flips a
mode flag and re-renders with edit affordances (pencil icons, dashed borders, reorder
arrows, a toolbar). No page navigation between view and edit.
- The caption above the tree changes text depending on mode.
- Per-node plain-English explanations and the "what you seem to value" summary panel
  (previously only in the Python renderer) are now ported into the JS renderer too, with
  the same dynamic row-height/line-wrap logic, so both stay visually consistent.
- "Apply changes" POSTs the edited tree to `/edit/apply`, which saves it as an override
  and reloads `/results` so stats/explanations are recomputed server-side against the
  real training data.
- A 7-point "How well does this match your thinking?" scale posts to `/results/rate`
  and unlocks "Continue to Part 2 →".

## Fixes from the original visualization
- Condition text now reads in full, e.g. `|Urgency Score(A) - Urgency Score(B)| >= 1`,
  instead of the unexplained `|Δ| >= 1`.
- Leaf/box labels are genuinely short (`_PARAM_LABELS_SVG`) and everything wraps
  automatically instead of overflowing its box.
- The fall-through ("OTHERWISE") leaf now shows the negation of the *last* node's
  condition, instead of an independently-inferred "best separating feature" that could
  contradict the tree's own logic.

## Notes
- `fft_model.py` is unchanged from the Streamlit app — it was already pure/framework-
  agnostic. `fft_component.py` (used for the read-only `/final` page) had the three
  fixes above applied; the JS renderer in `model.html` mirrors the same fixes.
- `prior_transplants` is dropped; this cut only uses 5 features per the current spec.
- State/persistence still uses `users.json` + per-user response CSVs, behind Flask's
  session (signed cookie storing the username).
- LLM explanation calls (`explain_node_llm`, etc.) are still wired in fft_model.py and
  will work as before if GROQ credentials are set; not otherwise touched.
- No CSS framework — styling lives inline per-template as plain CSS, mirroring the
  original COLORS palette, so you have full control instead of fighting a widget library.