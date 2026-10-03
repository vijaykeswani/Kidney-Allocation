"""
Decision Tree (Fast-and-Frugal Tree) method: everything specific to the decision-tree model.

    handlers.py        route handlers called from app.py's dispatch points
    fft_model.py       training, prediction and explanations for the tree
    fft_component.py   the SVG drawing of the tree used on the final page
    tiebreaker.py      optional tie-breaker chain for near-tie cases
    templates/         model.html (editor), review.html, final.html
    static/            fft_tree.js (tree layout and drawing for the editor)

Shared pages (consent, start, questionnaire, not_ready) and the top-level
route dispatchers stay in app.py. Decision-tree-specific logic lives here.

The blueprint makes this folder's templates and static files visible to the app.
Template names are unchanged, so `render_template("model.html")` still works;
static files are served under /decision_tree/static/.
"""

from flask import Blueprint

bp = Blueprint("decision_tree", __name__, template_folder="templates",
               static_folder="static", static_url_path="/decision_tree/static")
