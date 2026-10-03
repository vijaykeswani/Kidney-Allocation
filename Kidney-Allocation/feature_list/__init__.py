"""
Feature List (two-stage) method: everything specific to the feature-list model.

    handlers.py        route handlers called from app.py's dispatch points
    twostage_model.py  per-feature monotonic rules, combined by a fixed
                       aggregation rule (weighted sum, tally, or most
                       important factor)
    templates/         twostage.html (model page and final page)

Shared pages (consent, start, questionnaire, not_ready) and the top-level
route dispatchers stay in app.py. Feature-list-specific logic lives here.

The blueprint makes this folder's templates visible to the app, so
`render_template("twostage.html")` works unchanged.
"""

from flask import Blueprint

bp = Blueprint("feature_list", __name__, template_folder="templates")
