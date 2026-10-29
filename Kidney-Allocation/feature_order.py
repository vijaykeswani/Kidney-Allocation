"""
Per-participant feature order, counterbalanced.

Each participant sees the factors in ONE order for the whole study (same on every scenario, in the model views, in the
case browser...). Orders are assigned so that across participants every ordering is used about equally often:

  1. pick among the orderings used LEAST so far (so counts never differ by more than 1), then
  2. among those, prefer the one that best evens out how often each factor has appeared in each position
     (so even before all 120 orderings have been used, no factor is stuck near the top or bottom), then
  3. break remaining ties at random.

Only PRESENTATION order varies. Models, the database and the exported CSVs always use the canonical order
(features.FEATURE_KEYS), so nothing about analysis changes.
"""
import itertools
import random
from collections import Counter

import db
import features


def all_orders():
    return [list(p) for p in itertools.permutations(features.FEATURE_KEYS)]


def choose(existing_orders, rng=None):
    """The next order to hand out, given the orders already assigned (a list of key lists)."""
    rng = rng or random.SystemRandom()
    perms = all_orders()
    valid = {tuple(p) for p in perms}
    existing = [tuple(o) for o in existing_orders if tuple(o) in valid]
    used = Counter(existing)
    pos = Counter((f, i) for o in existing for i, f in enumerate(o))
    lowest = min(used.get(tuple(p), 0) for p in perms)
    candidates = [p for p in perms if used.get(tuple(p), 0) == lowest]
    score = lambda p: sum(pos[(f, i)] for i, f in enumerate(p))
    best = min(score(p) for p in candidates)
    return rng.choice([p for p in candidates if score(p) == best])


def for_user(username):
    """This participant's order, assigning one (atomically) the first time."""
    return db.get_feature_order(username) or db.assign_feature_order(username, choose)
