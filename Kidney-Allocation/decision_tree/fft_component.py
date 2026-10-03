"""
Interactive Fast-and-Frugal Tree visualisation (SVG renderer).
SURA 2026 · IIT Delhi

Renders a learned/edited FFT as a clean, themed SVG flow diagram. Given a test
pair's difference values it highlights the decision path and the predicted exit,
so editing a threshold (which re-renders this SVG) immediately shows the new
prediction. The accompanying editor widgets and persistence live in app.py.

The renderer is intentionally pure (string in → string out) so it is trivial to
test and reuse outside Streamlit.
"""

import html

import features

# ── Semantic direction labels ─────────────────────────────────────────────────
# Derived from features.json — see features.py. `_PARAM_LABELS_SVG` used to be
# a separately-maintained duplicate of PARAM_DIRECTION_LABELS (used in SVG
# leaf boxes); they were always meant to say the same thing, so they now come
# from the exact same source instead of two dicts that could quietly drift
# apart from each other.
PARAM_DIRECTION_LABELS = features.direction_labels(short=False)
_PARAM_LABELS_SVG = features.direction_labels(short=False)

# Even more compact — used only for the narrow follow-up-check mini-leaf
# boxes (108px, unwrapped single line), which can't fit the fuller
# "Choose ..." phrasing above.
_MINI_PARAM_LABELS_SVG = features.direction_labels(short=True)


def outcome_label(node, short=False):
    """
    Return the outcome label for a tree node's YES exit.
    - If node has use_abs=True: returns a semantic label from PARAM_DIRECTION_LABELS.
    - For legacy signed-diff nodes (use_abs missing/False): infers direction from
      op/threshold/exit_class so a semantic label is always returned.
    `short=True` uses compact single/two-word labels for SVG boxes.
    """
    base = node.get("feature", "").replace("_diff", "")
    if short:
        pair = _PARAM_LABELS_SVG.get(base, ("Higher", "Lower"))
    else:
        pair = PARAM_DIRECTION_LABELS.get(base, ("higher value", "lower value"))

    if node.get("use_abs"):
        prefer_higher = node.get("prefer_higher", True)
    else:
        # Legacy signed-diff node: infer which direction the YES branch favours.
        # op=">=" thr>=0 → fires when A is notably higher → exit_class=1 means prefer A (higher).
        # op="<=" thr<=0 → fires when B is notably higher → exit_class=0 means prefer B (higher).
        op  = node.get("op", ">=")
        thr = float(node.get("threshold", 0))
        ec  = int(node.get("exit_class", 1))
        if op == ">=" and thr >= 0:
            prefer_higher = (ec == 1)
        elif op == "<=" and thr <= 0:
            prefer_higher = (ec == 0)
        else:
            prefer_higher = (ec == 1)

    return pair[0] if prefer_higher else pair[1]


def _fmt_num(x, **_):
    """Round to nearest integer for display."""
    return str(int(round(float(x))))


def _display_name(base):
    """Feature's display name — from features.json's "label" if defined
    there, otherwise mechanically title-cased so an unlisted/legacy feature
    still degrades gracefully instead of erroring."""
    entry = features.FEATURE_BY_KEY.get(base)
    if entry:
        return entry["label"]
    return base.replace("_", " ").title()


def pretty_feature(feature):
    """'urgency_score_diff' -> 'Urgency Score(A) - Urgency Score(B)'."""
    base = feature[:-5] if feature.endswith("_diff") else feature
    name = _display_name(base)
    return f"{name}(A) - {name}(B)"


def _short_feature(feature):
    """Short label for tight boxes: 'urgency_score_diff' -> 'Urgency Score'."""
    base = feature[:-5] if feature.endswith("_diff") else feature
    return _display_name(base)


def _leaf_text(cls):
    """Fallback when no feature context is available."""
    return "Prefer A" if cls == 1 else "Prefer B"


def _refine_branch_label(refine, is_true_branch):
    """
    Semantic label for a tie-breaker refine branch.
    TRUE branch: condition fires → infer who is higher from op/threshold/class.
    FALSE branch: condition doesn't fire → opposite direction.
    Returns an HTML-escaped short label.
    """
    op      = refine.get("op", ">=")
    thr     = float(refine.get("threshold", 0))
    feature = refine.get("feature", "")
    tc      = int(refine.get("true_class", 1))

    # When condition fires: determine who is "higher" on the refine feature
    if op == ">=" and thr > 0:
        ph_true = (tc == 1)    # A is higher when fires, prefer A → prefer_higher
    elif op == "<=" and thr < 0:
        ph_true = (tc == 0)    # B is higher when fires, prefer B → prefer_higher
    else:
        ph_true = (tc == 1)

    prefer_higher = ph_true if is_true_branch else (not ph_true)
    base = feature.replace("_diff", "")
    pair = _MINI_PARAM_LABELS_SVG.get(base, ("Higher", "Lower"))
    label = pair[0] if prefer_higher else pair[1]
    return html.escape(label)


def _node_cond_text(node):
    """
    Full plain condition text for a node, e.g.:
      '|Urgency Score(A) - Urgency Score(B)| ≥ 1'   (abs nodes)
      'Urgency Score(A) - Urgency Score(B) ≥ 1'     (legacy directional nodes)
    """
    val = _fmt_num(node["threshold"])
    feat = pretty_feature(node["feature"])
    if node.get("use_abs"):
        return f"|{feat}| ≥ {val}"
    op_sym = "≥" if node["op"] == ">=" else "≤"
    return f"{feat} {op_sym} {val}"


def _wrap_leaf_lines(text, max_chars=17, max_lines=2):
    """
    Greedy word-wrap for text drawn inside a fixed-width leaf/mini box, so a
    longer-than-expected label (e.g. a custom feature name) degrades to two
    lines instead of overflowing the box.
    """
    words = str(text).split()
    lines, cur = [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if len(trial) <= max_chars or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
        if len(lines) == max_lines - 1 and len(cur) > max_chars:
            break
    if cur:
        lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
    if lines and len(lines[-1]) > max_chars:
        lines[-1] = lines[-1][: max_chars - 1].rstrip() + "…"
    return lines


def _leaf_text_svg(cx, cy, text, fill, font_size=13, font_weight=700, max_chars=17):
    """Centered, auto-wrapped (<=2 lines) SVG <text> for a fixed-width leaf box."""
    lines = _wrap_leaf_lines(text, max_chars=max_chars)
    line_h = font_size + 2
    start_y = cy + font_size * 0.35 - (len(lines) - 1) * line_h / 2
    out = []
    for li, line in enumerate(lines):
        out.append(
            f'<text x="{cx}" y="{start_y + li * line_h}" font-size="{font_size}" '
            f'text-anchor="middle" fill="{fill}" font-weight="{font_weight}">'
            f'{html.escape(line)}</text>'
        )
    return "".join(out)


def _default_outcome_label(tree, short=True):
    """
    Label for the fall-through (default) leaf — shown when nothing above it
    fired. Deliberately non-directional: the model still has to pick someone
    when forced to (see default_class), but asserting "choose the younger
    patient" here overstates the model's confidence in a case where, by
    definition, no rule found a clear enough difference to act on. Framing it
    as "too similar to tell apart" matches what actually happened.
    """
    return "Too similar to differentiate" if short else "the two patients are too similar to differentiate"


def _eval_path(tree, diffs):
    """Return ('exit', node_index, class) or ('default', -1, class)."""
    for i, n in enumerate(tree["nodes"]):
        x = float(diffs.get(n["feature"], 0.0))
        if n.get("use_abs"):
            cond = abs(x) >= n["threshold"]
            if cond:
                refine = n.get("refine")
                if refine:
                    rx = float(diffs.get(refine["feature"], 0.0))
                    if refine.get("use_abs"):
                        rcond = abs(rx) >= refine["threshold"]
                        if rcond:
                            ph = refine.get("prefer_higher", True)
                            cls = 1 if (rx > 0) == ph else 0
                        else:
                            cls = int(refine["false_class"])
                    else:
                        rcond = ((rx >= refine["threshold"]) if refine["op"] == ">="
                                 else (rx <= refine["threshold"]))
                        cls = refine["true_class"] if rcond else refine["false_class"]
                    return "exit", i, cls
                ph = n.get("prefer_higher", True)
                return "exit", i, (1 if (x > 0) == ph else 0)
        else:
            cond = (x >= n["threshold"]) if n["op"] == ">=" else (x <= n["threshold"])
            if cond:
                return "exit", i, n["exit_class"]
    return "default", -1, tree["default_class"]


def fft_svg(tree, palette, test_diffs=None, width=760):
    """
    Build an SVG string for the given tree dict.

    tree       : FastFrugalTree.to_dict()
    palette    : dict with bg, card, border, text, dim, muted, accent, a, b
    test_diffs : optional {feature: difference_value} for the live test pair.
                 When present, the matching path and exit are highlighted and a
                 prediction badge is shown.

    Any node carrying a `refine` sub-node (see fft_model's near-tie tie-breaker)
    is drawn with its primary condition rephrased as a symmetric closeness check
    ("|A − B| ≤ x") — since that's what a near-tie cue actually means in plain
    terms — and its YES branch fans out into a second box, stretched further
    right, that resolves the close call instead of going straight to a leaf.
    """
    p = palette
    nodes = tree["nodes"]
    n = len(nodes)

    # geometry
    pad_top = 70 if test_diffs is not None else 30
    row_h = 122
    node_x, node_w, node_h = 36, 372, 66
    leaf_x, leaf_w, leaf_h = 540, 150, 46
    refine_gap, refine_w, refine_h = 34, 190, 54
    refine_x = leaf_x + leaf_w + refine_gap
    mini_gap, mini_w, mini_h = 26, 108, 34
    mini_x = refine_x + refine_w + mini_gap
    height = pad_top + n * row_h + 150
    out_width = max(width, mini_x + mini_w + 40)

    kind, exit_i, pred_class, refine_branch = (None, -2, None, None)
    if test_diffs is not None:
        kind, exit_i, pred_class, refine_branch = _eval_path_explained(tree, test_diffs)

    def leaf_fill(cls):
        return p["a"] if cls == 1 else p["b"]

    s = []
    s.append(
        f'<svg viewBox="0 0 {out_width} {height}" width="100%" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'font-family="-apple-system,Segoe UI,Roboto,sans-serif">'
    )
    s.append(f'<rect x="0" y="0" width="{out_width}" height="{height}" fill="none"/>')

    # arrow marker
    s.append(
        f'<defs><marker id="ah" markerWidth="9" markerHeight="9" refX="6" refY="3" '
        f'orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="{p["muted"]}"/></marker>'
        f'<marker id="ahx" markerWidth="9" markerHeight="9" refX="6" refY="3" '
        f'orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="{p["accent"]}"/></marker></defs>'
    )

    # prediction badge
    if test_diffs is not None:
        pf = leaf_fill(pred_class)
        pred_lbl = "Prefer A" if pred_class == 1 else "Prefer B"
        s.append(
            f'<rect x="{node_x}" y="18" rx="14" width="300" height="34" '
            f'fill="{pf}" opacity="0.16" stroke="{pf}"/>'
            f'<text x="{node_x + 16}" y="40" font-size="15" font-weight="700" '
            f'fill="{pf}">Prediction:&#160;{pred_lbl}</text>'
        )

    for i, node in enumerate(nodes):
        y = pad_top + i * row_h
        cy = y + node_h / 2
        on_path = (test_diffs is None) or (kind == "default") or (i <= exit_i)
        is_exit = (kind == "exit" and i == exit_i)
        dim = (test_diffs is not None) and (not on_path)
        node_op = 0.32 if dim else 1.0

        box_stroke = p["accent"] if is_exit else p["border"]
        box_sw = 2.5 if is_exit else 1.2
        has_refine = bool(node.get("refine"))

        # condition text: full plain-language check, e.g. "|Age(A) - Age(B)| >= 5"
        if has_refine:
            cond_text = f"|{pretty_feature(node['feature'])}| ≤ {_fmt_num(abs(node['threshold']))}"
        else:
            cond_text = _node_cond_text(node)
        cond_lines = _wrap(cond_text, 46)
        s.append(
            f'<g opacity="{node_op}">'
            f'<rect x="{node_x}" y="{y}" rx="11" width="{node_w}" height="{node_h}" '
            f'fill="{p["card"]}" stroke="{box_stroke}" stroke-width="{box_sw}"/>'
            f'<text x="{node_x + 16}" y="{y + 22}" font-size="13" '
            f'fill="{p["muted"]}" font-weight="600">STEP {i + 1}</text>'
        )
        for cli, cline in enumerate(cond_lines):
            s.append(
                f'<text x="{node_x + 16}" y="{y + 44 + cli * 16}" font-size="13" '
                f'fill="{p["text"]}" font-weight="600" font-family="monospace">'
                f'{html.escape(cline)}</text>'
            )
        s.append('</g>')

        # YES branch -> exit leaf, or (near-tie) the tie-breaker box, to the right
        yes_hl = is_exit
        yes_col = p["accent"] if yes_hl else p["muted"]
        marker = "ahx" if yes_hl else "ah"
        target_x = (refine_x - 6) if has_refine else (leaf_x - 6)
        s.append(
            f'<line x1="{node_x + node_w}" y1="{cy}" x2="{target_x}" y2="{cy}" '
            f'stroke="{yes_col}" stroke-width="{2.4 if yes_hl else 1.3}" '
            f'opacity="{node_op}" marker-end="url(#{marker})"/>'
            f'<text x="{node_x + node_w + 12}" y="{cy - 8}" font-size="12.5" '
            f'fill="{yes_col}" opacity="{node_op}" font-weight="600">YES</text>'
        )

        if not has_refine:
            # For use_abs nodes derive color from prefer_higher so color matches label
            if node.get("use_abs"):
                node_exit_cls = 1 if node.get("prefer_higher", True) else 0
            else:
                node_exit_cls = node.get("exit_class", 1)
            lf = leaf_fill(node_exit_cls)
            leaf_lbl = outcome_label(node, short=True)
            leaf_op = node_op if (test_diffs is None or is_exit or not on_path) else 0.5
            if is_exit:
                leaf_op = 1.0
            s.append(
                f'<g opacity="{leaf_op}">'
                f'<rect x="{leaf_x}" y="{cy - leaf_h / 2}" rx="10" width="{leaf_w}" '
                f'height="{leaf_h}" fill="{lf}" opacity="0.16"/>'
                f'<rect x="{leaf_x}" y="{cy - leaf_h / 2}" rx="10" width="{leaf_w}" '
                f'height="{leaf_h}" fill="none" stroke="{lf}" '
                f'stroke-width="{2.4 if is_exit else 1.4}"/>'
                f'{_leaf_text_svg(leaf_x + leaf_w / 2, cy, leaf_lbl, lf)}'
                f'</g>'
            )
        else:
            # Close call -> one more node, stretched right, that resolves it.
            refine = node["refine"]
            r_true_hit = is_exit and (refine_branch is True)
            r_false_hit = is_exit and (refine_branch is False)
            r_stroke = p["accent"] if is_exit else p["border"]
            r_sw = 2.4 if is_exit else 1.3
            r_op = 1.0 if (test_diffs is None or is_exit or not on_path) else 0.55
            ry = cy - refine_h / 2

            _r_cond = (f'|{pretty_feature(refine["feature"])}| ≥ {_fmt_num(refine["threshold"])}'
                       if refine.get("use_abs") else
                       f'{html.escape(refine.get("op",">="))} {_fmt_num(refine["threshold"])}')
            s.append(
                f'<g opacity="{r_op}">'
                f'<rect x="{refine_x}" y="{ry}" rx="10" width="{refine_w}" height="{refine_h}" '
                f'fill="{p["card"]}" stroke="{r_stroke}" stroke-width="{r_sw}" '
                f'stroke-dasharray="4 3"/>'
                f'<text x="{refine_x + 12}" y="{ry + 18}" font-size="11.5" fill="{p["muted"]}" '
                f'font-weight="700">CLOSE CALL — TIE-BREAKER</text>'
                f'<text x="{refine_x + 12}" y="{ry + 35}" font-size="13" fill="{p["text"]}" '
                f'font-weight="600">{html.escape(pretty_feature(refine["feature"]))}</text>'
                f'<text x="{refine_x + 12}" y="{ry + 50}" font-size="12.5" fill="{p["dim"]}" '
                f'font-family="monospace">{_r_cond}</text></g>'
            )

            true_cy, false_cy = cy - 30, cy + 30
            tf, ff = leaf_fill(refine["true_class"]), leaf_fill(refine["false_class"])
            t_col = p["accent"] if r_true_hit else p["muted"]
            f_col = p["accent"] if r_false_hit else p["muted"]

            s.append(
                f'<line x1="{refine_x + refine_w}" y1="{ry + 12}" x2="{mini_x - 6}" '
                f'y2="{true_cy}" stroke="{t_col}" stroke-width="{2.2 if r_true_hit else 1.2}" '
                f'marker-end="url(#{"ahx" if r_true_hit else "ah"})"/>'
                f'<text x="{refine_x + refine_w + 6}" y="{true_cy - 6}" font-size="11.5" '
                f'fill="{t_col}" font-weight="600">YES</text>'
                f'<g opacity="{1.0 if (test_diffs is None or r_true_hit) else 0.45}">'
                f'<rect x="{mini_x}" y="{true_cy - mini_h / 2}" rx="8" width="{mini_w}" '
                f'height="{mini_h}" fill="{tf}" opacity="0.16"/>'
                f'<rect x="{mini_x}" y="{true_cy - mini_h / 2}" rx="8" width="{mini_w}" '
                f'height="{mini_h}" fill="none" stroke="{tf}" '
                f'stroke-width="{2.2 if r_true_hit else 1.3}"/>'
                f'<text x="{mini_x + mini_w / 2}" y="{true_cy + 4}" font-size="13.5" '
                f'text-anchor="middle" fill="{tf}" font-weight="700">'
                f'{_refine_branch_label(refine, True)}</text></g>'
            )
            s.append(
                f'<line x1="{refine_x + refine_w}" y1="{ry + refine_h - 12}" x2="{mini_x - 6}" '
                f'y2="{false_cy}" stroke="{f_col}" stroke-width="{2.2 if r_false_hit else 1.2}" '
                f'marker-end="url(#{"ahx" if r_false_hit else "ah"})"/>'
                f'<text x="{refine_x + refine_w + 6}" y="{false_cy + 16}" font-size="11.5" '
                f'fill="{f_col}" font-weight="600">NO</text>'
                f'<g opacity="{1.0 if (test_diffs is None or r_false_hit) else 0.45}">'
                f'<rect x="{mini_x}" y="{false_cy - mini_h / 2}" rx="8" width="{mini_w}" '
                f'height="{mini_h}" fill="{ff}" opacity="0.16"/>'
                f'<rect x="{mini_x}" y="{false_cy - mini_h / 2}" rx="8" width="{mini_w}" '
                f'height="{mini_h}" fill="none" stroke="{ff}" '
                f'stroke-width="{2.2 if r_false_hit else 1.3}"/>'
                f'<text x="{mini_x + mini_w / 2}" y="{false_cy + 4}" font-size="13.5" '
                f'text-anchor="middle" fill="{ff}" font-weight="700">'
                f'{_refine_branch_label(refine, False)}</text></g>'
            )

        # NO branch -> down to next node (or default leaf)
        no_y2 = y + row_h
        no_hl = (kind == "exit" and i < exit_i) or (kind == "default")
        no_col = p["accent"] if no_hl else p["muted"]
        no_marker = "ahx" if no_hl else "ah"
        no_op = 1.0 if no_hl else node_op
        s.append(
            f'<line x1="{node_x + 24}" y1="{y + node_h}" x2="{node_x + 24}" '
            f'y2="{no_y2 - 4}" stroke="{no_col}" '
            f'stroke-width="{2.2 if no_hl else 1.3}" opacity="{no_op}" '
            f'marker-end="url(#{no_marker})"/>'
            f'<text x="{node_x + 32}" y="{y + node_h + 24}" font-size="12.5" '
            f'fill="{no_col}" opacity="{no_op}" font-weight="600">NO</text>'
        )

    # default leaf — neutral color, since "too similar to differentiate" is
    # deliberately not asserting an A/B direction the way other leaves do.
    y = pad_top + n * row_h
    df = p["dim"]
    d_hl = (kind == "default")
    d_op = 1.0 if d_hl else (0.4 if test_diffs is not None else 1.0)
    _def_lbl = _default_outcome_label(tree, short=True)
    _def_lines = _wrap_leaf_lines(_def_lbl, max_chars=17)
    def_box_h = max(leaf_h, 40 + len(_def_lines) * 16)
    _def_cx = node_x + 24 + 14
    _def_cy = y + 4 + def_box_h / 2
    _def_line_h = 15
    _def_start_y = _def_cy + 8 - (len(_def_lines) - 1) * _def_line_h / 2
    s.append(
        f'<g opacity="{d_op}">'
        f'<rect x="{node_x + 24 - leaf_w / 2 + 14}" y="{y + 4}" rx="10" '
        f'width="{leaf_w}" height="{def_box_h}" fill="{df}" opacity="0.16"/>'
        f'<rect x="{node_x + 24 - leaf_w / 2 + 14}" y="{y + 4}" rx="10" '
        f'width="{leaf_w}" height="{def_box_h}" fill="none" stroke="{df}" '
        f'stroke-width="{2.4 if d_hl else 1.4}"/>'
        f'<text x="{_def_cx}" y="{y + 18}" font-size="10" text-anchor="middle" '
        f'fill="{df}" opacity="0.75" font-weight="700" letter-spacing=".04em">OTHERWISE</text>'
    )
    for li, line in enumerate(_def_lines):
        s.append(
            f'<text x="{_def_cx}" y="{_def_start_y + li * _def_line_h}" font-size="13.5" '
            f'text-anchor="middle" fill="{df}" font-weight="700">{html.escape(line)}</text>'
        )
    s.append('</g>')

    s.append("</svg>")
    return "".join(s)


# ════════════════════════════════════════════════════════════════════════════
# EXPLAINED RENDERER — summary panel + per-node captions + near-tie tie-breaker
# ════════════════════════════════════════════════════════════════════════════
#
# Companion to fft_svg() above (left untouched for backward compatibility).
# This version is meant to be paired with fft_model.train_fft()'s
# `node_explanations` / `summary_explanation` output, and with any node that
# carries a `refine` sub-node (fft_model's near-tie tie-breaker): instead of
# that node's YES branch going straight to a leaf, it fans out into a second,
# dashed "close call" box drawn further to the right, which itself resolves
# to one of two small leaves. Everything else about a plain node is unchanged.

DEFAULT_FFT_PALETTE = {
    "bg": "#ffffff", "card": "#f8fafc", "border": "#e2e8f0", "text": "#0f172a",
    "dim": "#64748b", "muted": "#94a3b8", "accent": "#2563eb",
    "a": "#b91c1c", "b": "#1d4ed8",
}


def _wrap(text, max_chars):
    """Greedy word-wrap. Returns a list of lines, each <= max_chars (best effort)."""
    if not text:
        return []
    words = text.split()
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if len(trial) <= max_chars:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _eval_path_explained(tree, diffs):
    """Like _eval_path, but also reports which side of a tie-breaker fired.
    Returns (kind, node_index, class, refine_branch) — refine_branch is
    None / True / False."""
    for i, n in enumerate(tree["nodes"]):
        x = float(diffs.get(n["feature"], 0.0))
        if n.get("use_abs"):
            cond = abs(x) >= n["threshold"]
        else:
            cond = (x >= n["threshold"]) if n["op"] == ">=" else (x <= n["threshold"])
        if cond:
            refine = n.get("refine")
            if refine:
                rx = float(diffs.get(refine["feature"], 0.0))
                if refine.get("use_abs"):
                    rcond = abs(rx) >= refine["threshold"]
                    if rcond:
                        ph = refine.get("prefer_higher", True)
                        cls = 1 if (rx > 0) == ph else 0
                    else:
                        cls = int(refine["false_class"])
                else:
                    rcond = ((rx >= refine["threshold"]) if refine["op"] == ">="
                             else (rx <= refine["threshold"]))
                    cls = refine["true_class"] if rcond else refine["false_class"]
                return "exit", i, cls, bool(rcond)
            if n.get("use_abs"):
                ph = n.get("prefer_higher", True)
                cls = 1 if (x > 0) == ph else 0
                return "exit", i, cls, None
            return "exit", i, n["exit_class"], None
    return "default", -1, tree["default_class"], None


def fft_svg_explained(tree, palette=None, node_explanations=None,
                       summary_explanation=None, test_diffs=None, width=900):
    """
    Enhanced SVG renderer. In addition to everything fft_svg() does, this adds:

      * a summary panel at the top ("what you seem to value"), from
        stats['summary_explanation']
      * a short plain-English caption under every node, from
        stats['node_explanations'][i]['explanation']
      * for any node with a `refine` sub-node (a near-tie tie-breaker, see
        fft_model.FastFrugalTree.attach_near_tie_refinements), a dashed
        "CLOSE CALL — TIE-BREAKER" box drawn stretching further to the right
        instead of a single leaf, fanning out to its own two leaves, with its
        own caption from node_explanations[i]['refine_explanation'].

    tree                : FastFrugalTree.to_dict()  (i.e. stats['tree'])
    palette             : optional dict with bg, card, border, text, dim,
                           muted, accent, a, b (defaults to DEFAULT_FFT_PALETTE)
    node_explanations   : stats['node_explanations'] from train_fft() — optional,
                           node captions are simply omitted if not supplied
    summary_explanation : stats['summary_explanation'] from train_fft() — optional
    test_diffs          : optional {feature: difference_value} to highlight the
                           live decision path, same as fft_svg()
    """
    p = palette or DEFAULT_FFT_PALETTE
    nodes = tree["nodes"]
    n = len(nodes)
    node_explanations = node_explanations or [{} for _ in nodes]

    CAP_CHARS, CAP_LINE_H = 44, 17
    RCAP_CHARS, RCAP_LINE_H = 28, 14
    node_x, node_w, node_h_base = 36, 372, 68
    leaf_x, leaf_w, leaf_h = 540, 150, 46
    refine_gap, refine_w, refine_h = 34, 190, 54
    refine_x = leaf_x + leaf_w + refine_gap
    mini_gap, mini_w, mini_h = 26, 108, 34
    mini_x = refine_x + refine_w + mini_gap

    # "WHAT YOU SEEM TO VALUE" summary panel disabled
    # summary_lines = _wrap(summary_explanation, 96) if summary_explanation else []
    # summary_h = (26 + len(summary_lines) * 19 + 20) if summary_lines else 0
    summary_lines = []
    summary_h = 0
    pad_top = (44 if test_diffs is not None else 20)

    kind, exit_i, pred_class, refine_branch = (None, -2, None, None)
    if test_diffs is not None:
        kind, exit_i, pred_class, refine_branch = _eval_path_explained(tree, test_diffs)

    def leaf_fill(cls):
        return p["a"] if cls == 1 else p["b"]

    # pre-compute each row's height (room for wrapped caption + wrapped
    # condition text + refine box)
    row_heights = []
    for i, node in enumerate(nodes):
        exp = node_explanations[i] if i < len(node_explanations) else {}
        cap_lines = _wrap(exp.get("explanation", ""), CAP_CHARS)
        if node.get("refine"):
            _ctext = f"|{pretty_feature(node['feature'])}| ≤ {_fmt_num(abs(node['threshold']))}"
        else:
            _ctext = _node_cond_text(node)
        cond_extra = (len(_wrap(_ctext, 46)) - 1) * 14
        node_h = (node_h_base + (len(cap_lines) * CAP_LINE_H if cap_lines else 0)
                  + max(0, cond_extra))
        refine_extra = 0
        if node.get("refine"):
            rcap_lines = _wrap(exp.get("refine_explanation", ""), RCAP_CHARS)
            eff_rh = max(refine_h, 32 + len(rcap_lines) * RCAP_LINE_H + 26)
            refine_extra = eff_rh + 20
        row_h = max(node_h, refine_extra) + 56
        row_heights.append((row_h, cap_lines, node_h))

    total_h = pad_top + sum(rh for rh, _, _ in row_heights) + 150 + (summary_h + 30 if summary_lines else 0)
    out_width = max(width, mini_x + mini_w + 40)

    s = [
        f'<svg viewBox="0 0 {out_width} {total_h}" width="100%" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'font-family="-apple-system,Segoe UI,Roboto,sans-serif">',
        f'<rect x="0" y="0" width="{out_width}" height="{total_h}" fill="none"/>',
        f'<defs>'
        f'<marker id="ah2" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto">'
        f'<path d="M0,0 L7,3 L0,6 Z" fill="{p["muted"]}"/></marker>'
        f'<marker id="ahx2" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto">'
        f'<path d="M0,0 L7,3 L0,6 Z" fill="{p["accent"]}"/></marker>'
        f'</defs>',
    ]

    if test_diffs is not None:
        pf = leaf_fill(pred_class)
        by = 6
        s.append(
            f'<rect x="{node_x}" y="{by}" rx="14" width="300" height="32" '
            f'fill="{pf}" opacity="0.16" stroke="{pf}"/>'
            f'<text x="{node_x + 16}" y="{by + 21}" font-size="14" font-weight="700" '
            f'fill="{pf}">Prediction:&#160;{_leaf_text(pred_class)}</text>'
        )

    y_cursor = pad_top
    for i, node in enumerate(nodes):
        row_h, cap_lines, node_h = row_heights[i]
        y = y_cursor
        cy = y + node_h / 2
        exp = node_explanations[i] if i < len(node_explanations) else {}

        on_path = (test_diffs is None) or (kind == "default") or (i <= exit_i)
        is_exit_here = (kind == "exit" and i == exit_i)
        dim = (test_diffs is not None) and (not on_path)
        node_op = 0.32 if dim else 1.0

        box_stroke = p["accent"] if is_exit_here else p["border"]
        box_sw = 2.5 if is_exit_here else 1.2
        has_refine = bool(node.get("refine"))
        if has_refine:
            cond_text = f"|{pretty_feature(node['feature'])}| ≤ {_fmt_num(abs(node['threshold']))}"
        else:
            cond_text = _node_cond_text(node)
        cond_lines = _wrap(cond_text, 46)

        s.append(f'<g opacity="{node_op}">')
        s.append(
            f'<rect x="{node_x}" y="{y}" rx="11" width="{node_w}" height="{node_h}" '
            f'fill="{p["card"]}" stroke="{box_stroke}" stroke-width="{box_sw}"/>'
            f'<text x="{node_x + 16}" y="{y + 22}" font-size="13" fill="{p["muted"]}" '
            f'font-weight="600">STEP {i + 1}</text>'
        )
        # Explanation caption first (primary) — then the exact condition below (secondary)
        for li, line in enumerate(cap_lines):
            s.append(f'<text x="{node_x + 16}" y="{y + 38 + li * CAP_LINE_H}" font-size="13.5" '
                     f'fill="{p["text"]}" font-weight="600">{html.escape(line)}</text>')
        _math_y = y + 38 + len(cap_lines) * CAP_LINE_H
        for cli, cline in enumerate(cond_lines):
            s.append(
                f'<text x="{node_x + 16}" y="{_math_y + 14 + cli * 14}" font-size="12.5" '
                f'fill="{p["dim"]}" font-family="monospace">{html.escape(cline)}</text>'
            )
        s.append('</g>')

        # YES branch
        has_refine = bool(node.get("refine"))
        yes_hl = is_exit_here
        yes_col = p["accent"] if yes_hl else p["muted"]
        marker = "ahx2" if yes_hl else "ah2"
        target_x = (refine_x - 6) if has_refine else (leaf_x - 6)
        s.append(
            f'<line x1="{node_x + node_w}" y1="{cy}" x2="{target_x}" y2="{cy}" '
            f'stroke="{yes_col}" stroke-width="{2.4 if yes_hl else 1.3}" '
            f'opacity="{node_op}" marker-end="url(#{marker})"/>'
            f'<text x="{node_x + node_w + 12}" y="{cy - 8}" font-size="12.5" fill="{yes_col}" '
            f'opacity="{node_op}" font-weight="600">YES</text>'
        )

        if not has_refine:
            if node.get("use_abs"):
                _leaf_cls = 1 if node.get("prefer_higher", True) else 0
            else:
                _leaf_cls = node.get("exit_class", 1)
            lf = leaf_fill(_leaf_cls)
            leaf_lbl = outcome_label(node, short=True)
            leaf_op = 1.0 if (test_diffs is None or is_exit_here) else 0.5
            s.append(
                f'<g opacity="{leaf_op}">'
                f'<rect x="{leaf_x}" y="{cy - leaf_h / 2}" rx="10" width="{leaf_w}" '
                f'height="{leaf_h}" fill="{lf}" opacity="0.16"/>'
                f'<rect x="{leaf_x}" y="{cy - leaf_h / 2}" rx="10" width="{leaf_w}" '
                f'height="{leaf_h}" fill="none" stroke="{lf}" '
                f'stroke-width="{2.4 if is_exit_here else 1.4}"/>'
                f'{_leaf_text_svg(leaf_x + leaf_w / 2, cy, leaf_lbl, lf)}'
                f'</g>'
            )
        else:
            refine = node["refine"]
            r_true_hit = is_exit_here and (refine_branch is True)
            r_false_hit = is_exit_here and (refine_branch is False)
            r_stroke = p["accent"] if is_exit_here else p["border"]
            r_sw = 2.4 if is_exit_here else 1.3
            r_op = 1.0 if (test_diffs is None or is_exit_here or not on_path) else 0.55
            rcap_lines = _wrap(exp.get("refine_explanation", ""), RCAP_CHARS)
            eff_rh = max(refine_h, 32 + len(rcap_lines) * RCAP_LINE_H + 26)
            ry = max(y + 4, cy - eff_rh / 2)

            s.append(
                f'<g opacity="{r_op}">'
                f'<rect x="{refine_x}" y="{ry}" rx="10" width="{refine_w}" height="{eff_rh}" '
                f'fill="{p["card"]}" stroke="{r_stroke}" stroke-width="{r_sw}" '
                f'stroke-dasharray="4 3"/>'
                f'<text x="{refine_x + 12}" y="{ry + 18}" font-size="11.5" fill="{p["muted"]}" '
                f'font-weight="700">CLOSE CALL — TIE-BREAKER</text>'
            )
            # Text form first (primary) — then math below (secondary)
            for li, line in enumerate(rcap_lines):
                s.append(f'<text x="{refine_x + 12}" y="{ry + 32 + li * RCAP_LINE_H}" '
                         f'font-size="11.5" fill="{p["text"]}" font-weight="600">'
                         f'{html.escape(line)}</text>')
            _rmath_y = ry + 32 + len(rcap_lines) * RCAP_LINE_H
            _r2_cond = (f'|A − B| ≥ {_fmt_num(refine["threshold"])}'
                        if refine.get("use_abs") else
                        f'{html.escape(refine.get("op",">="))} {_fmt_num(refine["threshold"])}')
            s.append(
                f'<text x="{refine_x + 12}" y="{_rmath_y + 8}" font-size="12" '
                f'fill="{p["muted"]}" font-weight="500">'
                f'{html.escape(pretty_feature(refine["feature"]))}</text>'
                f'<text x="{refine_x + 12}" y="{_rmath_y + 20}" font-size="11.5" '
                f'fill="{p["dim"]}" font-family="monospace">{_r2_cond}</text></g>'
            )

            true_cy, false_cy = cy - 30, cy + 30
            tf, ff = leaf_fill(refine["true_class"]), leaf_fill(refine["false_class"])
            t_col = p["accent"] if r_true_hit else p["muted"]
            f_col = p["accent"] if r_false_hit else p["muted"]

            s.append(
                f'<line x1="{refine_x + refine_w}" y1="{ry + 12}" x2="{mini_x - 6}" y2="{true_cy}" '
                f'stroke="{t_col}" stroke-width="{2.2 if r_true_hit else 1.2}" '
                f'marker-end="url(#{"ahx2" if r_true_hit else "ah2"})"/>'
                f'<text x="{refine_x + refine_w + 6}" y="{true_cy - 6}" font-size="11.5" '
                f'fill="{t_col}" font-weight="600">YES</text>'
                f'<g opacity="{1.0 if (test_diffs is None or r_true_hit) else 0.45}">'
                f'<rect x="{mini_x}" y="{true_cy - mini_h / 2}" rx="8" width="{mini_w}" '
                f'height="{mini_h}" fill="{tf}" opacity="0.16"/>'
                f'<rect x="{mini_x}" y="{true_cy - mini_h / 2}" rx="8" width="{mini_w}" '
                f'height="{mini_h}" fill="none" stroke="{tf}" '
                f'stroke-width="{2.2 if r_true_hit else 1.3}"/>'
                f'<text x="{mini_x + mini_w / 2}" y="{true_cy + 4}" font-size="13.5" '
                f'text-anchor="middle" fill="{tf}" font-weight="700">'
                f'{_refine_branch_label(refine, True)}</text></g>'
            )
            s.append(
                f'<line x1="{refine_x + refine_w}" y1="{ry + eff_rh - 12}" x2="{mini_x - 6}" '
                f'y2="{false_cy}" stroke="{f_col}" stroke-width="{2.2 if r_false_hit else 1.2}" '
                f'marker-end="url(#{"ahx2" if r_false_hit else "ah2"})"/>'
                f'<text x="{refine_x + refine_w + 6}" y="{false_cy + 16}" font-size="11.5" '
                f'fill="{f_col}" font-weight="600">NO</text>'
                f'<g opacity="{1.0 if (test_diffs is None or r_false_hit) else 0.45}">'
                f'<rect x="{mini_x}" y="{false_cy - mini_h / 2}" rx="8" width="{mini_w}" '
                f'height="{mini_h}" fill="{ff}" opacity="0.16"/>'
                f'<rect x="{mini_x}" y="{false_cy - mini_h / 2}" rx="8" width="{mini_w}" '
                f'height="{mini_h}" fill="none" stroke="{ff}" '
                f'stroke-width="{2.2 if r_false_hit else 1.3}"/>'
                f'<text x="{mini_x + mini_w / 2}" y="{false_cy + 4}" font-size="13.5" '
                f'text-anchor="middle" fill="{ff}" font-weight="700">'
                f'{_refine_branch_label(refine, False)}</text></g>'
            )

        # NO branch -> down to next node (or default leaf)
        no_y2 = y + row_h
        no_hl = (kind == "exit" and i < exit_i) or (kind == "default")
        no_col = p["accent"] if no_hl else p["muted"]
        no_marker = "ahx2" if no_hl else "ah2"
        no_op = 1.0 if no_hl else node_op
        s.append(
            f'<line x1="{node_x + 24}" y1="{y + node_h}" x2="{node_x + 24}" y2="{no_y2 - 4}" '
            f'stroke="{no_col}" stroke-width="{2.2 if no_hl else 1.3}" opacity="{no_op}" '
            f'marker-end="url(#{no_marker})"/>'
            f'<text x="{node_x + 32}" y="{y + node_h + 24}" font-size="12.5" fill="{no_col}" '
            f'opacity="{no_op}" font-weight="600">NO</text>'
        )

        y_cursor += row_h

    # default leaf — neutral color, since "too similar to differentiate" is
    # deliberately not asserting an A/B direction the way other leaves do.
    y = y_cursor
    df = p["dim"]
    d_hl = (kind == "default")
    d_op = 1.0 if d_hl else (0.4 if test_diffs is not None else 1.0)
    _def2_lbl = _default_outcome_label(tree, short=True)
    _def2_lines = _wrap_leaf_lines(_def2_lbl, max_chars=17)
    def2_box_h = max(leaf_h, 40 + len(_def2_lines) * 16)
    _def2_cx = node_x + 24 + 14
    _def2_cy = y + 4 + def2_box_h / 2
    _def2_line_h = 15
    _def2_start_y = _def2_cy + 8 - (len(_def2_lines) - 1) * _def2_line_h / 2
    s.append(
        f'<g opacity="{d_op}">'
        f'<rect x="{node_x + 24 - leaf_w / 2 + 14}" y="{y + 4}" rx="10" width="{leaf_w}" '
        f'height="{def2_box_h}" fill="{df}" opacity="0.16"/>'
        f'<rect x="{node_x + 24 - leaf_w / 2 + 14}" y="{y + 4}" rx="10" width="{leaf_w}" '
        f'height="{def2_box_h}" fill="none" stroke="{df}" stroke-width="{2.4 if d_hl else 1.4}"/>'
        f'<text x="{_def2_cx}" y="{y + 18}" font-size="10" text-anchor="middle" '
        f'fill="{df}" opacity="0.75" font-weight="700" letter-spacing=".04em">OTHERWISE</text>'
    )
    for li, line in enumerate(_def2_lines):
        s.append(
            f'<text x="{_def2_cx}" y="{_def2_start_y + li * _def2_line_h}" font-size="13.5" '
            f'text-anchor="middle" fill="{df}" font-weight="700">{html.escape(line)}</text>'
        )
    s.append('</g>')

    # "WHAT YOU SEEM TO VALUE" summary panel disabled
    # if summary_lines:
    #     sy = y + def2_box_h + 30
    #     s.append(
    #         f'<rect x="{node_x}" y="{sy}" rx="10" width="{out_width - node_x * 2}" '
    #         f'height="{summary_h - 10}" fill="{p["card"]}" stroke="{p["border"]}"/>'
    #         f'<text x="{node_x + 16}" y="{sy + 20}" font-size="13.5" font-weight="700" '
    #         f'fill="{p["muted"]}">WHAT YOU SEEM TO VALUE</text>'
    #     )
    #     for li, line in enumerate(summary_lines):
    #         s.append(f'<text x="{node_x + 16}" y="{sy + 40 + li * 19}" font-size="15" '
    #                  f'fill="{p["text"]}">{html.escape(line)}</text>')

    s.append("</svg>")
    return "".join(s)