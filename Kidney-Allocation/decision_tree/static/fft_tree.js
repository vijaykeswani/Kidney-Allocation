/**
 * Top-down decision tree renderer for the preference model.
 * SURA 2026 · IIT Delhi
 *
 * The model is a Fast-and-Frugal Tree: an ordered list of checks where each
 * check either exits to an outcome (YES) or falls through to the next check
 * (NO). This renders it the conventional way — first check at the top, YES
 * branching down-left to its outcome, NO continuing down-right to the next
 * check — using real HTML elements for every box and SVG only for the
 * connector lines between them.
 *
 * Boxes are HTML (not hand-drawn SVG) because: text needs to wrap and its
 * wrapped height needs to actually affect layout, which CSS does natively
 * and SVG does not (the previous version pre-wrapped text by estimating
 * character counts, which is exactly the kind of thing that goes subtly
 * wrong); and native drag-and-drop, focus states and hover states all work
 * far more reliably on real elements than on hand-positioned SVG groups.
 *
 * Layout is a two-pass tidy-tree, same as before: build the boxes first
 * (invisibly, at their fixed width) so the browser can tell us their real
 * rendered height, then measure every subtree's width bottom-up, then place
 * each parent centred over its two children.
 *
 * Usage:
 *     var out = FFTTree.render(containerEl, tree, {
 *       editing: false,
 *       selectedNode: null,
 *       explanations: [{explanation, refine_explanation}, ...],
 *       canAddMore: true,
 *       nodeCount: 3,
 *       helpers: {...}     // see REQUIRED HELPERS below
 *     });
 *     out.layout            // box geometry, for positioning the floating edit panels
 *     out.width, out.height // unscaled canvas size, for zoom/scroll sizing
 *
 * REQUIRED HELPERS (supplied by the caller so label wording stays in one place):
 *     esc(s)                        HTML-escape
 *     fmt(x)                        number -> display string
 *     prettyFeature(f)              "age_diff" -> "Age(A) - Age(B)"
 *     nodeCondText(node)            full condition text for a check
 *     outcomeLabel(node, short)     outcome text for a check's YES exit
 *     refineOutcomeLabel(r, isTrue) outcome text for a tie-breaker branch
 *     defaultOutcomeLabel()         outcome text for the fall-through leaf
 */
(function (global) {
  'use strict';

  // ── geometry ───────────────────────────────────────────────────────────────
  var PAD    = 26;   // canvas margin
  var NODE_W = 268;  // decision box width
  var LEAF_W = 158;  // leaf/outcome box width (also used for refine exits)
  var H_GAP  = 26;   // between sibling subtrees
  var V_GAP  = 56;   // between levels (room for branch labels)

  // References the same design tokens defined in model.html's :root — one
  // definition, not two copies that can drift apart (which is exactly how
  // "edit mode blue" and "Patient B blue" ended up being the same color).
  var C = {
    text: 'var(--text-primary)', card: 'var(--surface)', border: 'var(--border)',
    dim: 'var(--text-muted)', muted: 'var(--text-faint)', accent: 'var(--accent)',
    a: 'var(--patient-a)', b: 'var(--patient-b)'
  };

  function leafFill(cls) { return cls === 1 ? C.a : C.b; }

  // ── model -> binary tree ───────────────────────────────────────────────────
  //
  // Check i:  YES -> outcome (or follow-up sub-check),  NO -> check i+1.
  // The last check's NO branch is the fall-through ("OTHERWISE") outcome.
  // Text is kept as plain strings here — CSS wraps it, so there's no
  // character-count estimation to get wrong.

  function toBinary(tree, H) {
    var nodes = (tree && tree.nodes) || [];

    function defaultLeaf() {
      return {
        kind: 'leaf', role: 'default',
        cls: tree.default_class === 1 ? 1 : 0,
        label: H.defaultOutcomeLabel(),
        eyebrow: 'OTHERWISE'
      };
    }

    function yesBranch(nd, i, exp) {
      if (!nd.refine) {
        var cls = nd.use_abs ? (nd.prefer_higher !== false ? 1 : 0) : nd.exit_class;
        return {kind: 'leaf', role: 'exit', nodeIndex: i, cls: cls, label: H.outcomeLabel(nd, true)};
      }
      var r = nd.refine;
      return {
        kind: 'decision', role: 'refine', nodeIndex: i,
        eyebrow: 'FOLLOW-UP CHECK',
        caption: (exp && exp.refine_explanation) || 'Closer look',
        cond: '|' + H.prettyFeature(r.feature) + '| ' + (r.op === '<=' ? '\u2264' : '\u2265') +
              ' ' + H.fmt(Math.abs(r.threshold)),
        yes: {kind: 'leaf', role: 'refine-exit', nodeIndex: i, branch: true,
              cls: r.true_class, label: H.refineOutcomeLabel(r, true)},
        no:  {kind: 'leaf', role: 'refine-exit', nodeIndex: i, branch: false,
              cls: r.false_class, label: H.refineOutcomeLabel(r, false)}
      };
    }

    function checkAt(i) {
      if (i >= nodes.length) return defaultLeaf();
      var nd = nodes[i];
      var exp = (H.explanations && H.explanations[i]) || {};
      var condText = nd.refine
        ? ('|' + H.prettyFeature(nd.feature) + '| ' +
           (nd.op === '<=' ? '\u2264' : '\u2265') + ' ' + H.fmt(Math.abs(nd.threshold)))
        : H.nodeCondText(nd);
      return {
        kind: 'decision', role: 'check', nodeIndex: i,
        eyebrow: 'STEP ' + (i + 1),
        caption: exp.explanation || '',
        cond: condText,
        yes: yesBranch(nd, i, exp),
        no: checkAt(i + 1)
      };
    }

    return nodes.length ? checkAt(0) : defaultLeaf();
  }

  // ── build DOM elements (invisible, fixed width, natural height) ────────────

  function makeBadge(cls, glyph, title, dataAttrs) {
    var btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'fft-badge ' + cls;
    btn.textContent = glyph;
    if (title) btn.title = title;
    if (dataAttrs) {
      Object.keys(dataAttrs).forEach(function (k) { btn.setAttribute(k, dataAttrs[k]); });
    }
    return btn;
  }

  function buildLeafEl(n, opts) {
    var editing = opts.editing;
    var el = document.createElement('div');
    var f = n.role === 'default' ? C.dim : leafFill(n.cls);
    el.className = 'fft-leaf' + (editing ? ' fft-editing' : '');
    el.style.width = LEAF_W + 'px';
    el.style.color = f;
    el.style.background = f + '26';
    el.style.borderColor = f;
    if (n.role === 'exit') el.setAttribute('data-leaf', n.nodeIndex);
    else if (n.role === 'default') el.id = 'default-leaf';
    else if (n.role === 'refine-exit') el.setAttribute('data-refine', n.nodeIndex);

    if (n.eyebrow) {
      var eyebrow = document.createElement('div');
      eyebrow.className = 'fft-leaf-eyebrow';
      eyebrow.textContent = n.eyebrow;
      el.appendChild(eyebrow);
    }
    var label = document.createElement('div');
    label.className = 'fft-leaf-label';
    label.textContent = n.label;
    el.appendChild(label);

    if (editing) {
      el.appendChild(makeBadge('fft-badge-pencil fft-badge-corner', '\u270E', 'Change this outcome'));
    }
    return el;
  }

  function buildDecisionEl(n, opts) {
    var editing = opts.editing;
    var isRefine = n.role === 'refine';
    var sel = editing && !isRefine && opts.selectedNode === n.nodeIndex;
    var el = document.createElement('div');
    el.className = 'fft-node' +
      (editing ? ' fft-editing' : '') +
      (sel ? ' fft-selected' : '') +
      (isRefine ? ' fft-refine' : '');
    el.style.width = NODE_W + 'px';
    el.setAttribute(isRefine ? 'data-refine' : 'data-node', n.nodeIndex);

    var eyebrow = document.createElement('div');
    eyebrow.className = 'fft-eyebrow';
    eyebrow.textContent = n.eyebrow;
    el.appendChild(eyebrow);

    if (n.caption) {
      var cap = document.createElement('div');
      cap.className = 'fft-caption';
      cap.textContent = n.caption;
      el.appendChild(cap);
    }
    var cond = document.createElement('div');
    cond.className = 'fft-cond';
    // n.cond is built entirely from our own template strings (prettyFeature
    // wraps the factor name in a span so it can be color-highlighted) — no
    // free-form user text ever ends up in here, so innerHTML is safe.
    cond.innerHTML = n.cond;
    el.appendChild(cond);

    if (isRefine) {
      if (editing) {
        el.appendChild(makeBadge('fft-badge-pencil fft-badge-corner', '\u270E',
          'Edit this follow-up check', {'data-refine': n.nodeIndex}));
      }
    } else if (editing) {
      var i = n.nodeIndex;
      el.appendChild(makeBadge('fft-badge-pencil fft-badge-tl', '\u270E',
        'Edit this check', {'data-node': i}));
      if (opts.nodeCount && opts.nodeCount > 1) {
        el.appendChild(makeBadge('fft-badge-delete fft-badge-tl2', '\uD83D\uDDD1', 'Delete this check',
          {'data-delete-node': i}));
      }
      var addCls = 'fft-badge-add fft-badge-br' + (opts.canAddMore === false ? ' fft-disabled' : '');
      el.appendChild(makeBadge(addCls, '+',
        opts.canAddMore === false ? 'Limit reached — you can add two checks at a time'
                                   : 'Add a check straight after this one',
        {'data-add-after': i}));
      if (opts.nodeCount && opts.nodeCount > 1) {
        el.appendChild(makeBadge('fft-badge-grip fft-badge-tr3', '\u283F', 'Drag to reorder',
          {'data-drag-handle': i}));
      }
      if (i > 0) {
        el.appendChild(makeBadge('fft-badge-move fft-badge-tr2', '\u25B2', 'Move this check earlier',
          {'data-move-up': i}));
      }
      if (opts.nodeCount && i < opts.nodeCount - 1) {
        el.appendChild(makeBadge('fft-badge-move fft-badge-tr1', '\u25BC', 'Move this check later',
          {'data-move-down': i}));
      }
    }
    return el;
  }

  function buildEl(n, opts) {
    n.el = n.kind === 'leaf' ? buildLeafEl(n, opts) : buildDecisionEl(n, opts);
    n.el.style.position = 'absolute';
    n.el.style.visibility = 'hidden';
    n.el.style.left = '0px';
    n.el.style.top = '0px';
    if (n.kind === 'decision') { buildEl(n.yes, opts); buildEl(n.no, opts); }
  }

  function appendAll(n, container) {
    container.appendChild(n.el);
    if (n.kind === 'decision') { appendAll(n.yes, container); appendAll(n.no, container); }
  }

  // ── measure / place (tidy-tree, same two-pass algorithm as before) ─────────

  function measure(n) {
    n.w = n.el.offsetWidth;
    n.h = n.el.offsetHeight;
    if (n.kind === 'leaf') { n.subW = n.w; return; }
    measure(n.yes);
    measure(n.no);
    n.subW = Math.max(n.w, n.yes.subW + H_GAP + n.no.subW);
  }

  function place(n, left, depth, rowH) {
    n.depth = depth;
    rowH[depth] = Math.max(rowH[depth] || 0, n.h);
    if (n.kind === 'leaf') { n.cx = left + n.subW / 2; return; }
    place(n.yes, left, depth + 1, rowH);
    place(n.no, left + n.yes.subW + H_GAP, depth + 1, rowH);
    n.cx = (n.yes.cx + n.no.cx) / 2;
  }

  function assignY(n, rowY) {
    n.y = rowY[n.depth];
    n.x = n.cx - n.w / 2;
    if (n.kind === 'decision') { assignY(n.yes, rowY); assignY(n.no, rowY); }
  }

  function positionAll(n, layout) {
    n.el.style.left = n.x + 'px';
    n.el.style.top = n.y + 'px';
    n.el.style.visibility = '';
    if (n.role === 'exit') layout.leaves[n.nodeIndex] = {x: n.x, y: n.y, w: n.w, h: n.h};
    else if (n.role === 'default') layout.defaultLeaf = {x: n.x, y: n.y, w: n.w, h: n.h};
    else if (n.role === 'refine') layout.refines[n.nodeIndex] = {x: n.x, y: n.y, w: n.w, h: n.h};
    else if (n.role === 'check') layout.nodes[n.nodeIndex] = {x: n.x, y: n.y, w: n.w, h: n.h, cx: n.cx};
    if (n.kind === 'decision') { positionAll(n.yes, layout); positionAll(n.no, layout); }
  }

  // ── connectors (SVG, drawn after boxes are positioned) ──────────────────────

  function elbow(x1, y1, x2, y2) {
    var mid = y1 + (y2 - y1) / 2;
    return 'M' + x1 + ',' + y1 + ' L' + x1 + ',' + mid +
           ' L' + x2 + ',' + mid + ' L' + x2 + ',' + (y2 - 5);
  }

  function svgEl(tag, attrs) {
    var el = document.createElementNS('http://www.w3.org/2000/svg', tag);
    Object.keys(attrs).forEach(function (k) { el.setAttribute(k, attrs[k]); });
    return el;
  }

  function drawEdges(svg, n) {
    if (n.kind !== 'decision') return;
    [['yes', n.yes, 'YES'], ['no', n.no, 'NO']].forEach(function (pair) {
      var child = pair[1];
      var x1 = n.cx, y1 = n.y + n.h, x2 = child.cx, y2 = child.y;
      svg.appendChild(svgEl('path', {
        d: elbow(x1, y1, x2, y2), fill: 'none', stroke: C.muted,
        'stroke-width': '1.4', 'marker-end': 'url(#fft-ah)'
      }));
      var lx = x2 + (x2 < x1 ? -10 : 10);
      var anchor = x2 < x1 ? 'end' : 'start';
      if (Math.abs(x2 - x1) < 4) { lx = x1 + 10; anchor = 'start'; }
      var label = svgEl('text', {
        x: lx, y: y1 + (y2 - y1) / 2 - 6, 'font-size': '11.5', 'text-anchor': anchor,
        fill: C.muted, 'font-weight': '700', 'letter-spacing': '.05em'
      });
      label.textContent = pair[2];
      svg.appendChild(label);
    });
    drawEdges(svg, n.yes);
    drawEdges(svg, n.no);
  }

  // ── public API ─────────────────────────────────────────────────────────────

  function render(container, tree, opts) {
    opts = opts || {};
    var H = opts.helpers || {};
    H.explanations = opts.explanations || [];

    var root = toBinary(tree, H);

    container.innerHTML = '';
    var wrap = document.createElement('div');
    wrap.className = 'fft-canvas';
    container.appendChild(wrap);

    var svg = svgEl('svg', {xmlns: 'http://www.w3.org/2000/svg'});
    svg.setAttribute('class', 'fft-lines');
    var defs = svgEl('defs', {});
    var marker = svgEl('marker', {
      id: 'fft-ah', markerWidth: '9', markerHeight: '9', refX: '6', refY: '3', orient: 'auto'
    });
    marker.appendChild(svgEl('path', {d: 'M0,0 L7,3 L0,6 Z', fill: C.muted}));
    defs.appendChild(marker);
    svg.appendChild(defs);
    wrap.appendChild(svg);

    // Build every box invisibly at its fixed width so the browser tells us
    // its real (wrapped) height, then lay out using those real heights.
    buildEl(root, opts);
    appendAll(root, wrap);

    measure(root);
    var rowH = [];
    place(root, PAD, 0, rowH);
    var rowY = [], acc = PAD;
    for (var d = 0; d < rowH.length; d++) { rowY[d] = acc; acc += rowH[d] + V_GAP; }
    assignY(root, rowY);

    var W = root.subW + PAD * 2;
    var H_TOTAL = acc - V_GAP + PAD;
    wrap.style.width = W + 'px';
    wrap.style.height = H_TOTAL + 'px';
    svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H_TOTAL);
    svg.setAttribute('width', W);
    svg.setAttribute('height', H_TOTAL);

    var layout = {nodes: {}, leaves: {}, refines: {}, defaultLeaf: null, width: W, height: H_TOTAL};
    positionAll(root, layout);
    drawEdges(svg, root);

    return {layout: layout, width: W, height: H_TOTAL};
  }

  global.FFTTree = {render: render, COLORS: C};
})(window);