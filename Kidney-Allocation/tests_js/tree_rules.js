// Run by test_tree_rules.py:  node tests_js/tree_rules.js <tree_results.html>     (needs jsdom)
// The decision-tree rules are sentences whose blue drop-downs edit the tree directly: this checks the sentences, their
// options (they must match the factors' real values), that a change reaches the tree, and the read-only version.
const { JSDOM } = require('jsdom'); const fs = require('fs');
const w = (() => { const d = new JSDOM(fs.readFileSync(process.argv[2], 'utf8'), { runScripts: 'dangerously', url: 'http://localhost/results', pretendToBeVisual: true });
  d.window.fetch = () => Promise.resolve({ json: () => Promise.resolve({}) });
  Object.defineProperty(d.window.HTMLElement.prototype, 'offsetWidth', { get() { return parseInt(this.style.width) || 100; } });
  Object.defineProperty(d.window.HTMLElement.prototype, 'offsetHeight', { get() { return this.className.indexOf('fft-node') !== -1 ? 90 : 46; } });
  d.window.HTMLElement.prototype.scrollIntoView = function() {}; d.window.onerror = () => {}; return d.window; })();
const fails = []; const check = (ok, msg) => { if (!ok) fails.push(msg); };
setTimeout(() => {
  const $$ = (s) => Array.from(w.document.querySelectorAll(s));
  const change = (sel, value) => { sel.value = value; sel.dispatchEvent(new w.Event('change', { bubbles: true })); };
  const text = () => $$('.fft-node').map(e => e.textContent).join(' ');

  // 1. rules are sentences with drop-downs; no equations anywhere
  check($$('.fft-rule').length === 3, `expected 2 rules + 1 follow-up as sentences, found ${$$('.fft-rule').length}`);
  check(!/\|/.test(text()) && !/\((A|B)\)/.test(text()), 'an equation like |factor(A) - factor(B)| is still shown');
  const rules = $$('.fft-rule').map(e => e.textContent);
  check(rules.every(t => /the gap in/.test(t) && /between A & B/.test(t)), 'the rule is not written as "the gap in X between A & B is at least N"');
  check(rules.every(t => !/Patient/.test(t) && !/choose the patient/i.test(t)), 'the rule sentence should not mention "Patient" or say who is chosen (the outcome box does)');
  check($$('select.rule-sel').length === 8, `expected 8 drop-downs (factor + gap on 3 rules, plus an outcome drop-down on 2 YES boxes), got ${$$('select.rule-sel').length}`);
  check($$('.fft-leaf select.rule-sel[data-field="dir"]').length === 2, 'the YES outcome boxes should carry the "which patient" drop-down');
  check($$('.fft-rule select.rule-sel[data-field="dir"]').length === 0, 'the sentence itself should not contain the "which patient" drop-down');
  $$('select.rule-sel').forEach(sel => { const z = sel.parentNode.querySelector('.rule-sel-size'); check(z && z.textContent === sel.options[sel.selectedIndex].textContent && !sel.style.width, 'every drop-down must be sized by a hidden copy of its selected text (never by a measured width, which can be wrong)'); });
  ['#edit-panel', '#leaf-panel', '#refine-panel'].forEach(id => check(!w.document.querySelector(id), id + ' should be gone'));

  // 2. the size-of-gap options are the gaps that can really occur between two patients on that factor (no more, no fewer)
  $$('select.rule-sel[data-field="threshold"]').forEach(sel => {
    const i = +sel.dataset.i, kind = sel.dataset.rule, nd = w.tree.nodes[i], r = kind === 'refine' ? nd.refine : nd;
    const vs = w.FEATURE_BY_KEY[r.feature.replace('_diff', '')].values, want = new Set();
    vs.forEach(a => vs.forEach(b => { if (b > a) want.add(b - a); })); want.add(Math.round(Math.abs(r.threshold)));
    const got = Array.from(sel.options).map(o => +o.value).sort((x, y) => x - y), exp = [...want].sort((x, y) => x - y);
    check(JSON.stringify(got) === JSON.stringify(exp), `gap options for ${r.feature}: got ${got} expected ${exp}`);
    check(+sel.value === Math.round(Math.abs(r.threshold)), `the selected gap should be the rule's own (${r.threshold})`);
  });
  check(Array.from(w.document.querySelector('select.rule-sel[data-field="feature"]').options).length === w.params.length, 'the factor list should offer every factor');

  // 2b. the outcome drop-down offers the two readable outcomes for that rule's factor, and shows the rule's current one
  $$('.fft-leaf select.rule-sel[data-field="dir"]').forEach(sel => {
    const i = +sel.dataset.i, r = sel.dataset.rule === 'refine' ? w.tree.nodes[i].refine : w.tree.nodes[i], sp = w.SHORT_LABELS[r.feature.replace('_diff', '')];
    check(JSON.stringify(Array.from(sel.options).map(o => o.textContent)) === JSON.stringify(sp), `outcome options for ${r.feature}: ${Array.from(sel.options).map(o => o.textContent)} vs ${sp}`);
    check(sel.value === (r.prefer_higher !== false ? 'higher' : 'lower'), 'the outcome drop-down should show the rule\'s own choice');
  });

  // 3. a change in a drop-down reaches the tree
  let sel = w.document.querySelector('select.rule-sel[data-i="1"][data-field="feature"]'); change(sel, 'dependents_diff');
  check(w.tree.nodes[1].feature === 'dependents_diff', 'changing the factor did not reach the tree');
  check([1, 2, 3].includes(w.tree.nodes[1].threshold), `after changing the factor the gap must be one that exists for it (got ${w.tree.nodes[1].threshold})`);
  sel = w.document.querySelector('select.rule-sel[data-i="1"][data-field="threshold"]'); change(sel, '2');
  check(w.tree.nodes[1].threshold === 2, 'changing the gap did not reach the tree');
  sel = w.document.querySelector('select.rule-sel[data-i="1"][data-field="dir"]'); const was = w.tree.nodes[1].prefer_higher !== false; change(sel, was ? 'lower' : 'higher');
  check((w.tree.nodes[1].prefer_higher !== false) === !was && w.tree.nodes[1].exit_class === (was ? 0 : 1), 'changing which patient is chosen did not reach the tree');
  sel = w.document.querySelector('select.rule-sel[data-rule="refine"][data-field="dir"]'); change(sel, 'lower');
  const rf = w.tree.nodes[0].refine; check(rf.prefer_higher === false && rf.true_class === 0 && rf.false_class === 1, 'the follow-up check direction did not reach the tree');
  check(/dependents/i.test(w.document.querySelector('.fft-node[data-node="1"] .fft-rule').textContent), 'the sentence did not update after the change');

  // 4. follow-up checks can be removed, but there is no "add" for now
  check(!w.document.querySelector('[data-add-followup]'), 'there should be no "add a follow-up check" link for now');
  w.document.querySelector('[data-remove-followup]').click(); check(!w.tree.nodes[0].refine, 'removing the follow-up check did not work');
  check(!w.document.querySelector('[data-add-followup]') && !/follow-up check/i.test(text().replace(/follow-up check below/g, '')), 'removing a follow-up must not leave an "add" link behind');
  w.document.querySelector('.fft-node[data-node="1"] [data-add-after]').click();
  check(w.tree.nodes.length === 3 && w.tree.nodes[2].threshold > 0, 'a rule added with + needs a real starting gap (not 0)');

  // 5. read-only (the final page): the same sentences, no drop-downs
  w.eval('editing = false; render();');
  check($$('select.rule-sel').length === 0 && $$('.rule-val').length > 0, 'the read-only tree must show values, not drop-downs');
  check(!/\|/.test(text()), 'an equation appeared in the read-only tree');
  console.log(fails.length ? 'TREE RULES FAILED: ' + fails.join(' || ') : 'tree rules: sentences, drop-down options, edits reaching the tree, follow-up checks, read-only view');
  process.exit(fails.length ? 1 : 0);
}, 400);
