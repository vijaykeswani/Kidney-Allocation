// Run by test_explanations.py:  node tests_js/explain_parity.js <cases.json> <twostage_results.html> <tree_results.html>
// Needs jsdom (npm i jsdom). Exits non-zero on any mismatch.
const { JSDOM } = require('jsdom'); const fs = require('fs');
const [CASES, TS_PAGE, TREE_PAGE] = process.argv.slice(2);
const data = JSON.parse(fs.readFileSync(CASES, 'utf8'));
let FAILED = 0;
const load = (f) => { const d = new JSDOM(fs.readFileSync(f, 'utf8'), { runScripts: 'dangerously', url: 'http://localhost/results', pretendToBeVisual: true });
  d.window.fetch = () => Promise.resolve({ json: () => Promise.resolve({}) });
  Object.defineProperty(d.window.HTMLElement.prototype, 'offsetWidth', { get() { return parseInt(this.style.width) || 100; } });
  Object.defineProperty(d.window.HTMLElement.prototype, 'offsetHeight', { get() { return this.className.indexOf('fft-node') !== -1 ? 90 : 46; } });
  d.window.HTMLElement.prototype.scrollIntoView = function() {}; d.window.onerror = () => {}; return d.window; };
const out = []; const log = (s) => { out.push(s); console.log(s); };
setTimeout(() => {
  // ================= two-stage (v2 page) =================
  let w = load(TS_PAGE); const fails = [];
  const AGG = { weighted: 0, tally: 0, prominent: 0 }; let n = 0, shown = 0, thrown = 0;
  for (const c of data.ts) {
    w.eval('model = ' + JSON.stringify(c.model));
    let ex; try { ex = w.explainCase(c.A, c.B, c.choice); w.explainHTML(c.A, c.B, c.choice); } catch (e) { thrown++; fails.push('threw: ' + e.message); continue; }
    n++; AGG[c.model.aggregation]++;
    if (ex.winner !== c.py_winner) fails.push('winner differs from server predict(): ' + JSON.stringify(c.model.aggregation));
    const on = ex.rows.filter(r => r.on), nA = on.filter(r => r.favors === 'A').length, nB = on.filter(r => r.favors === 'B').length;
    const z = on.reduce((s, r) => s + r.d, 0);
    if (ex.agg === 'weighted') { const want = z > 1e-12 ? 'A' : 'B'; if (nA + nB > 0 && ex.winner !== want) fails.push('weighted: winner disagrees with the sign of the summed contributions'); }
    if (ex.agg === 'tally' && nA !== nB && ex.winner !== (nA > nB ? 'A' : 'B')) fails.push('tally: winner disagrees with the vote count');
    if (ex.agg === 'tally' && nA !== nB) { const m = ex.head.match(/\((\d+) vs (\d+)\)/); if (!m || +m[1] !== Math.max(nA, nB) || +m[2] !== Math.min(nA, nB)) fails.push('tally: headline counts are wrong: ' + ex.head); }
    if (ex.agg === 'prominent') { const dec = ex.rows.find(r => r.decides); if (nA + nB > 0 && (!dec || dec.favors !== ex.winner)) fails.push('prominent: decider does not favour the winner'); if (dec) { const amp = (r) => w.ampFor(r.on ? w.model.features[r.key].matters : 0); if (ex.rows.some(r => r.on && r.favors && amp(r) > amp(dec) + 1e-12)) fails.push('prominent: a more important differing factor was skipped'); } }
    if (ex.note && ex.winner === c.choice) fails.push('note shown although the model agrees with the user');
    if (!ex.note && ex.winner !== c.choice) fails.push('no note although the model disagrees with the user');
    if (ex.winner !== c.choice) shown++;
  }
  log(`two-stage: ${n} explanations (${AGG.weighted} weighted / ${AGG.tally} tally / ${AGG.prominent} prominent) over random models with every shape and exceptions`);
  log(`   winner always equals the server's own predict(): ${fails.filter(f => f.startsWith('winner')).length === 0 ? 'yes' : 'NO'} | internal consistency failures: ${fails.filter(f => !f.startsWith('winner')).length} | exceptions thrown: ${thrown} | disagreement notes produced: ${shown}`);
  if (fails.length) log('   FAILURES: ' + [...new Set(fails)].slice(0, 6).join(' || '));
  // sample text
  const ex0 = data.ts.find(c => c.model.aggregation === 'weighted');
  // ================= tree: JS explanation vs the tree's own predict() =================
  w = load(TREE_PAGE); let tn = 0, wrong = 0, evalMismatch = 0, noted = 0;
  for (const c of data.tree) {
    w.eval('tree = ' + JSON.stringify(c.tree));
    const js = w.explainTreeCase(c.A, c.B, c.choice, false); tn++;
    if (js.winner !== c.py_pred) wrong++;
    if (!js.headline.includes('Patient ' + js.winner)) wrong++;
    if ((js.note !== null) !== (js.winner !== c.choice)) wrong++;
    if (js.exit_index >= 0 && !js.steps[js.steps.length - 1].applies) wrong++;
    const ev = w.evalTreeRow(w.diffsFromAB(c.A, c.B)); if ((ev.cls === 1 ? 'A' : 'B') !== js.winner) evalMismatch++;
    if (js.note) noted++;
  }
  log(`tree: ${tn} explanations over ${new Set(data.tree.map(c => JSON.stringify(c.tree))).size} trees (incl. a follow-up-check tree and empty/default-only trees)`);
  log(`   winner always equals the tree's own predict(): ${wrong === 0 ? 'yes' : 'NO (' + wrong + ')'} | differs from the page's own evaluator: ${evalMismatch} | disagreement notes produced: ${noted}`);
  FAILED += fails.length + wrong + evalMismatch;
  console.log(FAILED === 0 ? 'EXPLANATION PARITY OK' : 'EXPLANATION PARITY FAILED');
  process.exit(FAILED === 0 ? 0 : 1);
}, 700);
