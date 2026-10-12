// Crease Check front end. Plain JS, no build step: it calls the API and
// draws the answer as SVG.

const $ = (id) => document.getElementById(id);
const NS = 'http://www.w3.org/2000/svg';
const PAD = 20, SIDE = 360;            // the sheet inside the 400x400 viewBox
const NAMES = { kawasaki: 'Kawasaki', maekawa: 'Maekawa', big_little_big: 'Big-little-big',
  even_degree: 'Odd number of creases', unassigned: 'Unassigned creases', layers: 'Layers collide' };

let current = null;                    // the last API response

// --- talking to the API ------------------------------------------------------

async function call(url, options = {}) {
  showError(null);
  $('busy').hidden = false;
  try {
    const res = await fetch(url, options);
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(typeof body.detail === 'string' ? body.detail : `Request failed (${res.status}).`);
    return body;
  } finally {
    $('busy').hidden = true;
  }
}

function showError(msg) {
  $('error').hidden = !msg;
  $('error').textContent = msg || '';
}

// --- drawing -----------------------------------------------------------------

function el(name, attrs = {}, parent) {
  const node = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (parent) parent.appendChild(node);
  return node;
}

function projector(coords) {
  const xs = coords.map((c) => c[0]), ys = coords.map((c) => c[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const s = Math.max(x1 - x0, y1 - y0) || 1;
  const ox = (s - (x1 - x0)) / 2, oy = (s - (y1 - y0)) / 2;
  // Patterns are y-up; SVG is y-down.
  return ([x, y]) => [PAD + SIDE * (x - x0 + ox) / s, PAD + SIDE * (1 - (y - y0 + oy) / s)];
}

function groupByVertex(violations) {
  const spots = new Map();
  for (const v of violations) {
    if (!spots.has(v.vertex)) spots.set(v.vertex, { vertex: v.vertex, point: v.point, items: [], fix: null, fixText: null });
    const s = spots.get(v.vertex);
    s.items.push(v);
    if (!s.fix && v.fix) { s.fix = v.fix; s.fixText = v.fix_text; }
  }
  return [...spots.values()];
}

function render(body, { fromPhoto = false } = {}) {
  current = body;
  const svg = $('canvas');
  svg.replaceChildren();
  const fold = body.pattern;
  const coords = fold.vertices_coords;
  if (!coords.length) return;
  const P = projector(coords);
  const root = el('g', { class: body.flat_foldable ? 'passed' : '' }, svg);

  const paper = el('rect', { x: PAD, y: PAD, width: SIDE, height: SIDE, class: 'paper' + (fromPhoto ? ' see-through' : '') }, root);
  const glows = el('g', {}, root), lines = el('g', {}, root);
  fold.edges_vertices.forEach(([a, b], k) => {
    const kind = fold.edges_assignment[k];
    const [x1, y1] = P(coords[a]), [x2, y2] = P(coords[b]);
    const cls = { M: 'ln-m', V: 'ln-v', B: 'ln-b' }[kind] || 'ln-u';
    if (kind === 'M' || kind === 'V') el('line', { x1, y1, x2, y2, class: kind === 'M' ? 'glow-m' : 'glow-v' }, glows);
    const ln = el('line', { x1, y1, x2, y2, class: cls }, lines);
    if (fromPhoto && kind !== 'B') {
      ln.setAttribute('pathLength', '1');
      ln.classList.add('draw');
      ln.style.animationDelay = `${(k % 12) * 40}ms`;
      ln.addEventListener('animationend', () => { ln.classList.remove('draw'); ln.removeAttribute('pathLength'); }, { once: true });
    }
  });

  const spots = groupByVertex(body.violations);
  const marks = el('g', {}, root);
  spots.forEach((s, i) => {
    const [x, y] = P(s.point);
    const g = el('g', { class: 'spot', 'data-n': i + 1 }, marks);
    el('circle', { cx: x, cy: y, r: 15, class: 'ring' }, g);
    const lbl = el('text', { x: x + 18, y: y - 12, class: 'ring-label' }, g);
    lbl.textContent = i + 1;
    g.addEventListener('mouseenter', () => highlight(i + 1, true));
    g.addEventListener('mouseleave', () => highlight(i + 1, false));
  });

  if (fromPhoto) {
    requestAnimationFrame(() => requestAnimationFrame(() => {
      paper.classList.remove('see-through');
      $('photo').classList.add('fade');
    }));
  } else {
    $('photo').hidden = true;
  }

  // Verdict and list.
  const n = spots.length;
  const verdict = body.checked.creases === 0 ? 'Nothing to check'
    : body.flat_foldable ? 'Folds flat' : `${n} point${n === 1 ? '' : 's'} won't fold`;
  $('verdict').textContent = verdict;
  const ms = body.photo ? `photo read in ${body.photo.read_ms} ms` : `${Math.max(1, Math.round(body.elapsed_ms))} ms`;
  $('meta').textContent = `${body.checked.vertices} points checked · ${body.checked.creases} creases · ${ms}`;
  const note = body.flat_foldable ? 'Every point folds flat on its own. See How it works for what that does and does not promise.' : '';
  $('note').hidden = !note;
  $('note').textContent = note;

  const list = $('issues');
  list.replaceChildren();
  spots.forEach((s, i) => {
    const li = document.createElement('li');
    li.className = 'issue';
    li.dataset.n = i + 1;
    const num = document.createElement('span');
    num.className = 'num';
    num.textContent = i + 1;
    const what = document.createElement('div');
    what.className = 'what';
    const title = document.createElement('b');
    title.textContent = [...new Set(s.items.map((v) => NAMES[v.rule] || v.rule))].join(' · ') + ' ';
    const where = document.createElement('span');
    where.textContent = `at (${s.point[0].toFixed(2)}, ${s.point[1].toFixed(2)})`;
    title.appendChild(where);
    what.appendChild(title);
    for (const v of s.items) {
      const p = document.createElement('p');
      p.textContent = v.message;
      what.appendChild(p);
    }
    li.append(num, what);
    if (s.fix) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'fix';
      btn.innerHTML = '<svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24"><path d="M5 12h14"/><path d="M13 6l6 6-6 6"/></svg>';
      btn.append(document.createTextNode(s.fixText));
      btn.title = 'Apply this fix and check again';
      btn.addEventListener('click', () => applyFix(s));
      li.appendChild(btn);
    }
    li.addEventListener('mouseenter', () => highlight(i + 1, true));
    li.addEventListener('mouseleave', () => highlight(i + 1, false));
    list.appendChild(li);
  });

  $('json').textContent = JSON.stringify(trimForDisplay(body), null, 2);
}

function highlight(n, on) {
  document.querySelectorAll(`[data-n="${n}"]`).forEach((node) => node.classList.toggle('hot', on));
}

function trimForDisplay(body) {
  const copy = { ...body, pattern: `{ ${body.pattern.vertices_coords.length} vertices, ${body.pattern.edges_vertices.length} edges (FOLD format) }` };
  if (copy.photo) copy.photo = { ...copy.photo, lines: `[${copy.photo.lines.length} detected lines]`, sheet: copy.photo.sheet ? '<straightened photo>' : null };
  for (const k of ['check_id']) if (copy[k] == null) delete copy[k];
  return copy;
}

// --- actions -----------------------------------------------------------------

async function loadExample(id) {
  document.querySelectorAll('.chip').forEach((c) => c.setAttribute('aria-pressed', String(c.dataset.id === id)));
  try { render(await call(`/api/examples/${id}`)); } catch (e) { showError(e.message); }
}

async function applyFix(spot) {
  try {
    const body = await call('/api/fix', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ fold: current.pattern, vertex: spot.vertex, fix: spot.fix }),
    });
    render(body);
  } catch (e) { showError(e.message); }
}

async function uploadFile(file) {
  const form = new FormData();
  form.append('file', file);
  clearChips();
  try { render(await call('/api/check/file', { method: 'POST', body: form })); } catch (e) { showError(e.message); }
}

async function uploadPhoto(file) {
  const form = new FormData();
  form.append('file', file);
  form.append('snap', $('opt-snap').checked ? 'true' : 'false');
  clearChips();
  try {
    const body = await call('/api/photo', { method: 'POST', body: form });
    const img = $('photo');
    if (body.photo && body.photo.sheet) {
      img.classList.remove('fade');
      img.src = body.photo.sheet;
      img.hidden = false;
      await img.decode().catch(() => {});
    }
    render(body, { fromPhoto: Boolean(body.photo && body.photo.sheet) });
  } catch (e) { showError(e.message); }
}

function clearChips() {
  document.querySelectorAll('.chip').forEach((c) => c.setAttribute('aria-pressed', 'false'));
}

// --- wiring ------------------------------------------------------------------

$('pick-photo').addEventListener('click', () => $('photo-input').click());
$('pick-file').addEventListener('click', () => $('file-input').click());
$('photo-input').addEventListener('change', (e) => { if (e.target.files[0]) uploadPhoto(e.target.files[0]); e.target.value = ''; });
$('file-input').addEventListener('change', (e) => { if (e.target.files[0]) uploadFile(e.target.files[0]); e.target.value = ''; });
$('opt-fixes').addEventListener('change', (e) => $('issues').classList.toggle('hide-fixes', !e.target.checked));

(async function start() {
  try {
    const examples = await call('/api/examples');
    const box = $('examples');
    for (const ex of examples) {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'chip';
      b.dataset.id = ex.id;
      b.textContent = ex.label;
      b.setAttribute('aria-pressed', 'false');
      b.addEventListener('click', () => loadExample(ex.id));
      box.appendChild(b);
    }
    loadExample(examples[0].id);
  } catch (e) {
    showError('Could not reach the checker. Is the server running?');
  }
})();
