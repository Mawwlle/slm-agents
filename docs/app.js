// The page's numbers and charts, from window.DATA (docs/data.js, written by paper/site.py). Charts are plain SVG
// drawn at the plot's pixel width and redrawn when it changes.
'use strict';

const DATA = window.DATA;
const NAMES = {
  'qwen3.5-9b': 'Qwen3.5-9B', 'qwen3.5-4b': 'Qwen3.5-4B', 'qwen3.5-2b': 'Qwen3.5-2B', 'qwen3.5-0.8b': 'Qwen3.5-0.8B',
  'qwen3.5-0.8b-wsp': 'Qwen3.5-0.8B (weight-edited)', 'qwen3-0.6b': 'Qwen3-0.6B', 'qwen3-vl-8b': 'Qwen3-VL-8B',
  'qwen3-vl-30b': 'Qwen3-VL-30B-A3B', 'gemma-4-26b-a4b': 'Gemma-4-26B-A4B', 'gemma-4-e4b': 'Gemma-4-E4B',
  'gemma-4-e2b': 'Gemma-4-E2B', 'granite-4.0-micro': 'Granite-4.0-micro', 'mnn-pocket-0.3b': 'MNN-Pocket-0.3B',
  'lfm2.5-8b-a1b': 'LFM2.5-8B-A1B', 'lfm2.5-2.6b': 'LFM2.5-2.6B', 'lfm2.5-1.2b': 'LFM2.5-1.2B',
  'lfm2.5-350m': 'LFM2.5-350M', 'lfm2.5-230m': 'LFM2.5-230M',
};
const STAGES = { A: 'Stage 1 · escalation', H: 'Stage 2 · harness routing', L: 'Phase L · replication',
                 P: 'Stage 3 · delegation', C: 'Stage 4 · cheap planners' };
const SVG = 'http://www.w3.org/2000/svg';

const pct = (x, digits = 0) => `${(100 * x).toFixed(digits)}%`;
const SUPERSCRIPT = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
const power = (e) => `10${String(e).split('').map((c) => SUPERSCRIPT[c]).join('')}`;

function sci(value) {
  const e = Math.floor(Math.log10(value));
  return `${(value / 10 ** e).toFixed(1)}×${power(e)}`;
}

function el(tag, attrs, parent, text) {
  const node = document.createElementNS(SVG, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  if (text !== undefined) node.textContent = text;
  parent.appendChild(node);
  return node;
}

function linear([d0, d1], [r0, r1]) {
  return (v) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);
}

function logarithmic([d0, d1], range) {
  const scale = linear([Math.log10(d0), Math.log10(d1)], range);
  return (v) => scale(Math.log10(v));
}

/** An SVG as wide as the plot, with the drawing area inside `margin`. */
function canvas(plot, height, margin) {
  plot.replaceChildren();
  const width = plot.clientWidth;
  const svg = el('svg', { viewBox: `0 0 ${width} ${height}`, width, height, role: 'img' }, plot);
  return { svg, width, height, left: margin[3], right: width - margin[1], top: margin[0], bottom: height - margin[2] };
}

function xAxis(c, x, ticks, format, title) {
  for (const t of ticks) {
    el('line', { x1: x(t), x2: x(t), y1: c.top, y2: c.bottom, class: 'gridline' }, c.svg);
    el('text', { x: x(t), y: c.bottom + 16, 'text-anchor': 'middle', 'font-size': 11 }, c.svg, format(t));
  }
  el('line', { x1: c.left, x2: c.right, y1: c.bottom, y2: c.bottom, class: 'axis' }, c.svg);
  if (title) el('text', { x: c.right, y: c.bottom + 32, 'text-anchor': 'end', 'font-size': 11 }, c.svg, title);
}

function yAxis(c, y, ticks, format, title) {
  for (const t of ticks) {
    el('line', { x1: c.left, x2: c.right, y1: y(t), y2: y(t), class: 'gridline' }, c.svg);
    el('text', { x: c.left - 8, y: y(t) + 4, 'text-anchor': 'end', 'font-size': 11 }, c.svg, format(t));
  }
  if (title) el('text', { x: c.left, y: c.top - 10, 'font-size': 11 }, c.svg, title);
}

/** A marker of a given shape, with a surface-coloured ring so overlapping marks stay apart. */
function marker(svg, shape, x, y, size, colour) {
  const style = `fill: ${colour}; stroke: var(--surface); stroke-width: 1.5`;
  const s = size;
  if (shape === 'circle') return el('circle', { cx: x, cy: y, r: s, style }, svg);
  if (shape === 'square') return el('rect', { x: x - s, y: y - s, width: 2 * s, height: 2 * s, rx: 1.5, style }, svg);
  if (shape === 'triangle') return el('path', { d: `M${x},${y - 1.2 * s} L${x + 1.1 * s},${y + 0.8 * s} L${x - 1.1 * s},${y + 0.8 * s} Z`, style }, svg);
  return el('path', { d: `M${x - s},${y - s} L${x + s},${y + s} M${x - s},${y + s} L${x + s},${y - s}`,
                      style: `stroke: ${colour}; stroke-width: 2.2; fill: none` }, svg);
}

const tooltip = document.getElementById('tooltip');

/** A hit area bigger than the mark that shows `html` on hover, tap or keyboard focus. */
function hover(svg, x, y, html) {
  const hit = el('circle', { cx: x, cy: y, r: 12, class: 'hit', tabindex: 0 }, svg);
  const show = () => {
    tooltip.innerHTML = html;
    tooltip.hidden = false;
    const box = hit.getBoundingClientRect();
    const tip = tooltip.getBoundingClientRect();
    const left = Math.min(Math.max(8, box.left + box.width / 2 - tip.width / 2), window.innerWidth - tip.width - 8);
    const top = box.top - tip.height - 6 < 8 ? box.bottom + 6 : box.top - tip.height - 6;
    tooltip.style.left = `${left}px`;
    tooltip.style.top = `${top}px`;
  };
  const hide = () => { tooltip.hidden = true; };
  hit.addEventListener('pointerenter', show);
  hit.addEventListener('pointerleave', hide);
  hit.addEventListener('focus', show);
  hit.addEventListener('blur', hide);
}

function legend(c, items) {
  let x = c.left;
  for (const [shape, colour, label] of items) {
    marker(c.svg, shape, x + 5, 14, 4.5, colour);
    const text = el('text', { x: x + 15, y: 18, 'font-size': 11, class: 'label' }, c.svg, label);
    x += 15 + text.getComputedTextLength() + 18;
  }
}

/* Figure 1: replaceability against active parameters. */
function replaceChart(plot, track) {
  const c = canvas(plot, 320, [30, 18, 40, 46]);
  const x = logarithmic([0.2, 12], [c.left, c.right]);
  const y = linear([0, 1], [c.bottom, c.top]);
  xAxis(c, x, [0.25, 0.5, 1, 2, 4, 8], (t) => `${t}B`, 'active parameters, log scale');
  yAxis(c, y, [0, 0.25, 0.5, 0.75, 1], (t) => pct(t), 'solves what the 27B model solves');
  el('line', { x1: x(1.5), x2: x(1.5), y1: c.top, y2: c.bottom, class: 'ref' }, c.svg);
  el('text', { x: x(1.5) + 4, y: c.top + 10, 'font-size': 10 }, c.svg, '1.5B');
  // the models the text names, each labelled on the side where its neighbours leave room: [dx, dy, anchor]
  const labelled = { 'qwen3.5-9b': [-8, -7, 'end'], 'qwen3.5-4b': [8, 16, 'start'], 'qwen3-vl-30b': [-8, -7, 'end'],
                     'lfm2.5-1.2b': [-8, -7, 'end'] };
  for (const [model, r] of Object.entries(DATA.replace)) {
    const px = x(r.active_b), py = y(r.replace[track]);
    marker(c.svg, 'circle', px, py, 4.5, 'var(--s1)');
    if (labelled[model]) {
      const [dx, dy, anchor] = labelled[model];
      el('text', { x: px + dx, y: py + dy, 'text-anchor': anchor, 'font-size': 10.5, class: 'label' }, c.svg, NAMES[model]);
    }
    hover(c.svg, px, py, `<b>${NAMES[model]}</b><br>${r.active_b}B active<br>solves ${pct(r.replace[track])} of what the 27B
      model solves<br>${r.cost_ratio.toFixed(1)}× cheaper per task`);
  }
}

/* Figure 2: each executor alone and under qwen3.8's plan. */
function levelChart(plot, track) {
  const executors = Object.entries(DATA.delegate.executors).sort((a, b) => a[1].active_b - b[1].active_b);
  const row = 30;
  const narrow = plot.clientWidth < 480;
  const c = canvas(plot, 44 + executors.length * row + 40, [44, 18, 40, narrow ? 104 : 128]);
  const x = linear([0, 1], [c.left, c.right]);
  xAxis(c, x, [0, 0.25, 0.5, 0.75, 1], (t) => pct(t), `${track === 'shell' ? 'shell' : 'multi-turn'} tasks solved (test)`);
  legend(c, [['circle', 'var(--s1)', 'alone'], ['square', 'var(--s2)', "with qwen3.8's plan"]]);
  const own = DATA.delegate.p_to_b_by_track[track];
  el('line', { x1: x(own), x2: x(own), y1: c.top - 6, y2: c.bottom, class: 'ref' }, c.svg);
  executors.forEach(([model, r], i) => {
    const py = c.top + row * (i + 0.5);
    const alone = r.alone[track], planned = r.planned[track];
    el('text', { x: c.left - 10, y: py + 4, 'text-anchor': 'end', 'font-size': narrow ? 10 : 11, class: 'label' }, c.svg, NAMES[model]);
    el('line', { x1: x(alone), x2: x(planned), y1: py, y2: py, class: 'link' }, c.svg);
    marker(c.svg, 'circle', x(alone), py, 5, 'var(--s1)');
    marker(c.svg, 'square', x(planned), py, 4.5, 'var(--s2)');
    const html = `<b>${NAMES[model]}</b> · ${r.active_b}B active<br>alone ${pct(alone)} → planned ${pct(planned)}<br>planner's
      share of compute ${pct(r.planner_share)}`;
    hover(c.svg, x(alone), py, html);
    hover(c.svg, x(planned), py, html);
  });
}

/* Figure 3: planner protocol violations by size. */
function protocolChart(plot) {
  const rows = Object.entries(DATA.cheap.protocol);
  const row = 38;
  const c = canvas(plot, 16 + rows.length * row + 40, [16, 48, 40, 150]);
  const x = linear([0, 1], [c.left, c.right]);
  xAxis(c, x, [0, 0.25, 0.5, 0.75, 1], (t) => pct(t), 'planner turns that break the protocol');
  rows.forEach(([model, r], i) => {
    const py = c.top + row * i + 8;
    const h = row - 16;
    el('text', { x: c.left - 10, y: py + h / 2 + 4, 'text-anchor': 'end', 'font-size': 11, class: 'label' }, c.svg,
       `${NAMES[model]} · ${r.active_b}B`);
    el('rect', { x: c.left, y: py, width: Math.max(2, x(r.violation_rate) - c.left), height: h, rx: 3,
                 style: 'fill: var(--s2)' }, c.svg);
    el('text', { x: x(r.violation_rate) + 6, y: py + h / 2 + 4, 'font-size': 11, class: 'label' }, c.svg, pct(r.violation_rate));
  });
}

/* Figure 4: stage-4 cost against quality, with the router's threshold on a slider. */
function systemName(system) {
  const solo = system.match(/^S\((.+)\)$/);
  if (solo) return NAMES[solo[1]];
  const [planner, executor] = system.slice(2, -1).split('→');
  return `${NAMES[planner]} plans for ${NAMES[executor]}`;
}

function frontierChart(plot, state) {
  const { systems, router } = DATA.cheap.frontier;
  const c = canvas(plot, 380, [30, 18, 40, 46]);
  const x = logarithmic([1e13, 1.5e15], [c.left, c.right]);
  const y = linear([0, 0.65], [c.bottom, c.top]);
  xAxis(c, x, [1e13, 1e14, 1e15], (t) => power(Math.round(Math.log10(t))), 'FLOPs per episode, log scale');
  yAxis(c, y, [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], (t) => pct(t), 'tasks solved, mean over tracks');
  const tiny = 'D(lfm2.5-2.6b→lfm2.5-350m)';
  if (state.plan) {
    for (const [system, r] of Object.entries(systems)) {
      if (!system.startsWith('D(')) continue;
      const shape = system === tiny ? 'cross' : 'square';
      marker(c.svg, shape, x(r.flops), y(r.q), 4.5, system === tiny ? 'var(--ink)' : 'var(--s2)');
      hover(c.svg, x(r.flops), y(r.q), `<b>${systemName(system)}</b>${system === tiny ? ' (the tiny stack)' : ''}<br>solves
        ${pct(r.q, 1)} · ${sci(r.flops)} FLOPs`);
    }
    const t = systems[tiny];
    el('text', { x: x(t.flops), y: y(t.q) - 22, 'text-anchor': 'middle', 'font-size': 10.5, class: 'label' }, c.svg, 'tiny stack');
  }
  if (state.router) {
    // the detector's threshold sweep is a curve; planning after an observed failure is one point beside it
    const sweep = router.filter((r) => r.signal === 'det');
    el('polyline', { points: sweep.map((r) => `${x(r.flops)},${y(r.q)}`).join(' '),
                     style: 'fill: none; stroke: var(--s3); stroke-width: 2' }, c.svg);
    for (const r of router) marker(c.svg, 'triangle', x(r.flops), y(r.q), 3.5, 'var(--s3)');
    const failure = router.find((r) => r.signal === 'obs');
    if (c.width >= 560) el('text', { x: x(failure.flops), y: y(failure.q) - 12, 'text-anchor': 'middle', 'font-size': 10.5, class: 'label' },
       c.svg, 'after a failure');
    const pick = router[state.index];
    el('circle', { cx: x(pick.flops), cy: y(pick.q), r: 9, style: 'fill: none; stroke: var(--s3); stroke-width: 2' }, c.svg);
    marker(c.svg, 'triangle', x(pick.flops), y(pick.q), 5, 'var(--s3)');
  }
  if (state.single) {
    for (const [system, r] of Object.entries(systems)) {
      if (!system.startsWith('S(')) continue;
      const px = x(r.flops), py = y(r.q), left = r.flops < 1e14;
      marker(c.svg, 'circle', px, py, 5, 'var(--s1)');
      el('text', { x: px + (left ? 8 : -8), y: py - 9, 'text-anchor': left ? 'start' : 'end', 'font-size': 10.5, class: 'label' },
         c.svg, systemName(system));
      hover(c.svg, px, py, `<b>${systemName(system)}</b> alone<br>solves ${pct(r.q, 1)} · ${sci(r.flops)} FLOPs`);
    }
  }
}

function routerReadout(index) {
  const r = DATA.cheap.frontier.router[index];
  const base = DATA.cheap.frontier.systems['S(qwen3.5-4b)'];
  const signal = r.signal === 'obs' ? 'plan after an observed failure' : `plan when risk ≥ ${r.tau.toFixed(2)}`;
  return `${signal} · planned on ${pct(r.planned)} of tasks · solves ${pct(r.q, 1)} · ` +
    `${sci(r.flops)} FLOPs, ${(r.flops / base.flops).toFixed(2)}× Qwen3.5-4B alone`;
}

/** A chart's figure: its track buttons, and a redraw whenever the state or the plot's width changes. */
function figure(plotId, draw, state) {
  const plot = document.getElementById(plotId);
  const fig = plot.closest('figure');
  const redraw = () => draw(plot, state);
  for (const button of fig.querySelectorAll('[data-track]')) {
    button.addEventListener('click', () => {
      for (const b of fig.querySelectorAll('[data-track]')) b.setAttribute('aria-pressed', String(b === button));
      state.track = button.dataset.track;
      redraw();
    });
  }
  let width = 0;
  new ResizeObserver(() => {
    if (plot.clientWidth !== width) {
      width = plot.clientWidth;
      redraw();
    }
  }).observe(plot);
  return redraw;
}

function verdictTable(filter) {
  const body = document.getElementById('verdicts');
  body.replaceChildren();
  let stage = null;
  for (const v of DATA.verdicts) {
    if (filter === 'holds' && !v.holds) continue;
    if (filter === 'not' && v.holds) continue;
    if (STAGES[v.id[0]] !== stage) {
      stage = STAGES[v.id[0]];
      body.insertAdjacentHTML('beforeend', `<tr class="stage"><td colspan="4">${stage}</td></tr>`);
    }
    const p = v.holds ? (v.p < 0.001 ? '< 0.001' : v.p.toFixed(3)) : '—';
    const verdict = v.holds ? '<span class="pill holds">holds</span>' : '<span class="pill">not shown</span>';
    body.insertAdjacentHTML('beforeend', `<tr><td>${v.id}</td><td>${v.claim}</td><td>${verdict}</td><td class="num">${p}</td></tr>`);
  }
}

function contents() {
  const links = new Map([...document.querySelectorAll('.toc a')].map((a) => [a.getAttribute('href').slice(1), a]));
  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      for (const a of links.values()) a.classList.remove('active');
      links.get(entry.target.id)?.classList.add('active');
    }
  }, { rootMargin: '-15% 0px -75% 0px' });
  for (const id of links.keys()) observer.observe(document.getElementById(id));
}

function theme() {
  const root = document.documentElement;
  try {
    const saved = localStorage.getItem('theme');
    if (saved) root.dataset.theme = saved;
  } catch { /* storage may be blocked; the system theme applies */ }
  document.getElementById('theme').addEventListener('click', () => {
    const dark = root.dataset.theme ? root.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
    root.dataset.theme = dark ? 'light' : 'dark';
    try { localStorage.setItem('theme', root.dataset.theme); } catch { /* not remembered */ }
  });
}

function copyButton() {
  const button = document.getElementById('copy');
  const bib = document.getElementById('bib');
  button.addEventListener('click', () => {
    navigator.clipboard.writeText(bib.textContent)
      .then(() => { button.textContent = 'Copied'; })
      .catch(() => { getSelection().selectAllChildren(bib); button.textContent = 'Selected'; });
  });
}

function main() {
  for (const node of document.querySelectorAll('[data-n]')) node.textContent = DATA.numbers[node.dataset.n];

  figure('chart-replace', (plot, s) => replaceChart(plot, s.track), { track: 'multiturn' });
  figure('chart-level', (plot, s) => levelChart(plot, s.track), { track: 'multiturn' });
  figure('chart-protocol', protocolChart, {});

  const router = DATA.cheap.frontier.router;
  const state = { single: true, plan: true, router: true, index: router.findIndex((r) => r.signal === 'obs') };
  const redraw = figure('chart-frontier', frontierChart, state);
  const slider = document.getElementById('tau');
  const out = document.getElementById('tau-out');
  slider.max = router.length - 1;
  slider.value = state.index;
  out.textContent = routerReadout(state.index);
  slider.addEventListener('input', () => {
    state.index = Number(slider.value);
    out.textContent = routerReadout(state.index);
    redraw();
  });
  for (const key of ['single', 'plan', 'router']) {
    document.getElementById(`show-${key}`).addEventListener('change', (e) => {
      state[key] = e.target.checked;
      redraw();
    });
  }

  verdictTable('all');
  for (const button of document.querySelectorAll('[data-filter]')) {
    button.addEventListener('click', () => {
      for (const b of document.querySelectorAll('[data-filter]')) b.setAttribute('aria-pressed', String(b === button));
      verdictTable(button.dataset.filter);
    });
  }
  contents();
  theme();
  copyButton();
}

main();
