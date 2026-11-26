// Province detail panel: latest B3 forecast, selected-quarter back-test, conditions vs
// normal (context only), flags/regrouping notes and a Chart.js history chart with a
// data-table alternative. Listens for `anisight:select-unit` on document.
// Chart.js is loaded as a classic script (window.Chart) from app/vendor/.
import { CROP_LABELS, quarterLabel, parseQuarter } from './lib/format.js';
import {
  buildChartSeries, chartTableRows, chartSummary, flagMessages, forecastFlags,
  conditionsLines, regroupNotes, forecastCard, backtestSentence, partialWindowFlag,
} from './lib/detail.js';

// Okabe-Ito palette (colour-blind safe); every series also differs by line/point style.
const COLORS = {
  actual: '#000000',
  b3: '#0072B2',
  band: 'rgba(0, 114, 178, 0.18)',
  b0: '#E69F00',
  b2: '#009E73',
  forecast: '#D55E00',
};

// Dataset indexes (used by toggles and tooltip callbacks).
const DS = { LO: 0, HI: 1, B3: 2, ACTUAL: 3, B0: 4, B2: 5, FC: 6, FC_RANGE: 7 };

/** Tiny element builder: h('p', {class: 'x'}, 'text', child). Text is never parsed as HTML. */
function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'class') el.className = v;
    else el.setAttribute(k, v === true ? '' : String(v));
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

export function createPanel(el, { meta, forecast, predictions, units, onClose }) {
  const features = units.features;
  const definitions = meta?.definitions ?? {};
  const toggles = { b0: false, b2: false }; // B0/B2 off by default, kept across units
  let chart = null;
  let openUnit = null; // unit shown, or null when closed
  let opener = null; // element focused when the panel opened

  el.setAttribute('aria-labelledby', 'panel-title');
  el.setAttribute('tabindex', '-1');

  function close({ user = false } = {}) {
    if (chart) { chart.destroy(); chart = null; }
    const hadFocus = el.contains(document.activeElement);
    const unit = openUnit;
    el.hidden = true;
    el.replaceChildren();
    openUnit = null;
    if (user || hadFocus) restoreFocus(unit);
  }

  function restoreFocus(unit) {
    // The unit list is re-rendered on every selection, so a list button that opened the
    // panel is gone; fall back to the current button for the same unit.
    const usable = (x) => x && x.isConnected && x !== document.body
      && x !== document.documentElement && !el.contains(x);
    const target = (usable(opener) ? opener : null)
      ?? document.querySelector(`.unit-btn[data-unit="${CSS.escape(unit ?? '')}"]`)
      ?? document.getElementById('unit-list-heading');
    opener = null;
    target?.focus();
  }

  // Esc closes the panel from anywhere except an open native control.
  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape' || el.hidden || e.defaultPrevented) return;
    if (e.target instanceof HTMLSelectElement) return;
    e.preventDefault();
    onClose?.();
  });

  document.addEventListener('anisight:select-unit', (e) => {
    const d = e.detail;
    if (!d.unit) {
      if (!el.hidden) close({ user: Boolean(d.userAction) });
      return;
    }
    const isNewUnit = d.unit !== openUnit;
    const wasHidden = el.hidden;
    if (wasHidden) opener = document.activeElement;
    render(d);
    if (d.userAction && (wasHidden || isNewUnit)) {
      el.querySelector('#panel-title')?.focus({ preventScroll: true });
      el.scrollIntoView({ block: 'start', behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' });
    }
  });

  function render({ crop, quarter, unit, row }) {
    if (chart) { chart.destroy(); chart = null; }
    openUnit = unit;
    const cropLabel = CROP_LABELS[crop] ?? crop;
    const fq = forecast.target_quarter;
    const fRow = forecast.forecasts?.[crop]?.[unit] ?? null;
    const history = predictions?.[crop]?.[unit] ?? [];
    const isForecastQuarter = quarter === fq;
    const feature = features.find((f) => f.properties.unit === unit);

    const header = h('div', { class: 'panel-head' },
      h('h2', { id: 'panel-title', tabindex: '-1' }, `${unit}`, h('span', { class: 'panel-crop' }, ` · ${cropLabel}`)),
      h('button', { type: 'button', class: 'panel-close', 'aria-label': `Close details for ${unit}` }, 'Close ✕'),
    );
    header.querySelector('.panel-close').addEventListener('click', () => onClose?.());

    const sections = [header];
    sections.push(forecastSection(fRow, fq, feature));
    if (!isForecastQuarter) sections.push(backtestSection(row, quarter));
    sections.push(conditionsSection(isForecastQuarter ? fRow?.conditions : row, quarter));
    sections.push(notesSection({ unit, quarter, row, fRow, isForecastQuarter }));
    sections.push(chartSection({ unit, cropLabel, history, fq, fRow }));
    sections.push(h('p', { class: 'panel-esc-hint' }, 'Press Esc to close these details.'));

    el.replaceChildren(...sections);
    el.hidden = false;
    drawChart({ unit, cropLabel, history, fq, fRow });
  }

  function forecastSection(fRow, fq, feature) {
    const sec = h('section', { class: 'card', 'aria-labelledby': 'fc-title' },
      h('h3', { id: 'fc-title' }, `Latest forecast: ${quarterLabel(fq)}`));
    const card = forecastCard(fRow, { quarter: fq, note: forecast.area_note });
    if (!card) {
      const last = feature?.properties?.last;
      sec.append(h('p', {}, last
        ? `No live forecast for this unit: PSA last reported it this way in ${quarterLabel(last)}.`
        : 'No live forecast for this unit.'));
      return sec;
    }
    sec.append(
      h('p', { class: 'fc-yield' }, h('strong', {}, 'Yield: '), card.yieldText),
      h('p', {}, card.vsNormal),
      h('dl', { class: 'fc-list' },
        h('dt', {}, 'Harvested area'), h('dd', {}, card.area),
        h('dt', {}, 'Predicted production'), h('dd', {}, card.production)),
      card.note ? h('p', { class: 'note' }, `Production ${card.note}.`) : null,
      h('p', { class: 'note' }, card.model),
      h('p', { class: 'note' }, `PSA has not published ${quarterLabel(fq)} yet. Forecast made with PSA data through ${quarterLabel(fRow.data_vintage ?? forecast.data_vintage)}.`),
    );
    return sec;
  }

  function backtestSection(row, quarter) {
    return h('section', { class: 'card', 'aria-labelledby': 'bt-title' },
      h('h3', { id: 'bt-title' }, `Selected quarter: ${quarterLabel(quarter)}`),
      h('p', {}, backtestSentence(row) ?? 'No data for this quarter.'));
  }

  function conditionsSection(cond, quarter) {
    const { lines, notes } = conditionsLines(cond);
    const contextNote = forecast.conditions_note ?? definitions.conditions_note;
    const defs = [
      ['Rainfall', definitions.rain_pct && definitions.rain_anom ? `${definitions.rain_anom}; ${definitions.rain_pct}` : null],
      ['NDVI', definitions.ndvi_anom],
      ['EVI', definitions.evi_anom],
    ].filter(([, v]) => v);
    return h('section', { class: 'card card-context', 'aria-labelledby': 'cond-title' },
      h('h3', { id: 'cond-title' }, `Conditions vs normal: ${quarterLabel(quarter)}`),
      contextNote ? h('p', { class: 'context-badge' }, capitalise(contextNote)) : null,
      h('ul', { class: 'cond-list' }, lines.map((l) => h('li', {}, l.text))),
      notes.map((n) => h('p', { class: 'note' }, n)),
      defs.length ? h('details', {},
        h('summary', {}, 'How "normal" is defined'),
        h('dl', { class: 'defs' }, defs.flatMap(([k, v]) => [h('dt', {}, k), h('dd', {}, v)]))) : null);
  }

  function notesSection({ unit, quarter, row, fRow, isForecastQuarter }) {
    const flagDefs = definitions.flags ?? {};
    const items = [];
    if (isForecastQuarter) {
      for (const m of flagMessages(forecastFlags(fRow), flagDefs)) items.push(m.text);
      const partial = partialWindowFlag(fRow?.conditions);
      if (partial) items.push(partial);
    } else {
      for (const m of flagMessages(row?.flags, flagDefs)) items.push(m.text);
    }
    const regroup = regroupNotes(unit, features);
    const sec = h('section', { class: 'card', 'aria-labelledby': 'notes-title' },
      h('h3', { id: 'notes-title' }, `Notes for ${quarterLabel(quarter)}`));
    sec.append(items.length
      ? h('ul', { class: 'flag-list' }, items.map((t) => h('li', {}, h('span', { class: 'flag-mark', 'aria-hidden': 'true' }, '! '), t)))
      : h('p', {}, 'No data flags for this quarter.'));
    if (regroup.length) {
      sec.append(h('h4', {}, 'Reporting-unit changes'), h('ul', { class: 'flag-list' }, regroup.map((t) => h('li', {}, t))));
    }
    return sec;
  }

  function chartSection({ unit, cropLabel, history, fq, fRow }) {
    const series = buildChartSeries(history, { forecastQuarter: fq, forecastRow: fRow });
    const summary = chartSummary(unit, cropLabel, series);
    const rows = chartTableRows(history, { forecastQuarter: fq, forecastRow: fRow });
    const toggle = (key, label) => {
      const id = `toggle-${key}`;
      const input = h('input', { type: 'checkbox', id, checked: toggles[key] });
      input.addEventListener('change', () => {
        toggles[key] = input.checked;
        if (chart) {
          chart.setDatasetVisibility(key === 'b0' ? DS.B0 : DS.B2, input.checked);
          chart.update();
        }
      });
      return h('label', { for: id, class: 'toggle' }, input, ` ${label}`);
    };
    const firstBand = series.labels.find((_, i) => series.lo[i] !== null && series.hi[i] !== null);
    const sec = h('section', { class: 'card', 'aria-labelledby': 'chart-title' },
      h('h3', { id: 'chart-title' }, 'Yield history (t/ha)'),
      h('p', { class: 'note' }, 'Black dots: PSA actual. Blue line: B3 back-test prediction, each made without that year\'s PSA data. '
        + 'Shaded: its 80% range' + (firstBand && firstBand !== series.labels[0] ? ` (available from ${quarterLabel(firstBand)})` : '') + '. '
        + (fRow ? `Large diamond and vertical bar: the live ${quarterLabel(fq)} forecast and its 80% range. ` : '')
        + 'Gaps are quarters PSA did not report for this unit.'),
      h('div', { class: 'toggles', role: 'group', 'aria-label': 'Show baseline forecasts' },
        toggle('b0', 'B0: same quarter last year'),
        toggle('b2', 'B2: average + trend')),
      h('div', { class: 'chart-box' },
        h('canvas', { id: 'history-chart', role: 'img', 'aria-label': summary })),
      series.outlier.some(Boolean) ? h('p', { class: 'note' }, 'Triangles mark PSA figures flagged as outliers.') : null,
      h('details', { class: 'chart-table' },
        h('summary', {}, 'Show the chart data as a table'),
        h('div', { class: 'table-scroll', tabindex: '0', role: 'region', 'aria-label': `Yield history table for ${unit}` },
          h('table', {},
            h('caption', { class: 'sr-only' }, summary),
            h('thead', {}, h('tr', {},
              ['Quarter', 'PSA actual', 'B3', 'B3 80% range', 'B0', 'B2'].map((t) => h('th', { scope: 'col' }, t)))),
            h('tbody', {}, rows.map((r) => h('tr', { class: r.kind === 'forecast' ? 'is-forecast' : null },
              h('th', { scope: 'row' }, r.label, r.outlier ? ' (outlier)' : ''),
              h('td', {}, r.actual), h('td', {}, r.b3), h('td', {}, r.range),
              h('td', {}, r.b0), h('td', {}, r.b2))))))));
    return sec;
  }

  function drawChart({ history, fq, fRow }) {
    const Chart = globalThis.Chart;
    const canvas = el.querySelector('#history-chart');
    if (!Chart || !canvas) {
      canvas?.replaceWith(h('p', { class: 'note' }, 'Chart unavailable; see the data table below.'));
      return;
    }
    const s = buildChartSeries(history, { forecastQuarter: fq, forecastRow: fRow });
    const f = s.forecast;
    const fmt = (v) => (v === null || v === undefined ? '—' : v.toFixed(2));
    chart = new Chart(canvas, {
      type: 'line',
      data: {
        labels: s.labels,
        datasets: [
          { label: '80% range (low)', data: s.lo, borderWidth: 0, pointRadius: 0, fill: false, spanGaps: false },
          {
            label: 'B3 80% range', data: s.hi, borderWidth: 0, pointRadius: 0, fill: '-1',
            backgroundColor: COLORS.band, borderColor: COLORS.band, spanGaps: false, pointStyle: 'rect',
          },
          {
            label: 'B3 back-test', data: s.pred, borderColor: COLORS.b3, backgroundColor: COLORS.b3,
            borderWidth: 2, pointRadius: 0, pointHitRadius: 6, spanGaps: false, pointStyle: 'line',
          },
          {
            label: 'PSA actual', data: s.actual, showLine: false, borderColor: COLORS.actual,
            backgroundColor: COLORS.actual,
            pointStyle: s.outlier.map((o) => (o ? 'triangle' : 'circle')),
            pointRadius: s.outlier.map((o) => (o ? 5 : 2.5)),
          },
          {
            label: 'B0: same quarter last year', data: s.b0, borderColor: COLORS.b0, backgroundColor: COLORS.b0,
            borderDash: [6, 4], borderWidth: 1.5, pointRadius: 0, pointHitRadius: 6, spanGaps: false,
            hidden: !toggles.b0, pointStyle: 'line',
          },
          {
            label: 'B2: average + trend', data: s.b2, borderColor: COLORS.b2, backgroundColor: COLORS.b2,
            borderDash: [2, 3], borderWidth: 1.5, pointRadius: 0, pointHitRadius: 6, spanGaps: false,
            hidden: !toggles.b2, pointStyle: 'line',
          },
          {
            label: `Forecast ${quarterLabel(fq)}`, data: f && f.y !== null ? [{ x: f.x, y: f.y }] : [],
            showLine: false, pointStyle: 'rectRot', pointRadius: 7, pointHoverRadius: 8,
            borderColor: '#000000', borderWidth: 1, backgroundColor: COLORS.forecast,
          },
          {
            label: 'Forecast 80% range',
            data: f && f.lo !== null && f.hi !== null ? [{ x: f.x, y: f.lo }, { x: f.x, y: f.hi }] : [],
            borderColor: COLORS.forecast, borderWidth: 3, pointRadius: 0, pointHitRadius: 0,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        // 'x' (not 'index'): the forecast datasets use {x, y} points, so their data
        // indexes do not line up with the label indexes.
        interaction: { mode: 'x', intersect: false },
        scales: {
          x: {
            ticks: {
              autoSkip: false,
              maxRotation: 0,
              callback(value) {
                const p = parseQuarter(this.getLabelForValue(value));
                if (!p || p.q !== 1) return '';
                const every = this.chart.width < 520 ? 3 : 1;
                return p.y % every === 0 ? String(p.y) : '';
              },
            },
            grid: { display: false },
          },
          y: { title: { display: true, text: 'Yield (t/ha)' } },
        },
        plugins: {
          legend: {
            onClick: () => {}, // visibility is controlled by the labelled checkboxes
            labels: {
              usePointStyle: true,
              boxWidth: 10,
              filter: (item, data) => item.datasetIndex !== DS.LO && item.datasetIndex !== DS.FC_RANGE
                && !data.datasets[item.datasetIndex].hidden
                && data.datasets[item.datasetIndex].data.length > 0,
            },
          },
          tooltip: {
            filter: (item) => item.datasetIndex !== DS.LO && item.datasetIndex !== DS.FC_RANGE
              && item.raw !== null && item.raw !== undefined,
            callbacks: {
              title: (items) => (items[0] ? quarterLabel(items[0].label) : ''),
              label(item) {
                const i = item.dataIndex;
                if (item.datasetIndex === DS.HI) return `80% range: ${fmt(s.lo[i])}–${fmt(s.hi[i])} t/ha`;
                if (item.datasetIndex === DS.FC) return `Forecast (B3): ${fmt(f.y)} t/ha (80% range ${fmt(f.lo)}–${fmt(f.hi)})`;
                const y = typeof item.raw === 'object' ? item.raw.y : item.raw;
                return `${item.dataset.label}: ${fmt(y)} t/ha`;
              },
            },
          },
        },
      },
    });
  }

  return { close };
}

function capitalise(s) {
  return s ? s[0].toUpperCase() + s.slice(1) : s;
}
