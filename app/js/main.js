// Home page bootstrap: load app/data/*.json, wire crop/quarter controls and the URL
// hash (#crop/quarter/unit), draw the map and the unit list.
// Task 5 hook: selecting a unit dispatches `anisight:select-unit` on document with
// detail {crop, quarter, unit, row, kind}; the province panel will listen for it.
import { createMap } from './map.js';
import { createPanel } from './panel.js';
import {
  CROPS, CROP_LABELS, availableQuarters, rowsForQuarter, parseHash, buildHash, resolveState,
  quarterLabel, formatPct, formatYield,
} from './lib/format.js';

const DEFAULT_CROP = 'palay';

function modelSentence(models) {
  const names = [...new Set(Object.values(models ?? {}))];
  const which = names.length === 1
    ? `Model: ${names[0]} for both crops`
    : `Models: ${CROPS.map((c) => `${CROP_LABELS[c]} ${models[c]}`).join(', ')}`;
  const b3 = names.includes('B3')
    ? ' (B3 blends "same quarter last year" with "average + trend" from PSA history)'
    : '';
  return `${which}${b3}.`;
}

async function getJSON(path) {
  const res = await fetch(path, { cache: 'no-cache' });
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  return res.json();
}

async function loadData() {
  const [meta, forecast, predictions, units] = await Promise.all([
    getJSON('data/meta.json'),
    getJSON('data/forecast.json'),
    getJSON('data/predictions.json'),
    getJSON('data/units.geojson'),
  ]);
  return { meta, forecast, predictions, units };
}

function main(data) {
  const { meta, forecast, predictions, units } = data;
  const els = {
    crop: document.getElementById('crop'),
    quarter: document.getElementById('quarter'),
    prev: document.getElementById('quarter-prev'),
    next: document.getElementById('quarter-next'),
    kind: document.getElementById('map-kind'),
    list: document.getElementById('unit-list'),
    vintage: document.getElementById('vintage'),
    selected: document.getElementById('selected-unit'),
  };

  const quartersFor = (crop) => availableQuarters(predictions, forecast, crop);
  const rowsFor = (crop, quarter) => rowsForQuarter({
    predictions, forecast, features: units.features, crop, quarter,
  });
  const unitsFor = (crop, quarter) => rowsFor(crop, quarter).rows.map((r) => r.unit);

  els.vintage.textContent =
    `Forecast for ${quarterLabel(forecast.target_quarter)} made with PSA data through `
    + `${quarterLabel(meta.data_vintage)}. ${modelSentence(meta.model)}`;

  for (const c of CROPS) els.crop.add(new Option(CROP_LABELS[c], c));

  let state = { crop: null, quarter: null, unit: null };
  let current = { rows: [] };

  const mapApi = createMap(document.getElementById('map'), {
    meta,
    legendEl: document.getElementById('legend'),
    onSelect: (unit) => go({ ...state, unit: unit === state.unit ? null : unit }, { user: true }),
  });

  createPanel(document.getElementById('panel'), {
    meta, forecast, predictions, units,
    onClose: () => go({ ...state, unit: null }, { user: true }),
  });

  function fillQuarters(crop) {
    const qs = quartersFor(crop);
    els.quarter.replaceChildren(...qs.map((q) => new Option(
      q === forecast.target_quarter ? `${quarterLabel(q)} (forecast)` : quarterLabel(q), q,
    )));
  }

  function renderList(rows) {
    els.list.replaceChildren(...rows.map((r) => {
      const li = document.createElement('li');
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'unit-btn';
      b.setAttribute('aria-pressed', String(r.unit === state.unit));
      b.setAttribute('aria-controls', 'panel');
      b.dataset.unit = r.unit;
      const pct = r.pct === null ? 'no normal yet' : `${formatPct(r.pct)} vs usual`;
      b.innerHTML = '<span class="unit-name"></span><span class="unit-val"></span>';
      b.querySelector('.unit-name').textContent = r.unit;
      b.querySelector('.unit-val').textContent = r.kind === 'forecast'
        ? `Forecast ${formatYield(r.value)} · ${pct}`
        : `Back-test ${formatYield(r.value)} · ${pct} · PSA actual ${formatYield(r.actual)}`;
      b.addEventListener('click', () => go({ ...state, unit: r.unit === state.unit ? null : r.unit }, { user: true }));
      li.append(b);
      return li;
    }));
  }

  // userAction: true when a click/key in this page caused the change (the panel then
  // manages focus); false for initial load and back/forward navigation.
  function emitSelection(userAction) {
    const r = current.rows.find((x) => x.unit === state.unit) ?? null;
    els.selected.textContent = r
      ? `Selected: ${r.unit}. Details are shown below.`
      : 'Select an area on the map or in the list to see its details.';
    document.dispatchEvent(new CustomEvent('anisight:select-unit', {
      detail: {
        crop: state.crop, quarter: state.quarter, unit: state.unit,
        row: r?.row ?? null, kind: r?.kind ?? null, userAction,
      },
    }));
  }

  function apply(next, { user = false } = {}) {
    const cropChanged = next.crop !== state.crop;
    const viewChanged = cropChanged || next.quarter !== state.quarter;
    state = next;
    if (cropChanged) fillQuarters(state.crop);
    els.crop.value = state.crop;
    els.quarter.value = state.quarter;
    const qs = quartersFor(state.crop);
    const i = qs.indexOf(state.quarter);
    els.prev.disabled = i >= qs.length - 1; // older
    els.next.disabled = i <= 0; // newer

    if (viewChanged) {
      current = rowsFor(state.crop, state.quarter);
      if (current.missingGeometry.length) {
        console.warn('No boundary for', current.missingGeometry, 'in', state.quarter);
      }
      const isForecast = state.quarter === forecast.target_quarter;
      els.kind.textContent = isForecast
        ? `${CROP_LABELS[state.crop]}, ${quarterLabel(state.quarter)}: B3 forecast vs this area's usual yield for the quarter. PSA has not published this quarter yet.`
        : `${CROP_LABELS[state.crop]}, ${quarterLabel(state.quarter)}: B3 back-test forecast (made without this year's PSA data) vs usual yield. PSA actual shown on hover and in the list.`;
      const title = document.getElementById('legend-title');
      if (title) title.textContent = isForecast ? 'B3 forecast vs usual yield' : 'B3 back-test vs usual yield';
      mapApi.render(current.rows, { quarter: state.quarter, selectedUnit: state.unit });
    } else {
      mapApi.setSelected(state.unit);
    }
    // Keep keyboard focus on the same list button across re-renders.
    const focusedUnit = document.activeElement?.closest?.('.unit-btn')?.dataset.unit ?? null;
    renderList(current.rows);
    if (focusedUnit !== null) {
      els.list.querySelector(`.unit-btn[data-unit="${CSS.escape(focusedUnit)}"]`)?.focus();
    }
    emitSelection(user);
  }

  function go(requested, { replace = false, user = false } = {}) {
    const next = resolveState(requested, { defaultCrop: DEFAULT_CROP, quartersFor, unitsFor });
    const hash = buildHash(next);
    if (location.hash !== hash) {
      history[replace ? 'replaceState' : 'pushState'](null, '', hash);
    }
    apply(next, { user });
  }

  els.crop.addEventListener('change', () => go({ ...state, crop: els.crop.value, quarter: state.quarter }));
  els.quarter.addEventListener('change', () => go({ ...state, quarter: els.quarter.value }));
  els.prev.addEventListener('click', () => {
    const qs = quartersFor(state.crop);
    go({ ...state, quarter: qs[qs.indexOf(state.quarter) + 1] });
  });
  els.next.addEventListener('click', () => {
    const qs = quartersFor(state.crop);
    go({ ...state, quarter: qs[qs.indexOf(state.quarter) - 1] });
  });
  window.addEventListener('popstate', () => go(parseHash(location.hash), { replace: true }));
  window.addEventListener('hashchange', () => go(parseHash(location.hash), { replace: true }));

  go(parseHash(location.hash), { replace: true });
}

loadData().then(main).catch((err) => {
  console.error(err);
  const box = document.getElementById('load-error');
  box.hidden = false;
  box.textContent = location.protocol === 'file:'
    ? 'AniSight must be served over http (for example `python -m http.server -d app 8000`); browsers block data loading from file:// pages.'
    : `Could not load the data files (${err.message}).`;
});
