// Pure helpers for the province detail panel: chart series, data-table rows, flag
// wording, conditions-vs-normal wording and regrouping notes. No DOM access; every
// number comes from app/data/*.json (nothing is modelled here).
import {
  isNum, formatNumber, formatYield, formatRange, pctWords, quarterKey, quarterLabel,
  quarterIndex, parseQuarter,
} from './format.js';

const MINUS = '\u2212';
const EN_DASH = '\u2013';
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

/** "2026-09-30" -> "30 Sep 2026". Invalid -> input unchanged; missing -> null. */
export function formatDate(iso) {
  if (!iso) return null;
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(String(iso));
  if (!m) return String(iso);
  return `${Number(m[3])} ${MONTHS[Number(m[2]) - 1]} ${m[1]}`;
}

/** Every quarter key from first to last inclusive, oldest first. */
export function quarterSpan(first, last) {
  const a = quarterIndex(first);
  const b = quarterIndex(last);
  if (Number.isNaN(a) || Number.isNaN(b) || b < a) return [];
  const out = [];
  for (let i = a; i <= b; i += 1) out.push(quarterKey(Math.floor(i / 4), (i % 4) + 1));
  return out;
}

/** The same quarter one year earlier: "2026Q4" -> "2025Q4". */
export function sameQuarterLastYear(quarter) {
  const p = parseQuarter(quarter);
  return p ? quarterKey(p.y - 1, p.q) : null;
}

const num = (v) => (isNum(v) ? v : null);

/**
 * Chart series for one unit's history (predictions.json rows) plus the live forecast
 * (forecast.json row, or null). Labels cover every quarter from the first row to the
 * last row (or the forecast quarter), so quarters PSA did not report for this unit
 * (e.g. while it was part of a combined unit) are null gaps, never zeros.
 */
export function buildChartSeries(rows, { forecastQuarter = null, forecastRow = null } = {}) {
  const byKey = new Map((rows ?? []).map((r) => [quarterKey(r.y, r.q), r]));
  const keys = [...byKey.keys()].sort((a, b) => quarterIndex(a) - quarterIndex(b));
  const hasForecast = Boolean(forecastRow && forecastQuarter);
  const first = keys[0] ?? (hasForecast ? forecastQuarter : null);
  let last = keys[keys.length - 1] ?? first;
  if (hasForecast && (!last || quarterIndex(forecastQuarter) > quarterIndex(last))) last = forecastQuarter;
  const labels = first ? quarterSpan(first, last) : [];
  const pick = (field) => labels.map((k) => num(byKey.get(k)?.[field]));
  const lo = pick('lo');
  const hi = pick('hi');
  return {
    labels,
    actual: pick('actual'),
    pred: pick('pred'),
    lo,
    hi,
    hasBand: lo.some((v, i) => v !== null && hi[i] !== null),
    // B0/B2 include the live forecast's components so the toggles extend to it.
    b0: withForecast(pick('b0'), labels, hasForecast && forecastQuarter, forecastRow?.b0),
    b2: withForecast(pick('b2'), labels, hasForecast && forecastQuarter, forecastRow?.b2),
    outlier: labels.map((k) => Boolean(byKey.get(k)?.flags?.includes('outlier'))),
    forecast: hasForecast
      ? {
        x: forecastQuarter,
        y: num(forecastRow.yield),
        lo: num(forecastRow.lo),
        hi: num(forecastRow.hi),
        b0: num(forecastRow.b0),
        b2: num(forecastRow.b2),
      }
      : null,
  };
}

function withForecast(values, labels, quarter, v) {
  if (!quarter) return values;
  const i = labels.indexOf(quarter);
  if (i >= 0 && values[i] === null) values[i] = num(v);
  return values;
}

/**
 * Rows for the chart's text alternative (a data table), newest first. Only quarters
 * with data; the live forecast is the first row when present.
 */
export function chartTableRows(rows, { forecastQuarter = null, forecastRow = null } = {}) {
  const out = (rows ?? [])
    .map((r) => ({
      quarter: quarterKey(r.y, r.q),
      label: quarterLabel(quarterKey(r.y, r.q)),
      kind: 'backtest',
      actual: formatYield(r.actual, { unit: false }),
      b3: formatYield(r.pred, { unit: false }),
      range: formatRange(r.lo, r.hi, { unit: false }),
      b0: formatYield(r.b0, { unit: false }),
      b2: formatYield(r.b2, { unit: false }),
      outlier: Boolean(r.flags?.includes('outlier')),
    }))
    .sort((a, b) => quarterIndex(b.quarter) - quarterIndex(a.quarter));
  if (forecastRow && forecastQuarter) {
    out.unshift({
      quarter: forecastQuarter,
      label: `${quarterLabel(forecastQuarter)} (forecast)`,
      kind: 'forecast',
      actual: 'not yet published',
      b3: formatYield(forecastRow.yield, { unit: false }),
      range: formatRange(forecastRow.lo, forecastRow.hi, { unit: false }),
      b0: formatYield(forecastRow.b0, { unit: false }),
      b2: formatYield(forecastRow.b2, { unit: false }),
      outlier: false,
    });
  }
  return out;
}

/** One-sentence summary of the chart for screen readers. */
export function chartSummary(unit, cropLabel, series) {
  if (!series.labels.length) return `No history for ${unit}.`;
  const n = series.actual.filter(isNum).length;
  const span = `${quarterLabel(series.labels[0])} to ${quarterLabel(series.labels[series.labels.length - 1])}`;
  const f = series.forecast;
  const fc = f && isNum(f.y)
    ? ` The ${quarterLabel(f.x)} forecast is ${formatYield(f.y)} (80% range ${formatRange(f.lo, f.hi)}).`
    : '';
  return `${cropLabel} yield in ${unit}, ${span}: ${n} PSA actual values with the B3 back-test `
    + `prediction and its 80% range.${fc} The same values are in the data table.`;
}

// ---------------------------------------------------------------- flags

const FLAG_TEXT = {
  combined_unit: 'PSA reported several provinces together as one unit for this quarter.',
  b0_fallback: 'No figure for the same quarter last year (the reporting unit changed), so the '
    + '"same quarter last year" part of the forecast used this unit\'s average for the quarter.',
  no_area_forecast: 'Newly regrouped unit with no area history: no area or production forecast.',
  area_unit_mean: 'No harvested area for the same quarter last year; this unit\'s average area was used.',
  outlier: 'The PSA figure for this quarter was flagged as an outlier in our data checks.',
};

/**
 * Plain-language messages for a row's flags. Unknown flags fall back to the
 * definition exported in meta.json (`definitions.flags`), then to the flag name.
 */
export function flagMessages(flags, definitions = {}) {
  return (flags ?? []).map((f) => ({
    flag: f,
    text: FLAG_TEXT[f] ?? definitions?.[f] ?? f.replace(/_/g, ' '),
  }));
}

/** Flags for a live forecast row: its own flags plus area source and partial window. */
export function forecastFlags(row) {
  if (!row) return [];
  const flags = [...(row.flags ?? [])];
  if (row.area_source && row.area_source !== 'last_year' && isNum(row.area_ha)
    && !flags.includes('area_unit_mean')) flags.push('area_unit_mean');
  if (!isNum(row.area_ha) && !flags.includes('no_area_forecast')) flags.push('no_area_forecast');
  return flags;
}

// ---------------------------------------------------------------- conditions

/** NDVI/EVI anomaly (index units) in words, 2 dp: 0.038 -> "0.04 above normal". */
export function indexAnomWords(v, subject) {
  if (!isNum(v)) return null;
  const r = Number(v.toFixed(2));
  if (r === 0) return `${subject} about normal`;
  return `${subject} ${Math.abs(r).toFixed(2)} ${r > 0 ? 'above' : 'below'} normal`;
}

/** Rainfall anomaly in mm with sign, e.g. 335.2 -> "+335 mm". */
export function formatMm(v) {
  if (!isNum(v)) return null;
  const r = Math.round(v);
  if (r === 0) return '0 mm';
  return `${r > 0 ? '+' : MINUS}${formatNumber(Math.abs(r))} mm`;
}

/**
 * Conditions-vs-normal lines for one quarter. `cond` has rain_pct, rain_anom,
 * ndvi_anom, evi_anom and, for the live forecast, window_start/window_end/
 * data_through/partial_window. Returns {lines: [{label, text}], notes: [string]}.
 */
export function conditionsLines(cond) {
  const c = cond ?? {};
  const lines = [];
  const notes = [];
  const rain = pctWords(c.rain_pct, 'Rainfall');
  if (rain) {
    const mm = formatMm(c.rain_anom);
    lines.push({ label: 'Rainfall', text: mm ? `${rain} (${mm})` : rain });
  } else {
    lines.push({ label: 'Rainfall', text: 'Rainfall: not yet available for this quarter' });
  }
  const ndvi = indexAnomWords(c.ndvi_anom, 'Greenness (NDVI)');
  lines.push({ label: 'NDVI', text: ndvi ?? 'Greenness (NDVI): no data' });
  const evi = indexAnomWords(c.evi_anom, 'Greenness (EVI)');
  lines.push({ label: 'EVI', text: evi ?? 'Greenness (EVI): no data' });
  if (c.partial_window) {
    const through = formatDate(c.data_through);
    const span = c.window_start && c.window_end
      ? ` of the ${formatDate(c.window_start)}${EN_DASH}${formatDate(c.window_end)} growing window`
      : ' of the growing window';
    notes.push(`Partial window: satellite imagery covers data through ${through ?? 'an earlier date'}${span}, so these values may change.`);
  }
  return { lines, notes };
}

/** Short flag line for a partial satellite window (the full sentence is in the conditions card). */
export function partialWindowFlag(cond) {
  if (!cond?.partial_window) return null;
  const through = formatDate(cond.data_through);
  return `Partial window: satellite context uses imagery through ${through ?? 'an earlier date'} only `
    + '(see "Conditions vs normal"). The forecast itself does not use it.';
}

// ---------------------------------------------------------------- regroupings

/**
 * Notes about PSA regroupings for a unit, from units.geojson feature properties
 * ({unit, members[], first, last}).
 * - Combined unit: which provinces it covers and when.
 * - Single unit that was part of a combined unit: when, and that those quarters are
 *   not in this unit's history.
 */
export function regroupNotes(unit, features) {
  const props = (features ?? []).map((f) => f.properties ?? f);
  const self = props.find((p) => p.unit === unit);
  const span = (p) => `${quarterLabel(p.first)}${EN_DASH}${quarterLabel(p.last)}`;
  const notes = [];
  if (self && (self.members?.length ?? 0) > 1) {
    notes.push(`Combined reporting unit: PSA reported ${listJoin(self.members)} together `
      + `from ${quarterLabel(self.first)} to ${quarterLabel(self.last)}.`);
    return notes;
  }
  const combos = props
    .filter((p) => p.unit !== unit && (p.members?.length ?? 0) > 1 && p.members.includes(unit))
    .sort((a, b) => quarterIndex(a.first) - quarterIndex(b.first));
  for (const p of combos) {
    const others = p.members.filter((m) => m !== unit);
    notes.push(`${span(p)}: PSA reported ${unit} together with ${listJoin(others)} as `
      + `"${p.unit}". Those quarters are not part of this history.`);
  }
  return notes;
}

function listJoin(xs) {
  if (xs.length <= 1) return xs.join('');
  return `${xs.slice(0, -1).join(', ')} and ${xs[xs.length - 1]}`;
}

// ---------------------------------------------------------------- forecast card

/**
 * Text for the latest-forecast card. Returns null when there is no forecast row.
 * `note` is forecast.json's `area_note`.
 */
export function forecastCard(row, { quarter, note }) {
  if (!row) return null;
  const yieldText = `${formatYield(row.yield)} (80% range ${formatRange(row.lo, row.hi)})`;
  const vsNormal = pctWords(row.pct, '', { normalWord: `the usual ${quarterLabel(quarter).split(' ')[0]}` });
  const lastYear = quarterLabel(sameQuarterLastYear(quarter));
  let area;
  let production;
  if (isNum(row.area_ha)) {
    const source = row.area_source === 'last_year'
      ? `PSA harvested area for ${lastYear}`
      : 'this unit\'s average PSA harvested area (no figure for the same quarter last year)';
    area = `${formatNumber(row.area_ha)} ha (${source}; not a forecast of planted area)`;
    production = isNum(row.production_mt)
      ? `${formatNumber(row.production_mt)} t`
        + (isNum(row.production_lo_mt) && isNum(row.production_hi_mt)
          ? ` (80% range ${formatNumber(row.production_lo_mt)}${EN_DASH}${formatNumber(row.production_hi_mt)} t)`
          : '')
      : 'no production forecast';
  } else {
    area = 'No area forecast for this unit (newly regrouped, no area history)';
    production = 'no production forecast';
  }
  const model = `${row.model ?? 'B3'} blends "same quarter last year" (B0: ${formatYield(row.b0)}) `
    + `and "average + trend" (B2: ${formatYield(row.b2)}), both from PSA history.`;
  return {
    yieldText,
    vsNormal: vsNormal ? `About ${vsNormal}.` : 'No usual yield yet (fewer than 3 years of history).',
    area,
    production,
    note: note ?? null,
    model,
  };
}

/** Back-test sentence for a historical quarter: actual vs B3 out-of-fold and its range. */
export function backtestSentence(row) {
  if (!row) return null;
  const range = isNum(row.lo) && isNum(row.hi) ? ` (80% range ${formatRange(row.lo, row.hi)})` : ' (no 80% range for this quarter)';
  let inside = '';
  if (isNum(row.lo) && isNum(row.hi) && isNum(row.actual)) {
    inside = row.actual >= row.lo && row.actual <= row.hi
      ? ' The actual fell inside the 80% range.'
      : ` The actual fell ${row.actual < row.lo ? 'below' : 'above'} the 80% range.`;
  }
  return `PSA actual ${formatYield(row.actual)}. B3 back-test ${formatYield(row.pred)}${range}, `
    + `made without that year's PSA data.${inside}`;
}
