// Pure helpers for AniSight: formatting, quarter handling, colour scale, hash state and
// selecting the rows to draw. No DOM access here, so everything is testable with
// `node --test app/tests`.

export const CROPS = ['palay', 'corn'];
export const CROP_LABELS = { palay: 'Palay (rice)', corn: 'Corn' };

const MINUS = '\u2212'; // typographic minus sign
const EN_DASH = '\u2013';

// ---------------------------------------------------------------- numbers

/** True for finite numbers; null/undefined/NaN/strings are "missing". */
export function isNum(v) {
  return typeof v === 'number' && Number.isFinite(v);
}

/** Thousands-separated number, e.g. 17427 -> "17,427". Missing -> "—". */
export function formatNumber(v, digits = 0) {
  if (!isNum(v)) return '\u2014';
  return v.toLocaleString('en-US', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).replace('-', MINUS);
}

/** Yield in t/ha with 2 decimals, e.g. 4.213 -> "4.21 t/ha". */
export function formatYield(v, { unit = true } = {}) {
  if (!isNum(v)) return '\u2014';
  const s = formatNumber(v, 2);
  return unit ? `${s} t/ha` : s;
}

/** Range of two yields, e.g. "3.95–4.48 t/ha". Missing ends -> "—". */
export function formatRange(lo, hi, { unit = true } = {}) {
  if (!isNum(lo) || !isNum(hi)) return '\u2014';
  const s = `${formatNumber(lo, 2)}${EN_DASH}${formatNumber(hi, 2)}`;
  return unit ? `${s} t/ha` : s;
}

/** Signed percent, rounded: 6.4 -> "+6%", -6.4 -> "−6%", 0.2 -> "0%". */
export function formatPct(v, digits = 0) {
  if (!isNum(v)) return '\u2014';
  const r = Number(v.toFixed(digits));
  if (r === 0) return `${(0).toFixed(digits)}%`;
  const s = Math.abs(r).toFixed(digits);
  return `${r > 0 ? '+' : MINUS}${s}%`;
}

/**
 * Plain-language comparison with normal.
 * pctWords(6.4) -> "6% above normal"; pctWords(-32, 'Rainfall') -> "Rainfall 32% below normal".
 * Values that round to 0 -> "about normal". Missing -> null (caller decides wording).
 */
export function pctWords(v, subject = '', { normalWord = 'normal' } = {}) {
  if (!isNum(v)) return null;
  const r = Math.round(v);
  const prefix = subject ? `${subject} ` : '';
  if (r === 0) return `${prefix}about ${normalWord}`;
  return `${prefix}${Math.abs(r)}% ${r > 0 ? 'above' : 'below'} ${normalWord}`;
}

// ---------------------------------------------------------------- quarters

/** "2026Q4" -> {y: 2026, q: 4}; invalid -> null. */
export function parseQuarter(s) {
  const m = /^(\d{4})Q([1-4])$/.exec(String(s ?? '').trim().toUpperCase());
  return m ? { y: Number(m[1]), q: Number(m[2]) } : null;
}

/** (2026, 4) -> "2026Q4". */
export function quarterKey(y, q) {
  return `${y}Q${q}`;
}

/** "2026Q4" -> "Q4 2026" (display form). Invalid -> the input unchanged. */
export function quarterLabel(s) {
  const p = parseQuarter(s);
  return p ? `Q${p.q} ${p.y}` : String(s);
}

/** Sortable integer index, e.g. 2026Q4 -> 2026*4+3. */
export function quarterIndex(s) {
  const p = parseQuarter(s);
  return p ? p.y * 4 + (p.q - 1) : NaN;
}

/** a within [first, last] inclusive (all "YYYYQn"). Missing bounds are open. */
export function quarterInRange(a, first, last) {
  const i = quarterIndex(a);
  if (Number.isNaN(i)) return false;
  if (first && i < quarterIndex(first)) return false;
  if (last && i > quarterIndex(last)) return false;
  return true;
}

/**
 * All quarters with data for a crop, newest first. The forecast quarter (from
 * forecast.json) is included when it has rows for the crop.
 */
export function availableQuarters(predictions, forecast, crop) {
  const keys = new Set();
  for (const rows of Object.values(predictions?.[crop] ?? {})) {
    for (const r of rows) keys.add(quarterKey(r.y, r.q));
  }
  const fq = forecast?.target_quarter;
  if (fq && Object.keys(forecast?.forecasts?.[crop] ?? {}).length) keys.add(fq);
  return [...keys].sort((a, b) => quarterIndex(b) - quarterIndex(a));
}

// ---------------------------------------------------------------- map rows

/**
 * Rows to draw for one crop and quarter.
 * - The forecast quarter uses forecast.json (kind "forecast", value = B3 forecast).
 * - Other quarters use predictions.json (kind "backtest", value = B3 out-of-fold
 *   prediction; PSA actual kept alongside).
 * A unit is drawn only if it has a row for that quarter AND a feature whose
 * first..last range contains the quarter. Rows without a matching feature are
 * returned in `missingGeometry` so the caller can report them.
 * `pct` is the exported "% vs historical normal" (never computed here).
 */
export function rowsForQuarter({ predictions, forecast, features, crop, quarter }) {
  const featureByUnit = new Map(
    (features ?? []).map((f) => [f.properties.unit, f]),
  );
  const out = [];
  const missingGeometry = [];
  const isForecast = quarter === forecast?.target_quarter;
  const push = (unit, rec) => {
    const f = featureByUnit.get(unit);
    if (!f || !quarterInRange(quarter, f.properties.first, f.properties.last)) {
      missingGeometry.push(unit);
      return;
    }
    out.push({ unit, feature: f, ...rec });
  };

  if (isForecast) {
    for (const [unit, r] of Object.entries(forecast?.forecasts?.[crop] ?? {})) {
      push(unit, {
        kind: 'forecast',
        value: isNum(r.yield) ? r.yield : null,
        lo: isNum(r.lo) ? r.lo : null,
        hi: isNum(r.hi) ? r.hi : null,
        actual: null,
        normal: isNum(r.normal) ? r.normal : null,
        pct: isNum(r.pct) ? r.pct : null,
        flags: r.flags ?? [],
        row: r,
      });
    }
  } else {
    const p = parseQuarter(quarter);
    if (p) {
      for (const [unit, rows] of Object.entries(predictions?.[crop] ?? {})) {
        const r = rows.find((x) => x.y === p.y && x.q === p.q);
        if (!r) continue;
        push(unit, {
          kind: 'backtest',
          value: isNum(r.pred) ? r.pred : null,
          lo: isNum(r.lo) ? r.lo : null,
          hi: isNum(r.hi) ? r.hi : null,
          actual: isNum(r.actual) ? r.actual : null,
          normal: isNum(r.normal) ? r.normal : null,
          pct: isNum(r.pct) ? r.pct : null,
          flags: r.flags ?? [],
          row: r,
        });
      }
    }
  }
  out.sort((a, b) => a.unit.localeCompare(b.unit));
  return { rows: out, missingGeometry };
}

// ---------------------------------------------------------------- colour scale

// ColorBrewer RdYlBu, 11 classes, reversed so that blue = above normal and
// red = below normal. Colour-blind safe (ColorBrewer). Source: colorbrewer2.org.
export const RDYLBU_11 = [
  '#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', '#ffffbf',
  '#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#313695',
];
export const NO_DATA_COLOR = '#bdbdbd';
export const PCT_CLAMP = 25;

function hexToRgb(hex) {
  const h = hex.replace('#', '');
  return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16));
}
function rgbToHex(rgb) {
  return `#${rgb.map((c) => Math.round(c).toString(16).padStart(2, '0')).join('')}`;
}

/**
 * Diverging colour for a % difference from normal, centred at 0 and clamped at
 * ±clamp. Linear interpolation between the RdYlBu stops. Missing -> NO_DATA_COLOR.
 */
export function colorForPct(pct, clamp = PCT_CLAMP, stops = RDYLBU_11) {
  if (!isNum(pct)) return NO_DATA_COLOR;
  const t = (Math.max(-clamp, Math.min(clamp, pct)) + clamp) / (2 * clamp); // 0..1
  const pos = t * (stops.length - 1);
  const i = Math.min(Math.floor(pos), stops.length - 2);
  const f = pos - i;
  const a = hexToRgb(stops[i]);
  const b = hexToRgb(stops[i + 1]);
  return rgbToHex(a.map((c, k) => c + (b[k] - c) * f));
}

/** WCAG relative luminance of a "#rrggbb" colour. */
export function luminance(hex) {
  const [r, g, b] = hexToRgb(hex).map((c) => {
    const s = c / 255;
    return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

/** WCAG contrast ratio between two colours. */
export function contrastRatio(a, b) {
  const [l1, l2] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (l1 + 0.05) / (l2 + 0.05);
}

/** Black or white text, whichever contrasts more with the background. */
export function textColorOn(bg) {
  return contrastRatio(bg, '#000000') >= contrastRatio(bg, '#ffffff') ? '#000000' : '#ffffff';
}

/** CSS linear-gradient for the legend bar (left = below normal). */
export function legendGradient(stops = RDYLBU_11) {
  const n = stops.length - 1;
  const parts = stops.map((c, i) => `${c} ${((i / n) * 100).toFixed(1)}%`);
  return `linear-gradient(to right, ${parts.join(', ')})`;
}

/** Legend tick labels from -clamp to +clamp, e.g. ["≤−25%", "−10%", "0%", "+10%", "≥+25%"]. */
export function legendTicks(clamp = PCT_CLAMP, values = [-clamp, -10, 0, 10, clamp]) {
  return values.map((v) => {
    const label = formatPct(v);
    if (v <= -clamp) return { value: v, label: `\u2264${label}` };
    if (v >= clamp) return { value: v, label: `\u2265${label}` };
    return { value: v, label };
  });
}

// ---------------------------------------------------------------- URL hash state

/** "#corn/2026Q4/Davao%20del%20Norte" -> {crop, quarter, unit} (missing parts null). */
export function parseHash(hash) {
  const parts = String(hash ?? '').replace(/^#/, '').split('/');
  const dec = (s) => {
    if (!s) return null;
    try {
      return decodeURIComponent(s);
    } catch {
      return null;
    }
  };
  const crop = dec(parts[0]);
  const quarter = dec(parts[1]);
  return {
    crop: CROPS.includes(crop) ? crop : null,
    quarter: parseQuarter(quarter) ? quarter.toUpperCase() : null,
    unit: dec(parts.slice(2).join('/')),
  };
}

/** {crop, quarter, unit} -> "#corn/2026Q4/Davao%20del%20Norte". */
export function buildHash({ crop, quarter, unit }) {
  const parts = [crop, quarter].map((s) => encodeURIComponent(s ?? ''));
  if (unit) parts.push(encodeURIComponent(unit));
  return `#${parts.join('/')}`;
}

/**
 * Resolve requested state against what exists: unknown crop -> default crop,
 * unknown quarter -> latest available, unit not drawn in that quarter -> null.
 */
export function resolveState(requested, { defaultCrop, quartersFor, unitsFor }) {
  const crop = CROPS.includes(requested.crop) ? requested.crop : defaultCrop;
  const quarters = quartersFor(crop);
  const quarter = quarters.includes(requested.quarter) ? requested.quarter : (quarters[0] ?? null);
  const units = quarter ? unitsFor(crop, quarter) : [];
  const unit = units.includes(requested.unit) ? requested.unit : null;
  return { crop, quarter, unit };
}
