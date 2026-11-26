// Run with: node --test app/tests
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  formatNumber, formatYield, formatRange, formatPct, pctWords,
  parseQuarter, quarterKey, quarterLabel, quarterIndex, quarterInRange,
  availableQuarters, rowsForQuarter,
  colorForPct, NO_DATA_COLOR, RDYLBU_11, contrastRatio, textColorOn,
  legendGradient, legendTicks,
  parseHash, buildHash, resolveState,
} from '../js/lib/format.js';

test('formatNumber / formatYield / formatRange', () => {
  assert.equal(formatNumber(17427), '17,427');
  assert.equal(formatNumber(-1234.5, 1), '\u22121,234.5');
  assert.equal(formatNumber(null), '\u2014');
  assert.equal(formatNumber(NaN), '\u2014');
  assert.equal(formatYield(4.213), '4.21 t/ha');
  assert.equal(formatYield(4.213, { unit: false }), '4.21');
  assert.equal(formatYield(undefined), '\u2014');
  assert.equal(formatRange(3.951, 4.478), '3.95\u20134.48 t/ha');
  assert.equal(formatRange(3.9, null), '\u2014');
});

test('formatPct signs and rounding', () => {
  assert.equal(formatPct(6.4), '+6%');
  assert.equal(formatPct(-6.4), '\u22126%');
  assert.equal(formatPct(0.2), '0%');
  assert.equal(formatPct(-0.4), '0%');
  assert.equal(formatPct(27.16, 1), '+27.2%');
  assert.equal(formatPct(null), '\u2014');
});

test('pctWords plain language', () => {
  assert.equal(pctWords(6.4), '6% above normal');
  assert.equal(pctWords(-32, 'Rainfall'), 'Rainfall 32% below normal');
  assert.equal(pctWords(0.3), 'about normal');
  assert.equal(pctWords(null), null);
});

test('quarter helpers', () => {
  assert.deepEqual(parseQuarter('2026Q4'), { y: 2026, q: 4 });
  assert.deepEqual(parseQuarter('2026q1'), { y: 2026, q: 1 });
  assert.equal(parseQuarter('2026Q5'), null);
  assert.equal(parseQuarter(null), null);
  assert.equal(quarterKey(2012, 1), '2012Q1');
  assert.equal(quarterLabel('2026Q4'), 'Q4 2026');
  assert.ok(quarterIndex('2026Q1') > quarterIndex('2025Q4'));
  assert.ok(quarterInRange('2021Q1', '2021Q1', '2025Q1'));
  assert.ok(quarterInRange('2025Q1', '2021Q1', '2025Q1'));
  assert.ok(!quarterInRange('2025Q2', '2021Q1', '2025Q1'));
  assert.ok(!quarterInRange('bad', '2021Q1', '2025Q1'));
});

// Tiny synthetic fixture mirroring the export schema.
const feat = (unit, first, last) => ({
  type: 'Feature', properties: { unit, members: unit.split('+'), first, last }, geometry: null,
});
const features = [
  feat('A', '2012Q1', '2026Q4'),
  feat('B', '2019Q1', '2026Q4'),
  feat('A+B', '2012Q1', '2018Q4'),
];
const row = (y, q, extra = {}) => ({
  y, q, actual: 2, pred: 2.1, lo: 1.8, hi: 2.4, b0: 2, b2: 2.2, normal: 2, pct: 5, flags: [], ...extra,
});
const predictions = {
  corn: {
    A: [row(2018, 4), row(2019, 1, { pct: -3 })],
    B: [row(2019, 1, { pct: null, normal: null })],
    'A+B': [row(2018, 4, { flags: ['combined_unit'] })],
    C: [row(2019, 1)], // no geometry
  },
  palay: {},
};
const forecast = {
  target_quarter: '2019Q2',
  forecasts: { corn: { A: { yield: 2.5, lo: 2.1, hi: 2.9, normal: 2, pct: 25, flags: [] } }, palay: {} },
};

test('availableQuarters is newest first and includes the forecast quarter', () => {
  assert.deepEqual(availableQuarters(predictions, forecast, 'corn'), ['2019Q2', '2019Q1', '2018Q4']);
  assert.deepEqual(availableQuarters(predictions, forecast, 'palay'), []);
});

test('rowsForQuarter: combined unit before the split, split units after', () => {
  const r2018 = rowsForQuarter({ predictions, forecast, features, crop: 'corn', quarter: '2018Q4' });
  assert.deepEqual(r2018.rows.map((r) => r.unit), ['A', 'A+B']);
  assert.ok(r2018.rows.every((r) => r.kind === 'backtest'));
  assert.equal(r2018.rows[0].value, 2.1);
  assert.equal(r2018.rows[0].actual, 2);

  const r2019 = rowsForQuarter({ predictions, forecast, features, crop: 'corn', quarter: '2019Q1' });
  assert.deepEqual(r2019.rows.map((r) => r.unit), ['A', 'B']);
  assert.equal(r2019.rows[1].pct, null);
  assert.deepEqual(r2019.missingGeometry, ['C']);
});

test('rowsForQuarter: forecast quarter uses forecast.json', () => {
  const r = rowsForQuarter({ predictions, forecast, features, crop: 'corn', quarter: '2019Q2' });
  assert.equal(r.rows.length, 1);
  assert.equal(r.rows[0].kind, 'forecast');
  assert.equal(r.rows[0].value, 2.5);
  assert.equal(r.rows[0].actual, null);
  assert.equal(r.rows[0].pct, 25);
});

test('colorForPct: centred, clamped, missing', () => {
  assert.equal(colorForPct(0), '#ffffbf');
  assert.equal(colorForPct(-25), RDYLBU_11[0]);
  assert.equal(colorForPct(25), RDYLBU_11[10]);
  assert.equal(colorForPct(-80), RDYLBU_11[0]);
  assert.equal(colorForPct(80), RDYLBU_11[10]);
  assert.equal(colorForPct(null), NO_DATA_COLOR);
  assert.equal(colorForPct(NaN), NO_DATA_COLOR);
  assert.match(colorForPct(7.3), /^#[0-9a-f]{6}$/);
  // above normal is bluer than below normal
  const blue = (h) => parseInt(h.slice(5, 7), 16);
  assert.ok(blue(colorForPct(10)) > blue(colorForPct(-10)));
});

test('text colour keeps WCAG AA contrast on every scale colour', () => {
  for (const c of [...RDYLBU_11, NO_DATA_COLOR]) {
    assert.ok(contrastRatio(c, textColorOn(c)) >= 4.5, c);
  }
  assert.equal(textColorOn('#ffffbf'), '#000000');
  assert.equal(textColorOn('#313695'), '#ffffff');
});

test('legend helpers', () => {
  assert.match(legendGradient(), /^linear-gradient\(to right, #a50026 0\.0%, .* #313695 100\.0%\)$/);
  const ticks = legendTicks();
  assert.equal(ticks[0].label, '\u2264\u221225%');
  assert.equal(ticks[2].label, '0%');
  assert.equal(ticks[4].label, '\u2265+25%');
});

test('hash round-trip and validation', () => {
  const h = buildHash({ crop: 'corn', quarter: '2026Q4', unit: 'Davao del Sur+Davao City' });
  assert.equal(h, '#corn/2026Q4/Davao%20del%20Sur%2BDavao%20City');
  assert.deepEqual(parseHash(h), { crop: 'corn', quarter: '2026Q4', unit: 'Davao del Sur+Davao City' });
  assert.deepEqual(parseHash('#corn/2026Q4/Davao%20del%20Norte'), {
    crop: 'corn', quarter: '2026Q4', unit: 'Davao del Norte',
  });
  assert.deepEqual(parseHash(''), { crop: null, quarter: null, unit: null });
  assert.deepEqual(parseHash('#wheat/2026Q9/%E0%A4%A'), { crop: null, quarter: null, unit: null });
  assert.equal(buildHash({ crop: 'palay', quarter: '2020Q1' }), '#palay/2020Q1');
});

test('resolveState falls back to defaults', () => {
  const ctx = {
    defaultCrop: 'palay',
    quartersFor: (c) => (c === 'corn' ? ['2026Q4', '2026Q3'] : ['2026Q4']),
    unitsFor: (c, q) => (q === '2026Q3' ? ['A'] : ['A', 'B']),
  };
  assert.deepEqual(resolveState({ crop: null, quarter: null, unit: null }, ctx),
    { crop: 'palay', quarter: '2026Q4', unit: null });
  assert.deepEqual(resolveState({ crop: 'corn', quarter: '2026Q3', unit: 'B' }, ctx),
    { crop: 'corn', quarter: '2026Q3', unit: null });
  assert.deepEqual(resolveState({ crop: 'corn', quarter: '1999Q1', unit: 'B' }, ctx),
    { crop: 'corn', quarter: '2026Q4', unit: 'B' });
});
