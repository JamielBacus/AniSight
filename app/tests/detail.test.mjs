// Run with: node --test 'app/tests/*.test.mjs'
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  formatDate, quarterSpan, sameQuarterLastYear, buildChartSeries, chartTableRows, chartSummary,
  flagMessages, forecastFlags, indexAnomWords, formatMm, conditionsLines, regroupNotes,
  forecastCard, backtestSentence,
} from '../js/lib/detail.js';

const row = (y, q, extra = {}) => ({
  y, q, actual: 3, pred: 3.1, lo: 2.8, hi: 3.4, b0: 3.2, b2: 3.0, flags: [], ...extra,
});

const fRow = {
  model: 'B3', data_vintage: '2026Q3', yield: 2.732, lo: 2.277, hi: 3.225, b0: 2.949, b2: 2.532,
  area_ha: 1849, area_source: 'last_year', production_mt: 5052, production_lo_mt: 4211,
  production_hi_mt: 5963, pct: 27.1, flags: [],
  conditions: {
    rain_anom: null, rain_pct: null, ndvi_anom: 0.038, evi_anom: -0.095,
    window_start: '2026-07-01', window_end: '2026-12-31', data_through: '2026-09-30', partial_window: true,
  },
};

test('formatDate, quarterSpan, sameQuarterLastYear', () => {
  assert.equal(formatDate('2026-09-30'), '30 Sep 2026');
  assert.equal(formatDate(null), null);
  assert.deepEqual(quarterSpan('2020Q3', '2021Q2'), ['2020Q3', '2020Q4', '2021Q1', '2021Q2']);
  assert.deepEqual(quarterSpan('2021Q2', '2020Q3'), []);
  assert.equal(sameQuarterLastYear('2026Q4'), '2025Q4');
});

test('buildChartSeries: null band stays null and missing quarters are gaps', () => {
  const rows = [
    row(2014, 4, { lo: null, hi: null }),
    row(2015, 1),
    // 2015Q2..2015Q3 missing (unit regrouped)
    row(2015, 4, { flags: ['outlier'] }),
  ];
  const s = buildChartSeries(rows);
  assert.deepEqual(s.labels, ['2014Q4', '2015Q1', '2015Q2', '2015Q3', '2015Q4']);
  assert.deepEqual(s.lo, [null, 2.8, null, null, 2.8]);
  assert.deepEqual(s.actual, [3, 3, null, null, 3]);
  assert.deepEqual(s.outlier, [false, false, false, false, true]);
  assert.equal(s.hasBand, true);
  assert.equal(s.forecast, null);
  assert.ok(!s.lo.includes(0), 'no zeros substituted for missing intervals');
});

test('buildChartSeries: forecast quarter appended with its B0/B2', () => {
  const s = buildChartSeries([row(2026, 2), row(2026, 3)], { forecastQuarter: '2026Q4', forecastRow: fRow });
  assert.deepEqual(s.labels, ['2026Q2', '2026Q3', '2026Q4']);
  assert.equal(s.pred[2], null);
  assert.equal(s.actual[2], null);
  assert.equal(s.b0[2], 2.949);
  assert.equal(s.b2[2], 2.532);
  assert.deepEqual(s.forecast, { x: '2026Q4', y: 2.732, lo: 2.277, hi: 3.225, b0: 2.949, b2: 2.532 });
});

test('buildChartSeries: no band at all', () => {
  const s = buildChartSeries([row(2012, 1, { lo: null, hi: null })]);
  assert.equal(s.hasBand, false);
});

test('chartTableRows newest first, forecast on top; chartSummary', () => {
  const rows = chartTableRows([row(2026, 2), row(2026, 3, { flags: ['outlier'] })],
    { forecastQuarter: '2026Q4', forecastRow: fRow });
  assert.deepEqual(rows.map((r) => r.quarter), ['2026Q4', '2026Q3', '2026Q2']);
  assert.equal(rows[0].kind, 'forecast');
  assert.equal(rows[0].actual, 'not yet published');
  assert.equal(rows[0].range, '2.28\u20133.23');
  assert.equal(rows[1].outlier, true);
  const s = buildChartSeries([row(2026, 2)], { forecastQuarter: '2026Q4', forecastRow: fRow });
  const text = chartSummary('Davao City', 'Corn', s);
  assert.match(text, /Q2 2026 to Q4 2026/);
  assert.match(text, /1 PSA actual values/);
  assert.match(text, /Q4 2026 forecast is 2\.73 t\/ha/);
  assert.equal(chartSummary('X', 'Corn', buildChartSeries([])), 'No history for X.');
});

test('flagMessages: known wording, meta fallback, name fallback', () => {
  const m = flagMessages(['b0_fallback', 'new_flag', 'other_thing'], { new_flag: 'from meta' });
  assert.match(m[0].text, /same quarter last year/);
  assert.equal(m[1].text, 'from meta');
  assert.equal(m[2].text, 'other thing');
  assert.deepEqual(flagMessages(undefined), []);
});

test('forecastFlags adds area fallbacks', () => {
  assert.deepEqual(forecastFlags(fRow), []);
  assert.deepEqual(forecastFlags({ ...fRow, area_source: 'unit_mean' }), ['area_unit_mean']);
  assert.deepEqual(forecastFlags({ ...fRow, area_ha: null, area_source: null }), ['no_area_forecast']);
  assert.deepEqual(forecastFlags(null), []);
});

test('indexAnomWords and formatMm', () => {
  assert.equal(indexAnomWords(0.038, 'NDVI'), 'NDVI 0.04 above normal');
  assert.equal(indexAnomWords(-0.095, 'EVI'), 'EVI 0.10 below normal');
  assert.equal(indexAnomWords(0.003, 'EVI'), 'EVI about normal');
  assert.equal(indexAnomWords(null, 'EVI'), null);
  assert.equal(formatMm(335.2), '+335 mm');
  assert.equal(formatMm(-1234.4), '\u22121,234 mm');
  assert.equal(formatMm(null), null);
});

test('conditionsLines: forecast quarter with partial window and no rainfall', () => {
  const { lines, notes } = conditionsLines(fRow.conditions);
  assert.equal(lines[0].text, 'Rainfall: not yet available for this quarter');
  assert.equal(lines[1].text, 'Greenness (NDVI) 0.04 above normal');
  assert.equal(lines[2].text, 'Greenness (EVI) 0.10 below normal');
  assert.equal(notes.length, 1);
  assert.match(notes[0], /data through 30 Sep 2026/);
  assert.match(notes[0], /1 Jul 2026\u201331 Dec 2026/);
});

test('conditionsLines: historical quarter', () => {
  const { lines, notes } = conditionsLines({ rain_pct: 43.8, rain_anom: 335.2, ndvi_anom: -0.004, evi_anom: null });
  assert.equal(lines[0].text, 'Rainfall 44% above normal (+335 mm)');
  assert.equal(lines[1].text, 'Greenness (NDVI) about normal');
  assert.equal(lines[2].text, 'Greenness (EVI): no data');
  assert.deepEqual(notes, []);
});

const features = [
  { properties: { unit: 'Davao City', members: ['Davao City'], first: '2012Q1', last: '2026Q4' } },
  { properties: { unit: 'Davao del Sur', members: ['Davao del Sur'], first: '2019Q1', last: '2026Q4' } },
  { properties: { unit: 'Davao del Sur+Davao City', members: ['Davao del Sur', 'Davao City'], first: '2021Q1', last: '2025Q1' } },
  { properties: { unit: 'Davao del Sur+Davao Occidental', members: ['Davao del Sur', 'Davao Occidental'], first: '2012Q1', last: '2018Q4' } },
  { properties: { unit: 'Davao Oriental', members: ['Davao Oriental'], first: '2012Q1', last: '2026Q4' } },
];

test('regroupNotes', () => {
  const city = regroupNotes('Davao City', features);
  assert.equal(city.length, 1);
  assert.match(city[0], /^Q1 2021\u2013Q1 2025: PSA reported Davao City together with Davao del Sur as "Davao del Sur\+Davao City"/);
  const sur = regroupNotes('Davao del Sur', features);
  assert.equal(sur.length, 2);
  assert.match(sur[0], /^Q1 2012\u2013Q4 2018/);
  const combo = regroupNotes('Davao del Sur+Davao City', features);
  assert.deepEqual(combo, ['Combined reporting unit: PSA reported Davao del Sur and Davao City together from Q1 2021 to Q1 2025.']);
  assert.deepEqual(regroupNotes('Davao Oriental', features), []);
});

test('forecastCard: area, production range, note and model wording', () => {
  const c = forecastCard(fRow, { quarter: '2026Q4', note: 'assumes the same harvested area as the same quarter last year' });
  assert.equal(c.yieldText, '2.73 t/ha (80% range 2.28\u20133.23 t/ha)');
  assert.equal(c.vsNormal, 'About 27% above the usual Q4.');
  assert.match(c.area, /^1,849 ha \(PSA harvested area for Q4 2025; not a forecast of planted area\)$/);
  assert.equal(c.production, '5,052 t (80% range 4,211\u20135,963 t)');
  assert.match(c.model, /B3 blends "same quarter last year" \(B0: 2\.95 t\/ha\) and "average \+ trend" \(B2: 2\.53 t\/ha\)/);
  assert.doesNotMatch(JSON.stringify(c), /satellite/i);
  assert.equal(forecastCard(null, { quarter: '2026Q4' }), null);
});

test('forecastCard: no area forecast and unit-mean area', () => {
  const none = forecastCard({ ...fRow, area_ha: null, production_mt: null }, { quarter: '2026Q4' });
  assert.match(none.area, /No area forecast/);
  assert.equal(none.production, 'no production forecast');
  const mean = forecastCard({ ...fRow, area_source: 'unit_mean' }, { quarter: '2026Q4' });
  assert.match(mean.area, /average PSA harvested area/);
  const noNormal = forecastCard({ ...fRow, pct: null }, { quarter: '2026Q4' });
  assert.match(noNormal.vsNormal, /No usual yield yet/);
});

test('backtestSentence', () => {
  assert.match(backtestSentence(row(2026, 3, { actual: 3.911, pred: 4.145, lo: 3.789, hi: 4.58 })),
    /PSA actual 3\.91 t\/ha\. B3 back-test 4\.15 t\/ha \(80% range 3\.79\u20134\.58 t\/ha\).*inside the 80% range/);
  assert.match(backtestSentence(row(2026, 3, { actual: 5 })), /above the 80% range/);
  assert.match(backtestSentence(row(2013, 1, { lo: null, hi: null })), /no 80% range for this quarter/);
  assert.equal(backtestSentence(null), null);
});
