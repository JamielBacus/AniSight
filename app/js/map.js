// Leaflet map: reporting-unit polygons coloured by B3 % vs historical normal,
// a legend, and an optional basemap. Leaflet is loaded as a classic script
// (window.L) from app/vendor/.
import {
  colorForPct, textColorOn, legendGradient, legendTicks, NO_DATA_COLOR, PCT_CLAMP,
  formatPct, formatYield, formatRange, pctWords, quarterLabel,
} from './lib/format.js';

const L = globalThis.L;

// OpenStreetMap standard tiles (design.md allows CARTO Positron or OSM). CARTO now
// returns "API KEY REQUIRED" placeholder tiles without a key, so OSM is used.
// OSM tile policy: attribution required, light use only, browser sends the Referer.
const TILE_URL = 'https://tile.openstreetmap.org/{z}/{x}/{y}.png';
const TILE_ATTRIBUTION =
  '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors';

const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) => (
  { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
));

/** Short hover/focus text for one unit. */
export function describeRow(r, quarter) {
  const q = quarterLabel(quarter).split(' ')[0]; // "Q4"
  const lines = [`<strong>${escapeHtml(r.unit)}</strong>`];
  if (r.kind === 'forecast') {
    lines.push(`Forecast ${formatYield(r.value)} (likely ${formatRange(r.lo, r.hi, { unit: false })})`);
  } else {
    lines.push(`Back-test forecast ${formatYield(r.value)}`);
    lines.push(`PSA actual ${formatYield(r.actual)}`);
  }
  const words = pctWords(r.pct, '', { normalWord: `this unit's usual ${q}` });
  lines.push(words ? `Forecast ${words}` : 'No historical normal yet (fewer than 3 years)');
  if (r.flags.includes('combined_unit')) lines.push('PSA reported these provinces together');
  return lines.join('<br>');
}

export function createMap(el, { meta, onSelect, legendEl }) {
  const map = L.map(el, {
    zoomSnap: 0.25,
    minZoom: 6,
    maxZoom: 12,
    attributionControl: true,
  });

  // Data attribution (from meta.json) stays visible whether or not tiles load.
  const bySource = (needle) => meta?.sources?.find((s) => s.name.includes(needle));
  const psa = bySource('PSA');
  if (psa) map.attributionControl.addAttribution(escapeHtml(psa.attribution));
  const gb = bySource('geoBoundaries');
  if (gb) {
    map.attributionControl.addAttribution(
      `Boundaries: <a href="${escapeHtml(gb.url)}">geoBoundaries</a> (${escapeHtml(gb.licence.split(' (')[0])})`,
    );
  }

  // Basemap is optional: polygons sit on a plain background if tiles fail.
  const tiles = L.tileLayer(TILE_URL, {
    attribution: TILE_ATTRIBUTION,
    maxZoom: 19,
    opacity: 0.55, // keep the busy OSM style behind the choropleth
  });
  let tileErrors = 0;
  let tileLoads = 0;
  const notice = document.getElementById('tile-notice');
  tiles.on('tileerror', () => {
    tileErrors += 1;
    if (notice && tileLoads === 0) notice.hidden = false;
  });
  tiles.on('tileload', () => {
    tileLoads += 1;
    if (notice) notice.hidden = true;
  });
  tiles.addTo(map);

  addLegend(legendEl);

  let layer = null;
  let selected = null;
  const layersByUnit = new Map();

  function styleFor(r) {
    const isSel = r.unit === selected;
    const noData = r.pct === null;
    return {
      fillColor: colorForPct(r.pct),
      fillOpacity: 0.85,
      color: isSel ? '#000000' : '#333333',
      weight: isSel ? 3 : 1,
      dashArray: noData ? '4 3' : null, // non-colour cue for "no normal"
    };
  }

  /** Redraw for a list of rows from rowsForQuarter(). */
  function render(rows, { quarter, selectedUnit }) {
    selected = selectedUnit ?? null;
    if (layer) layer.remove();
    layersByUnit.clear();
    const fc = { type: 'FeatureCollection', features: rows.map((r) => r.feature) };
    const rowByUnit = new Map(rows.map((r) => [r.unit, r]));
    layer = L.geoJSON(fc, {
      style: (f) => styleFor(rowByUnit.get(f.properties.unit)),
      onEachFeature: (f, lyr) => {
        const r = rowByUnit.get(f.properties.unit);
        layersByUnit.set(r.unit, lyr);
        lyr.bindTooltip(describeRow(r, quarter), { sticky: true, direction: 'top' });
        lyr.on('click', () => onSelect?.(r.unit));
        // Permanent value label: colour is never the only cue.
        const label = r.pct === null ? 'n/a' : formatPct(r.pct);
        const bg = colorForPct(r.pct);
        lyr.labelMarker = L.marker(lyr.getBounds().getCenter(), {
          interactive: false,
          keyboard: false,
          icon: L.divIcon({
            className: 'unit-label',
            html: `<span style="background:${bg};color:${textColorOn(bg)}">${escapeHtml(label)}</span>`,
            iconSize: null,
          }),
        });
      },
    }).addTo(map);
    // Labels placed at a point inside the polygon when the bounds centre falls outside.
    for (const lyr of layersByUnit.values()) {
      const c = lyr.getBounds().getCenter();
      if (!containsPoint(lyr, c)) lyr.labelMarker.setLatLng(interiorPoint(lyr));
      lyr.labelMarker.addTo(layer);
    }
    if (!map._anisightFitted && rows.length) {
      map.fitBounds(layer.getBounds(), { padding: [8, 8] });
      map._anisightFitted = true;
    }
  }

  function setSelected(unit) {
    selected = unit ?? null;
    if (!layer) return;
    layer.eachLayer((lyr) => {
      if (lyr.feature) {
        lyr.setStyle({
          color: lyr.feature.properties.unit === selected ? '#000000' : '#333333',
          weight: lyr.feature.properties.unit === selected ? 3 : 1,
        });
        if (lyr.feature.properties.unit === selected) lyr.bringToFront();
      }
    });
  }

  return { map, render, setSelected };
}

// The legend sits below the map (plain HTML), so it never covers areas on a phone.
function addLegend(div) {
  if (!div) return;
  const ticks = legendTicks(PCT_CLAMP);
  div.innerHTML = `
      <p class="legend-title" id="legend-title">B3 forecast vs usual yield</p>
      <div class="legend-bar" style="background:${legendGradient()}" aria-hidden="true"></div>
      <div class="legend-ticks" aria-hidden="true">${ticks.map((t) => `<span>${t.label}</span>`).join('')}</div>
      <div class="legend-ends" aria-hidden="true"><span>below usual</span><span>above usual</span></div>
      <div class="legend-nodata"><span class="swatch" style="background:${NO_DATA_COLOR}"></span>
        n/a: no historical normal yet (dashed outline)</div>
      <p class="sr-only">Colour scale from red (${ticks[0].label}, below usual) through pale yellow
        (0%) to blue (${ticks[ticks.length - 1].label}, above usual). Each area is also labelled with its value.</p>`;
}

// --- small geometry helpers (Leaflet layer -> rings in lat/lng) ---------------

function rings(lyr) {
  // getLatLngs on a MultiPolygon: [[[ring], [hole]...], ...]; on a Polygon: [[ring], ...]
  const ll = lyr.getLatLngs();
  const polys = Array.isArray(ll[0]?.[0]) ? ll : [ll];
  return polys.map((p) => p[0]);
}

function inRing(pt, ring) {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const a = ring[i];
    const b = ring[j];
    if ((a.lat > pt.lat) !== (b.lat > pt.lat)
      && pt.lng < ((b.lng - a.lng) * (pt.lat - a.lat)) / (b.lat - a.lat) + a.lng) inside = !inside;
  }
  return inside;
}

function containsPoint(lyr, pt) {
  return rings(lyr).some((r) => inRing(pt, r));
}

/** Rough interior point: centre of the largest ring's bounds, else its first vertex. */
function interiorPoint(lyr) {
  const rs = rings(lyr).sort((a, b) => b.length - a.length);
  const r = rs[0];
  const c = L.latLngBounds(r).getCenter();
  return inRing(c, r) ? c : r[Math.floor(r.length / 2)];
}
