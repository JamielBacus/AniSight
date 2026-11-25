"""Tests for cropyield.boundaries with synthetic polygons and rasters (no network)."""
from __future__ import annotations

import dataclasses
import shutil

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
from shapely.geometry import box

from cropyield import boundaries as b
from cropyield.config import DEFAULT_CONFIG_DIR, load_config

PROVINCES = ["Davao de Oro", "Davao del Norte", "Davao del Sur", "Davao Oriental",
             "Davao Occidental", "Davao City"]


@pytest.fixture
def cfg(tmp_path):
    base = load_config()
    paths = dataclasses.replace(base.pipeline.paths, raw=tmp_path / "raw",
                                interim=tmp_path / "interim")
    return dataclasses.replace(base, pipeline=dataclasses.replace(base.pipeline, paths=paths))


def _adm2(names: list[str]) -> gpd.GeoDataFrame:
    """One 0.1-degree square per name, laid out along longitude near 7 N."""
    geoms = [box(125 + i * 0.1, 7.0, 125.1 + i * 0.1, 7.1) for i in range(len(names))]
    return gpd.GeoDataFrame({"shapeName": names}, geometry=geoms, crs="EPSG:4326")


GB_NAMES = ["Compostela Valley", "Davao Del Norte", "Davao del Sur", "Davao Oriental",
            "Davao Occidental", "Cotabato"]
SUR = box(125.2, 7.0, 125.3, 7.1)  # Davao del Sur square from _adm2(GB_NAMES)
CITY = box(125.2, 7.0, 125.25, 7.05)  # synthetic City of Davao, inside SUR


def _adm3() -> gpd.GeoDataFrame:
    """Synthetic ADM3 with the city inside Davao del Sur plus an unrelated municipality."""
    return gpd.GeoDataFrame({"shapeName": ["City of Davao", "Digos City"]},
                            geometry=[CITY, box(125.25, 7.05, 125.3, 7.1)], crs="EPSG:4326")


def test_select_provinces_matches_names_and_aliases(cfg):
    out = b.select_provinces(_adm2(GB_NAMES), cfg.provinces, adm3=_adm3)
    assert list(out["province"]) == PROVINCES
    assert out.crs.to_string() == "EPSG:4326"
    assert out.loc[out.province == "Davao de Oro", "source_name"].item() == "Compostela Valley"


def test_select_provinces_names_unmatched_province(cfg):
    raw = _adm2([n for n in GB_NAMES if n != "Davao Oriental"])
    with pytest.raises(ValueError, match="Davao Oriental"):
        b.select_provinces(raw, cfg.provinces, adm3=_adm3)


def _area(out, name):
    return out.loc[out.province == name].geometry.item().area


def test_davao_city_from_adm2_is_kept_and_subtracted(cfg):
    # ADM2 has a separate city polygon that overlaps Davao del Sur; ADM3 must not be read.
    raw = pd.concat([_adm2(GB_NAMES), gpd.GeoDataFrame(
        {"shapeName": ["Davao City"]}, geometry=[CITY], crs="EPSG:4326")], ignore_index=True)
    def no_adm3():
        raise AssertionError("ADM3 should not be loaded when ADM2 has the city")
    out = b.select_provinces(raw, cfg.provinces, adm3=no_adm3)
    assert list(out["province"]) == PROVINCES
    assert out.loc[out.province == "Davao City", "source_name"].item() == "Davao City"
    sur = out.loc[out.province == "Davao del Sur"].geometry.item()
    city = out.loc[out.province == "Davao City"].geometry.item()
    assert sur.intersection(city).area == pytest.approx(0, abs=1e-12)
    assert sur.area == pytest.approx(SUR.area - CITY.area)


def test_davao_city_from_adm3_when_not_in_adm2(cfg):
    out = b.select_provinces(_adm2(GB_NAMES), cfg.provinces, adm3=_adm3)
    assert list(out["province"]) == PROVINCES
    assert out.loc[out.province == "Davao City", "source_name"].item() == "City of Davao (ADM3)"
    sur = out.loc[out.province == "Davao del Sur"].geometry.item()
    city = out.loc[out.province == "Davao City"].geometry.item()
    assert city.equals(CITY)
    assert sur.intersection(city).area == pytest.approx(0, abs=1e-12)
    assert sur.area == pytest.approx(SUR.area - CITY.area)
    assert sur.union(city).equals(SUR)  # nothing lost between the two units
    assert _area(out, "Davao Oriental") == pytest.approx(0.01)  # other provinces untouched


def test_davao_city_missing_everywhere_raises(cfg):
    with pytest.raises(ValueError, match="Davao City"):
        b.select_provinces(_adm2(GB_NAMES), cfg.provinces)
    empty = lambda: _adm3().iloc[1:]
    with pytest.raises(ValueError, match="Davao City"):
        b.select_provinces(_adm2(GB_NAMES), cfg.provinces, adm3=empty)


def test_part_of_must_name_a_configured_province(cfg):
    provs = tuple(dataclasses.replace(p, part_of="Cotabato") if p.part_of else p
                  for p in cfg.provinces.provinces)
    prov = dataclasses.replace(cfg.provinces, provinces=provs)
    with pytest.raises(ValueError, match="Cotabato"):
        b.select_provinces(_adm2(GB_NAMES), prov, adm3=_adm3)


def _fake_requests(monkeypatch, names=GB_NAMES) -> list[str]:
    """Patch requests.get to serve synthetic geoBoundaries ADM2/ADM3 API + GeoJSON;
    returns the requested URLs."""
    geojson = {"ADM2": _adm2(names).to_json(), "ADM3": _adm3().to_json()}
    calls = []

    class Resp:
        status_code = 200

        def __init__(self, url):
            self.url = url
            self.level = "ADM3" if "ADM3" in url.upper() else "ADM2"
            self.content = geojson[self.level].encode()

        def json(self):
            return {"gjDownloadURL": f"https://example.test/phl_{self.level.lower()}.geojson"}

        def raise_for_status(self):
            pass

    def fake_get(url, **kw):
        calls.append(url)
        return Resp(url)

    monkeypatch.setattr(b.requests, "get", fake_get)
    return calls


def test_load_boundaries_downloads_once_and_saves(cfg, monkeypatch):
    calls = _fake_requests(monkeypatch)
    gdf = b.load_boundaries(cfg)
    assert len(calls) == 4  # ADM2 + ADM3 (city not in ADM2), metadata + GeoJSON each
    assert calls[2] == cfg.pipeline.sources.boundaries_adm3_url
    assert (cfg.pipeline.paths.raw / "boundaries" / "geoBoundaries-PHL-ADM3.geojson").exists()
    b.load_boundaries(cfg)
    assert len(calls) == 4  # cached on re-run

    out = b.save_boundaries(gdf, cfg)
    assert out == cfg.pipeline.paths.interim / "boundaries.parquet"
    back = gpd.read_parquet(out)
    assert list(back["province"]) == PROVINCES
    assert back.crs.to_string() == "EPSG:4326"


def _landcover(bounds, res=0.001, cls_left=40, cls_right=10) -> xr.DataArray:
    """Synthetic WorldCover tile: left half cropland (40), right half tree cover (10)."""
    minx, miny, maxx, maxy = bounds
    x = np.arange(minx + res / 2, maxx, res)
    y = np.arange(maxy - res / 2, miny, -res)
    data = np.full((y.size, x.size), cls_right, dtype="uint8")
    data[:, : x.size // 2] = cls_left
    da = xr.DataArray(data, coords={"y": y, "x": x}, dims=("y", "x"))
    return da.rio.write_crs("EPSG:4326")


def test_mask_from_landcover_counts_cropland_hectares():
    geom = box(125.0, 7.0, 125.1, 7.1)
    lc = _landcover((124.95, 6.95, 125.15, 7.15))  # bigger than the polygon
    mask = b.mask_from_landcover(lc, geom, 40)
    assert mask.dtype == bool
    assert mask.rio.crs.to_string() == "EPSG:4326"
    # Cropland is x < 125.05, so 0.05 x 0.1 degree of the polygon (~5.5 km x 11 km).
    assert int(mask.sum()) == 50 * 100
    expected_ha = (b.EARTH_RADIUS_M ** 2 * np.radians(0.05)
                   * (np.sin(np.radians(7.1)) - np.sin(np.radians(7.0))) / 1e4)
    assert mask.attrs["cropland_ha"] == pytest.approx(expected_ha, rel=1e-3)


def test_cropland_mask_uses_loader_and_saves(cfg):
    gdf = b.select_provinces(_adm2(GB_NAMES), cfg.provinces, adm3=_adm3)
    seen = []

    def loader(bbox):
        seen.append(bbox)
        return _landcover((bbox[0] - 0.01, bbox[1] - 0.01, bbox[2] + 0.01, bbox[3] + 0.01))

    masks = b.cropland_mask(gdf, cfg, loader=loader)
    assert list(masks) == PROVINCES
    assert len(seen) == 6
    table = b.cropland_area_table(masks)
    assert list(table.columns) == ["province", "cropland_ha"]
    assert (table["cropland_ha"] > 0).all()

    out = b.save_cropland(masks, cfg)
    assert pd.read_parquet(out)["province"].tolist() == PROVINCES
    tif = cfg.pipeline.paths.interim / "cropland" / "davao_de_oro.tif"
    back = rioxarray.open_rasterio(tif).squeeze()
    assert int(back.sum()) == int(masks["Davao de Oro"].sum())


def test_stac_worldcover_loader_uses_latest_year(cfg, monkeypatch):
    import odc.stac
    import planetary_computer
    import pystac
    import pystac_client

    def item(id_, year):
        return pystac.Item(id_, None, None, None,
                           {"start_datetime": f"{year}-01-01T00:00:00Z",
                            "end_datetime": f"{year}-12-31T23:59:59Z"})

    items = [item("ESA_WorldCover_10m_2020_v100_N06E123", 2020),
             item("ESA_WorldCover_10m_2021_v200_N06E123", 2021)]
    seen = {}

    class Catalog:
        def search(self, **kw):
            seen["search"] = kw

            class Search:
                def items(self):
                    return iter(items)
            return Search()

    def fake_open(url, modifier=None):
        seen.update(url=url, modifier=modifier)
        return Catalog()

    bbox = (125.0, 7.0, 125.01, 7.01)

    def fake_load(its, **kw):
        seen.update(items=its, **kw)
        lc = _landcover(bbox)
        dims = ("time", "latitude", "longitude")
        return xr.Dataset({"map": (dims, lc.values[None])},
                          coords={"time": [0], "latitude": lc["y"].values,
                                  "longitude": lc["x"].values})

    monkeypatch.setattr(pystac_client.Client, "open", fake_open)
    monkeypatch.setattr(odc.stac, "load", fake_load)
    da = b.stac_worldcover_loader(cfg)(bbox)
    assert seen["url"] == cfg.pipeline.sources.stac_url
    assert seen["modifier"] is planetary_computer.sign_inplace
    assert seen["search"] == {"collections": ["esa-worldcover"], "bbox": bbox}
    assert [it.id for it in seen["items"]] == ["ESA_WorldCover_10m_2021_v200_N06E123"]
    assert seen["bands"] == ["map"] and seen["resolution"] == b.WORLDCOVER_RES_DEG
    assert da.dims == ("y", "x")
    assert da.rio.crs.to_string() == "EPSG:4326"
    assert int((da == 40).sum()) == 10 * 5


def test_stac_worldcover_loader_raises_without_items(cfg, monkeypatch):
    import pystac_client

    class Catalog:
        def search(self, **kw):
            class Search:
                def items(self):
                    return iter([])
            return Search()

    monkeypatch.setattr(pystac_client.Client, "open", lambda url, modifier=None: Catalog())
    with pytest.raises(RuntimeError, match="esa-worldcover"):
        b.stac_worldcover_loader(cfg)((125.0, 7.0, 125.01, 7.01))


def test_main_writes_boundaries_and_cropland(tmp_path, monkeypatch, capsys):
    conf = tmp_path / "configs"
    shutil.copytree(DEFAULT_CONFIG_DIR, conf)
    cfg = load_config(conf)
    calls = _fake_requests(monkeypatch)
    pad = lambda bb: (bb[0] - 0.01, bb[1] - 0.01, bb[2] + 0.01, bb[3] + 0.01)
    monkeypatch.setattr(b, "stac_worldcover_loader",
                        lambda c: (lambda bb: _landcover(pad(bb))))

    b.main(["--config-dir", str(conf)])
    interim = cfg.pipeline.paths.interim
    assert interim == tmp_path / "data" / "interim"
    assert list(gpd.read_parquet(interim / "boundaries.parquet")["province"]) == PROVINCES
    assert pd.read_parquet(interim / "cropland_area.parquet")["province"].tolist() == PROVINCES
    assert len(list((interim / "cropland").glob("*.tif"))) == 6
    assert "(6 provinces)" in capsys.readouterr().out
    assert len(calls) == 4

    # --step cropland reuses boundaries.parquet and does not download boundaries again.
    (interim / "cropland_area.parquet").unlink()
    b.main(["--config-dir", str(conf), "--step", "cropland"])
    assert (interim / "cropland_area.parquet").exists()
    assert len(calls) == 4


@pytest.mark.network
def test_worldcover_stac_tiny_bbox(cfg):
    """Smoke test against the Planetary Computer on a ~1 km box near Davao City."""
    loader = b.stac_worldcover_loader(cfg)
    lc = loader((125.40, 7.10, 125.41, 7.11))
    mask = b.mask_from_landcover(lc, box(125.40, 7.10, 125.41, 7.11), 40)
    assert mask.size > 0


def test_reporting_unit_geometries_union_host_and_absorbed(cfg):
    gdf = b.select_provinces(_adm2(GB_NAMES), cfg.provinces, adm3=_adm3)
    out = b.reporting_unit_geometries(gdf, cfg)
    assert out.crs.to_string() == "EPSG:4326"
    assert list(out["reporting_unit"]) == ["Davao del Norte+Davao de Oro",
                                           "Davao del Sur+Davao Occidental",
                                           "Davao del Sur+Davao City"]
    geom = dict(zip(gdf["province"], gdf.geometry))
    for name, members, g in zip(out["reporting_unit"], out["members"], out.geometry):
        assert name == "+".join(members)
        expected = gpd.GeoSeries([geom[m] for m in members]).union_all()
        assert g.symmetric_difference(expected).area < 1e-12
    # Davao del Sur with the city added back is the original ADM2 square.
    city_unit = out.set_index("reporting_unit").geometry["Davao del Sur+Davao City"]
    assert city_unit.symmetric_difference(SUR).area < 1e-12


def test_reporting_unit_geometries_needs_member_polygons(cfg):
    gdf = b.select_provinces(_adm2(GB_NAMES), cfg.provinces, adm3=_adm3)
    with pytest.raises(ValueError, match="Davao Occidental"):
        b.reporting_unit_geometries(gdf[gdf.province != "Davao Occidental"], cfg)
