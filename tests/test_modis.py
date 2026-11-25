"""Tests for cropyield.satellite.modis with synthetic rasters and mocked STAC (no network)."""
from __future__ import annotations

import dataclasses
import logging
import shutil

import numpy as np
import pandas as pd
import pystac
import pytest
import rioxarray  # noqa: F401
import xarray as xr
from shapely.geometry import box

from cropyield import boundaries as b
from cropyield.config import DEFAULT_CONFIG_DIR, load_config
from cropyield.satellite import modis as m
from cropyield.satellite import stac
from tests.test_boundaries import GB_NAMES, _adm2, _adm3, _landcover

RES = 0.001  # synthetic mask pixel; cropland_fraction groups 2 x 2 -> 0.002 degree cells


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(stac.time, "sleep", lambda s: None)


@pytest.fixture
def cfg(tmp_path):
    base = load_config()
    paths = dataclasses.replace(base.pipeline.paths, raw=tmp_path / "raw",
                                interim=tmp_path / "interim")
    return dataclasses.replace(base, pipeline=dataclasses.replace(base.pipeline, paths=paths))


def _mask(x0=125.0, y1=7.02, n=20) -> xr.DataArray:
    """n x n boolean mask at RES; the left half is cropland."""
    x = x0 + RES / 2 + RES * np.arange(n)
    y = y1 - RES / 2 - RES * np.arange(n)
    data = np.zeros((n, n), bool)
    data[:, : n // 2] = True
    return xr.DataArray(data, coords={"y": y, "x": x}, dims=("y", "x")).rio.write_crs("EPSG:4326")


def test_cropland_fraction_block_average():
    mk = _mask()
    mk[0, 0] = False  # one block becomes 3/4 cropland
    frac = m.cropland_fraction(mk)
    assert frac.shape == (10, 10)
    assert frac["x"].values[0] == pytest.approx(125.001)
    assert frac["y"].values[0] == pytest.approx(7.019)
    assert frac.values[0, 0] == pytest.approx(0.75)
    assert (frac.values[:, 1:5] == 1).all() and (frac.values[:, 5:] == 0).all()
    # Odd sizes are padded with non-cropland.
    assert m.cropland_fraction(_mask(n=21)).shape == (11, 11)


def _fake_loader(calls=None, fail_dates=()):
    """Loader returning two tiles per composite on the requested grid.

    Tile 0: NDVI = 2000 + 100 * column, EVI = 3000 (fill -3000 at row 1, column 0),
    reliability 3 (cloudy) on row 0, else 0. Tile 1 is all fill, so merging must keep
    tile 0. Columns/rows count from the first cell of the cropland grid.
    """
    def load(items, bbox, res):
        d = stac.item_end_date(items[0])
        if calls is not None:
            calls.append(d)
        if d in fail_dates:
            raise ConnectionError("simulated outage")
        x = np.arange(bbox[0], bbox[2] + res / 2, res)
        y = np.arange(bbox[3], bbox[1] - res / 2, -res)
        col = np.round((x - (bbox[0] + res)) / res).astype(int)[None, :]
        row = np.round(((bbox[3] - res) - y) / res).astype(int)[:, None]
        shape = (y.size, x.size)
        ndvi = np.broadcast_to(2000 + 100 * col, shape).astype("int16")
        evi = np.full(shape, 3000, "int16")
        evi[(row == 1) & (col == 0)] = -3000
        rel = np.where(np.broadcast_to(row == 0, shape), 3, 0).astype("int16")
        fill = lambda v: np.full(shape, v, "int16")
        dims = ("time", "y", "x")
        return xr.Dataset(
            {m.NDVI: (dims, np.stack([ndvi, fill(-3000)])),
             m.EVI: (dims, np.stack([evi, fill(-3000)])),
             m.RELIABILITY: (dims, np.stack([rel, fill(-1)]))},
            coords={"time": [0, 1], "y": y, "x": x})
    return load


def _items():
    def it(id_, start, end):
        return pystac.Item(id_, None, None, None,
                           {"start_datetime": f"{start}T00:00:00Z",
                            "end_datetime": f"{end}T23:59:59Z", "platform": ""})
    return [it("MOD13Q1.A2019353.h30v08.061", "2019-12-19", "2019-12-31"),  # before start
            it("MOD13Q1.A2020001.h30v08.061", "2020-01-01", "2020-01-16"),
            it("MYD13Q1.A2020009.h30v08.061", "2020-01-09", "2020-01-24"),  # Aqua: dropped
            it("MOD13Q1.A2020017.h30v08.061", "2020-01-17", "2020-02-01"),
            it("MOD13Q1.A2020033.h30v08.061", "2020-02-02", "2020-02-17")]  # after end


@pytest.fixture
def fake_stac(monkeypatch):
    searches = []

    def search(collection, bbox, start, end, query=None, *, catalog=None):
        searches.append((collection, start, end))
        return _items()

    monkeypatch.setattr(m, "open_catalog", lambda url=None: "catalog")
    monkeypatch.setattr(m, "search_items", search)
    return searches


def _gdf():
    import geopandas as gpd
    return gpd.GeoDataFrame({"province": ["Davao de Oro"]},
                            geometry=[box(125.0, 7.0, 125.02, 7.02)], crs="EPSG:4326")


def test_extract_modis_masks_scales_and_aggregates(cfg, fake_stac):
    df = m.extract_modis(_gdf(), {"Davao de Oro": _mask()}, "2020-01-01", "2020-02-01",
                         cfg=cfg, loader=_fake_loader())
    assert list(df.columns) == m.COLUMNS
    assert fake_stac == [("modis-13Q1-061", "2020-01-01", "2020-02-01")]
    # Rows are labelled with the composite end date.
    assert sorted(df["date"].unique()) == [pd.Timestamp("2020-01-16"), pd.Timestamp("2020-02-01")]
    assert len(df) == 4  # 2 composites x 2 indices
    ndvi = df[(df["index"] == "NDVI") & (df["date"] == "2020-01-16")].iloc[0]
    # 50 cropland cells (columns 0-4); row 0 is cloudy -> 45 valid.
    assert ndvi["n_pixels"] == 45
    assert ndvi["valid_frac"] == pytest.approx(0.9)
    assert ndvi["mean"] == pytest.approx(0.22)
    assert ndvi["median"] == pytest.approx(0.22)
    assert ndvi["max"] == pytest.approx(0.24)
    evi = df[(df["index"] == "EVI") & (df["date"] == "2020-01-16")].iloc[0]
    assert evi["n_pixels"] == 44  # plus one fill value
    assert evi["mean"] == pytest.approx(0.3)
    assert df[["mean", "median", "max"]].abs().le(1).all().all()


def test_extract_modis_resumes_and_skips_failed_chunks(cfg, fake_stac, caplog):
    calls = []
    masks = {"Davao de Oro": _mask()}
    with caplog.at_level(logging.ERROR):
        df = m.extract_modis(_gdf(), masks, "2020-01-01", "2020-02-01", cfg=cfg,
                             loader=_fake_loader(calls, {pd.Timestamp("2020-02-01")}))
    assert df["date"].unique().tolist() == [pd.Timestamp("2020-01-16")]
    assert "chunk failed" in caplog.text
    assert calls.count(pd.Timestamp("2020-02-01")) == 3  # retried before giving up
    cached = list((cfg.pipeline.paths.interim / "cache" / "modis").rglob("*.parquet"))
    assert [p.name for p in cached] == ["2020_01_16.parquet"]

    calls.clear()
    df2 = m.extract_modis(_gdf(), masks, "2020-01-01", "2020-02-01", cfg=cfg,
                          loader=_fake_loader(calls))
    assert calls == [pd.Timestamp("2020-02-01")]  # cached chunk not reloaded
    assert len(df2) == 4


def test_extract_modis_excludes_composites_ending_after_end(cfg, fake_stac):
    # The composite starting 2020-01-17 runs to 2020-02-01, past the 2020-01-31 cutoff.
    df = m.extract_modis(_gdf(), {"Davao de Oro": _mask()}, "2020-01-01", "2020-01-31",
                         cfg=cfg, loader=_fake_loader())
    assert df["date"].unique().tolist() == [pd.Timestamp("2020-01-16")]
    assert (df["date"] <= pd.Timestamp("2020-01-31")).all()


def test_stac_modis_loader_calls_odc_and_renames(monkeypatch):
    import odc.stac

    seen = {}

    def fake_load(items, **kw):
        seen.update(items=items, **kw)
        dims = ("time", "latitude", "longitude")
        coords = {"time": [0], "latitude": [7.01, 7.0], "longitude": [125.0, 125.01]}
        return xr.Dataset({b: (dims, np.zeros((1, 2, 2), "int16")) for b in m.BANDS},
                          coords=coords)

    monkeypatch.setattr(odc.stac, "load", fake_load)
    # The loader re-signs items right before reading (fresh SAS tokens); stub it offline.
    resigned = []

    def fake_fresh_sign(items):
        copies = [it.clone() for it in items]
        resigned.extend(copies)
        return copies

    monkeypatch.setattr(m, "fresh_sign", fake_fresh_sign)
    items = _items()[:1]
    bbox = (125.0, 7.0, 125.01, 7.01)
    ds = m.stac_modis_loader(items, bbox, m.MODIS_RES_DEG)
    assert set(ds.dims) == {"time", "y", "x"}
    assert set(ds.data_vars) == set(m.BANDS)
    assert seen["items"] == resigned  # the re-signed copies are what gets loaded
    assert [it.id for it in seen["items"]] == [it.id for it in items]
    assert seen["bands"] == list(m.BANDS)
    assert seen["bbox"] == bbox and seen["crs"] == "EPSG:4326"
    assert seen["resolution"] == m.MODIS_RES_DEG and seen["resampling"] == "nearest"
    assert seen["dtype"] == "int16"  # reliability band: int8 metadata with nodata 255


def test_extract_modis_requires_a_mask_per_province(cfg, fake_stac):
    with pytest.raises(ValueError, match="Davao de Oro"):
        m.extract_modis(_gdf(), {}, "2020-01-01", "2020-01-31", cfg=cfg,
                        loader=_fake_loader())


def test_composite_stats_no_valid_pixels():
    crop = xr.DataArray(np.ones((2, 2), bool), dims=("y", "x"))
    ds = xr.Dataset({k: (("y", "x"), np.full((2, 2), v, "int16"))
                     for k, v in [(m.NDVI, 5000), (m.EVI, 4000), (m.RELIABILITY, 3)]})
    rows = m.composite_stats(ds, crop)
    assert [r["n_pixels"] for r in rows] == [0, 0]
    assert rows[0]["valid_frac"] == 0 and np.isnan(rows[0]["mean"])


def test_main_reads_interim_outputs_and_writes_table(tmp_path, monkeypatch, fake_stac):
    conf = tmp_path / "configs"
    shutil.copytree(DEFAULT_CONFIG_DIR, conf)
    cfg = load_config(conf)
    gdf = b.select_provinces(_adm2(GB_NAMES), cfg.provinces, adm3=_adm3)
    b.save_boundaries(gdf, cfg)
    pad = lambda bb: (bb[0] - 0.01, bb[1] - 0.01, bb[2] + 0.01, bb[3] + 0.01)
    masks = b.cropland_mask(gdf, cfg, loader=lambda bb: _landcover(pad(bb)))
    b.save_cropland(masks, cfg)

    back = b.load_cropland(cfg, list(gdf["province"]))
    assert int(back["Davao de Oro"].sum()) == int(masks["Davao de Oro"].sum())
    assert back["Davao de Oro"].attrs["cropland_ha"] == pytest.approx(
        masks["Davao de Oro"].attrs["cropland_ha"])

    monkeypatch.setattr(m, "stac_modis_loader", _fake_loader())
    m.main(["--config-dir", str(conf), "--start", "2020-01-01", "--end", "2020-02-01"])
    out = pd.read_parquet(cfg.pipeline.paths.interim / "modis_vi.parquet")
    assert list(out.columns) == m.COLUMNS
    assert sorted(out["province"].unique()) == sorted(gdf["province"])
    assert len(out) == 6 * 2 * 2


@pytest.mark.network
def test_modis_stac_tiny_bbox(cfg):
    """Smoke test against the Planetary Computer: one composite on a ~2 km box."""
    mk = _mask(x0=125.40, y1=7.12, n=20)
    df = m.extract_modis(_gdf(), {"Davao de Oro": mk}, "2020-01-01", "2020-01-20", cfg=cfg)
    assert set(df["index"]) == {"NDVI", "EVI"}
    assert df["n_pixels"].max() > 0
