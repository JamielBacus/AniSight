"""Tests for cropyield.satellite.sentinel2 with synthetic rasters and mocked STAC (no network)."""
from __future__ import annotations

import dataclasses
import logging
import shutil
from datetime import date, datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pystac
import pytest
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
from shapely.geometry import box

from cropyield.config import DEFAULT_CONFIG_DIR, load_config
from cropyield.satellite import sentinel2 as s2

PROV = "Davao del Sur"
NY, NX = 4, 5
RES = 1 / 12000
TODAY = date(2024, 6, 15)


@pytest.fixture
def cfg(tmp_path):
    base = load_config()
    paths = dataclasses.replace(base.pipeline.paths, raw=tmp_path / "raw",
                                interim=tmp_path / "interim")
    return dataclasses.replace(base, pipeline=dataclasses.replace(base.pipeline, paths=paths))


@pytest.fixture
def gdf():
    return gpd.GeoDataFrame({"province": [PROV]},
                            geometry=[box(125.0, 7.0, 125.0 + NX * RES, 7.0 + NY * RES)],
                            crs="EPSG:4326")


def _grid(values) -> xr.DataArray:
    x = 125.0 + RES * (np.arange(NX) + 0.5)
    y = 7.0 + RES * (NY - np.arange(NY) - 0.5)
    return xr.DataArray(np.asarray(values), coords={"y": y, "x": x},
                        dims=("y", "x")).rio.write_crs("EPSG:4326")


@pytest.fixture
def mask():
    m = np.ones((NY, NX), dtype=bool)
    m[0, :] = False  # top row is not cropland -> 15 cropland pixels
    return {PROV: _grid(m)}


def _scenes(template: xr.DataArray, red, nir, blue, scl, times) -> xr.Dataset:
    """Dataset in the loader contract; each band arg is a list of per-time scalars/arrays."""
    def stack(vals, dtype):
        return (("time", "y", "x"),
                np.stack([np.broadcast_to(v, (NY, NX)) for v in vals]).astype(dtype))
    return xr.Dataset(
        {"red": stack(red, "float32"), "nir": stack(nir, "float32"),
         "blue": stack(blue, "float32"), "scl": stack(scl, "uint8")},
        coords={"time": pd.to_datetime(times), "y": template["y"], "x": template["x"]})


def _item(dt: str, baseline: str | None = "05.00") -> pystac.Item:
    props = {} if baseline is None else {"s2:processing_baseline": baseline}
    return pystac.Item(id=f"S2_{dt}", geometry=None, bbox=None,
                       datetime=datetime.fromisoformat(dt).replace(tzinfo=timezone.utc),
                       properties=props)


class FakeSearch:
    def __init__(self, items=None, fail_months=()):
        self.calls, self.items, self.fail_months = [], items, set(fail_months)

    def __call__(self, collection, bbox, start, end, query=None):
        self.calls.append((collection, bbox, start, end, query))
        if start[:7] in self.fail_months:
            raise ConnectionError("STAC down")
        return [_item(start)] if self.items is None else self.items


class FakeLoader:
    """Two clear dates and one cloudy date with an extreme value that must be ignored."""
    def __init__(self, scl_override=None):
        self.calls = 0
        self.scl_override = scl_override

    def __call__(self, items, template):
        self.calls += 1
        t = pd.Timestamp(items[0].datetime.date())
        scl = self.scl_override if self.scl_override is not None else [4, 4, 9]
        return _scenes(template, red=[0.1, 0.1, 0.0], nir=[0.5, 0.3, 0.9],
                       blue=[0.05, 0.05, 0.0], scl=scl,
                       times=[t, t + pd.Timedelta(days=5), t + pd.Timedelta(days=10)])


# ------------------------------------------------------------------------- unit tests
def test_vegetation_indices_formulas(mask):
    ds = _scenes(mask[PROV], red=[0.1], nir=[0.5], blue=[0.05], scl=[4], times=["2024-01-01"])
    vi = s2.vegetation_indices(ds)
    assert float(vi["NDVI"][0, 0, 0]) == pytest.approx(0.4 / 0.6)
    assert float(vi["EVI"][0, 0, 0]) == pytest.approx(2.5 * 0.4 / (0.5 + 0.6 - 0.375 + 1))


def test_evi_out_of_range_is_nan(mask):
    # denominator ~0 -> huge EVI -> NaN
    ds = _scenes(mask[PROV], red=[0.0], nir=[0.2], blue=[0.16], scl=[4], times=["2024-01-01"])
    assert np.isnan(s2.vegetation_indices(ds)["EVI"]).all()


def test_clear_sky_masks_cloud_classes_and_nodata():
    scl = xr.DataArray(np.arange(12, dtype="uint8"))
    clear = s2.clear_sky(scl, (3, 8, 9, 10)).values
    assert list(np.flatnonzero(~clear)) == [0, 3, 8, 9, 10]


def test_monthly_composite_is_median_of_clear_observations(mask):
    scl = np.full((NY, NX), 4)
    scl[1, 1] = 8  # pixel (1,1) cloudy on the second date
    ds = _scenes(mask[PROV], red=[0.1, 0.1, 0.1], nir=[0.5, 0.3, 0.7],
                 blue=[0.05, 0.05, 0.05], scl=[4, scl, 9], times=["2024-01-01", "2024-01-06",
                                                                  "2024-01-11"])
    ndvi = s2.monthly_composite(ds, (3, 8, 9, 10))["NDVI"]
    assert "time" not in ndvi.dims
    np.testing.assert_allclose(float(ndvi[0, 0]), np.median([0.4 / 0.6, 0.2 / 0.4]), rtol=1e-6)
    np.testing.assert_allclose(float(ndvi[1, 1]), 0.4 / 0.6, rtol=1e-6)


def test_reflectance_offset_by_processing_baseline():
    assert s2.reflectance_offset(_item("2021-06-01", "02.09")) == 0
    assert s2.reflectance_offset(_item("2022-03-01", "04.00")) == -1000
    assert s2.reflectance_offset(_item("2023-03-01", "05.09")) == -1000
    assert s2.reflectance_offset(_item("2021-06-01", None)) == 0
    assert s2.reflectance_offset(_item("2022-06-01", None)) == -1000


def test_stac_loader_scales_offsets_and_renames(monkeypatch, mask):
    import odc.stac
    tmpl = mask[PROV]

    def fake_load(items, bands, geobox, **kw):
        assert geobox.shape == tmpl.shape
        dn = 2000 if items[0].properties["s2:processing_baseline"] < "04" else 3000
        data = np.full((1, NY, NX), dn, dtype="uint16")
        data[0, 0, 0] = 0  # nodata
        coords = {"time": [pd.Timestamp(items[0].datetime.date())],
                  "latitude": tmpl["y"].values, "longitude": tmpl["x"].values}
        dims = ("time", "latitude", "longitude")
        ds = xr.Dataset({b: (dims, data) for b in ("B02", "B04", "B08")}, coords=coords)
        ds["SCL"] = (dims, np.full((1, NY, NX), 4, dtype="uint8"))
        return ds

    monkeypatch.setattr(odc.stac, "load", fake_load)
    ds = s2.stac_s2_loader()([_item("2022-03-01", "04.00"), _item("2021-03-01", "02.09")], tmpl)
    assert list(ds["time"].dt.year.values) == [2021, 2022]
    assert set(ds.dims) == {"time", "y", "x"}
    np.testing.assert_allclose(ds["red"].values[:, 1, 1], [0.2, 0.2])  # 2000/1e4, (3000-1000)/1e4
    assert np.isnan(ds["nir"].values[:, 0, 0]).all()
    assert (ds["scl"] == 4).all()


# ------------------------------------------------------------------- extract_s2 tests
def test_extract_s2_monthly_table(cfg, gdf, mask):
    search, loader = FakeSearch(), FakeLoader()
    df = s2.extract_s2(gdf, mask, "2024-01-10", "2024-02-20", cfg=cfg, search=search,
                       loader=loader, today=TODAY)
    assert list(df.columns) == s2.COLUMNS
    assert len(df) == 4
    assert list(df["date"].unique()) == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01")]
    assert set(df["index"]) == {"NDVI", "EVI"}
    ndvi = df[(df["index"] == "NDVI")].iloc[0]
    want = np.median([0.4 / 0.6, 0.2 / 0.4])  # cloudy third date is ignored
    assert ndvi["mean"] == pytest.approx(want, rel=1e-6)
    assert ndvi["median"] == pytest.approx(want, rel=1e-6)
    assert ndvi["max"] == pytest.approx(want, rel=1e-6)
    assert ndvi["valid_frac"] == 1.0
    assert ndvi["n_pixels"] == 15  # cropland pixels only
    coll, bbox, start, end, query = search.calls[0]
    assert coll == "sentinel-2-l2a"
    assert (start, end) == ("2024-01-01", "2024-01-31")
    assert bbox == pytest.approx(tuple(gdf.geometry.iloc[0].bounds))
    assert query == {"eo:cloud_cover": {"lt": s2.DEFAULT_MAX_SCENE_CLOUD}}


def test_low_coverage_month_marked_missing(cfg, gdf, mask):
    scl = np.full((NY, NX), 9)
    scl[3, 4] = 4  # one clear cropland pixel of 15 -> 6.7% < 10%
    loader = FakeLoader(scl_override=[scl, 9, 9])
    df = s2.extract_s2(gdf, mask, "2024-01-01", "2024-01-31", cfg=cfg, search=FakeSearch(),
                       loader=loader, today=TODAY)
    assert df[["mean", "median", "max"]].isna().all().all()
    assert df["n_pixels"].tolist() == [1, 1]
    assert df["valid_frac"].tolist() == pytest.approx([1 / 15, 1 / 15])


def test_coverage_at_threshold_is_reported(cfg, gdf, mask):
    scl = np.full((NY, NX), 9)
    scl[2, :2] = 4  # 2 of 15 = 13% >= 10%
    df = s2.extract_s2(gdf, mask, "2024-01-01", "2024-01-31", cfg=cfg, search=FakeSearch(),
                       loader=FakeLoader(scl_override=[scl, 9, 9]), today=TODAY)
    assert df["mean"].notna().all()


def test_month_without_scenes_is_missing(cfg, gdf, mask):
    loader = FakeLoader()
    df = s2.extract_s2(gdf, mask, "2024-01-01", "2024-01-31", cfg=cfg,
                       search=FakeSearch(items=[]), loader=loader, today=TODAY)
    assert loader.calls == 0
    assert len(df) == 2 and df["mean"].isna().all()
    assert (df["valid_frac"] == 0).all() and (df["n_pixels"] == 0).all()


def test_rerun_uses_cache_and_skips_future_months(cfg, gdf, mask):
    loader = FakeLoader()
    kw = dict(cfg=cfg, search=FakeSearch(), loader=loader, today=TODAY)
    first = s2.extract_s2(gdf, mask, "2024-04-01", "2024-08-31", **kw)
    assert sorted(first["date"].dt.month.unique()) == [4, 5, 6]  # Jul/Aug are in the future
    cache = cfg.pipeline.paths.interim / "cache" / "s2"
    assert sorted(p.name for p in cache.glob("*.parquet")) == [
        "davao_del_sur_2024-04.parquet", "davao_del_sur_2024-05.parquet"]  # June incomplete
    assert loader.calls == 3
    second = s2.extract_s2(gdf, mask, "2024-04-01", "2024-08-31", **kw)
    assert loader.calls == 4  # only the uncached current month is reloaded
    pd.testing.assert_frame_equal(first, second)


def test_failed_chunk_is_logged_and_skipped(cfg, gdf, mask, monkeypatch, caplog):
    monkeypatch.setattr(s2, "RETRY_DELAY_S", 0)
    search = FakeSearch(fail_months={"2024-02"})
    with caplog.at_level(logging.WARNING):
        df = s2.extract_s2(gdf, mask, "2024-01-01", "2024-03-31", cfg=cfg, search=search,
                           loader=FakeLoader(), today=TODAY)
    assert sorted(df["date"].dt.month.unique()) == [1, 3]
    assert sum(c[2] == "2024-02-01" for c in search.calls) == s2.RETRIES
    assert "2024-02 failed after" in caplog.text
    assert not (cfg.pipeline.paths.interim / "cache" / "s2" /
                "davao_del_sur_2024-02.parquet").exists()


def test_missing_mask_raises(cfg, gdf):
    with pytest.raises(ValueError, match=PROV):
        s2.extract_s2(gdf, {}, "2024-01-01", "2024-01-31", cfg=cfg, search=FakeSearch(),
                      loader=FakeLoader(), today=TODAY)


def test_main_writes_parquet(tmp_path, gdf, mask, monkeypatch, capsys):
    cfg_dir = tmp_path / "configs"
    shutil.copytree(DEFAULT_CONFIG_DIR, cfg_dir)
    interim = tmp_path / "data" / "interim"
    (interim / "cropland").mkdir(parents=True)
    gdf.to_parquet(interim / "boundaries.parquet")
    mask[PROV].astype("uint8").rio.to_raster(interim / "cropland" / "davao_del_sur.tif")
    monkeypatch.setattr(s2, "search_items", FakeSearch())
    monkeypatch.setattr(s2, "stac_s2_loader", lambda: FakeLoader())
    # Full resolution: the fixture mask is only a few pixels wide, so the default 6x
    # coarse grid would trim it away. The coarse default is covered in test_s2_coarsen.py.
    s2.main(["--config-dir", str(cfg_dir), "--start", "2023-01-01", "--end", "2023-02-28",
             "--coarsen", "1"])
    out = pd.read_parquet(interim / "s2_vi.parquet")
    assert list(out.columns) == s2.COLUMNS
    assert len(out) == 4 and (out["n_pixels"] == 15).all()
    assert "wrote" in capsys.readouterr().out


# ---------------------------------------------------------------------- network smoke
@pytest.mark.network
def test_network_smoke_tiny_bbox(cfg):
    """One month over a ~200 m square near Digos; hits the Planetary Computer."""
    x0, y0 = 125.30, 6.80
    small = gpd.GeoDataFrame({"province": [PROV]},
                             geometry=[box(x0, y0, x0 + 24 * RES, y0 + 24 * RES)],
                             crs="EPSG:4326")
    x = x0 + RES * (np.arange(24) + 0.5)
    y = y0 + RES * (24 - np.arange(24) - 0.5)
    m = xr.DataArray(np.ones((24, 24), bool), coords={"y": y, "x": x},
                     dims=("y", "x")).rio.write_crs("EPSG:4326")
    df = s2.extract_s2(small, {PROV: m}, "2023-03-01", "2023-03-31", cfg=cfg)
    assert list(df.columns) == s2.COLUMNS and len(df) == 2
