import numpy as np
import rioxarray  # noqa: F401
import xarray as xr

from cropyield.satellite import sentinel2 as s2

RES = 1 / 12000  # ~10 m in degrees


def _mask():
    y = 7.0 + RES * np.arange(120)[::-1]
    x = 125.0 + RES * np.arange(125)
    m = xr.DataArray(np.zeros((120, 125), bool), dims=("y", "x"), coords={"y": y, "x": x})
    m[:60, :] = True        # top half cropland -> 10 x 20 coarse cells
    m[60:66, :3] = True     # one coarse cell exactly 50% cropland -> counts
    return m.rio.write_crs("EPSG:4326")


def test_coarsen_mask_block_share_and_grid():
    c = s2.coarsen_mask(_mask(), 6)
    assert c.shape == (20, 20)          # 125 columns: trailing partial block dropped
    assert c.dtype == bool
    assert str(c.rio.crs) == "EPSG:4326"
    assert np.allclose(np.abs(c.rio.resolution()), 6 * RES)
    assert int(c.sum()) == 10 * 20 + 1


def test_coarsen_mask_factor_one_is_identity():
    m = _mask()
    assert s2.coarsen_mask(m, 1) is m


class _Stop(Exception):
    pass


def _capture(seen):
    def fake_extract(*args, **kwargs):
        seen.update(kwargs)
        raise _Stop  # stop before anything is written
    return fake_extract


def _run_main(monkeypatch, argv):
    seen = {}
    monkeypatch.setattr(s2, "extract_s2", _capture(seen))
    monkeypatch.setattr(s2, "load_masks", lambda gdf, cfg: {})
    monkeypatch.setattr(s2.gpd, "read_parquet", lambda path: None)
    try:
        s2.main(argv)
    except _Stop:
        pass
    return seen


def test_main_defaults_to_coarse_grid(monkeypatch):
    seen = _run_main(monkeypatch, ["--start", "2024-01-01", "--end", "2024-01-31"])
    assert seen["coarsen"] == s2.DEFAULT_COARSEN == 6


def test_main_coarsen_option(monkeypatch):
    seen = _run_main(monkeypatch, ["--start", "2024-01-01", "--end", "2024-01-31",
                                   "--coarsen", "1"])
    assert seen["coarsen"] == 1
