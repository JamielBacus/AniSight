"""Tests for cropyield.satellite.stac with mocked STAC clients (no network)."""
from __future__ import annotations

import datetime as dt

import pandas as pd
import planetary_computer
import pystac
import pytest

from cropyield.satellite import stac


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(stac.time, "sleep", lambda s: None)


def test_call_with_retry_retries_then_raises():
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("boom")
        return "ok"

    assert stac.call_with_retry(flaky, attempts=3) == "ok"
    assert len(calls) == 3
    with pytest.raises(ConnectionError):
        stac.call_with_retry(lambda: (_ for _ in ()).throw(ConnectionError("x")), attempts=2)


def test_open_catalog_signs_assets(monkeypatch):
    seen = {}

    def fake_open(url, modifier=None):
        seen.update(url=url, modifier=modifier)
        return "client"

    monkeypatch.setattr(stac.pystac_client.Client, "open", fake_open)
    assert stac.open_catalog() == "client"
    assert seen["url"] == "https://planetarycomputer.microsoft.com/api/stac/v1"
    assert seen["modifier"] is planetary_computer.sign_inplace


class FakeCatalog:
    def __init__(self, items, fail_first=0):
        self.items_, self.fail_first, self.calls = items, fail_first, []

    def search(self, **kw):
        self.calls.append(kw)
        if len(self.calls) <= self.fail_first:
            raise ConnectionError("temporary")
        items = self.items_

        class Search:
            def items(self):
                return iter(items)
        return Search()


def _item(id_, when):
    return pystac.Item(id_, None, None, when, {})


def test_search_items_passes_filters_and_retries():
    items = [_item("a", dt.datetime(2020, 1, 1)), _item("b", dt.datetime(2020, 1, 17))]
    cat = FakeCatalog(items, fail_first=1)
    out = stac.search_items("modis-13Q1-061", (125.0, 7.0, 125.1, 7.1), "2020-01-01",
                            "2020-12-31", {"platform": {"eq": "terra"}}, catalog=cat)
    assert [i.id for i in out] == ["a", "b"]
    assert len(cat.calls) == 2
    assert cat.calls[-1] == {"collections": ["modis-13Q1-061"],
                             "bbox": [125.0, 7.0, 125.1, 7.1],
                             "datetime": "2020-01-01/2020-12-31",
                             "query": {"platform": {"eq": "terra"}}}


def test_item_date_prefers_start_datetime():
    it = pystac.Item("x", None, None, None, {"start_datetime": "2020-02-18T00:00:00Z",
                                             "end_datetime": "2020-03-04T23:59:59Z"})
    assert stac.item_date(it) == pd.Timestamp("2020-02-18")
    assert stac.item_date(_item("y", dt.datetime(2021, 5, 1, 10))) == pd.Timestamp("2021-05-01")


def test_item_end_date_prefers_end_datetime():
    it = pystac.Item("x", None, None, None, {"start_datetime": "2020-03-21T00:00:00Z",
                                             "end_datetime": "2020-04-05T23:59:59Z"})
    assert stac.item_end_date(it) == pd.Timestamp("2020-04-05")
    assert stac.item_end_date(_item("y", dt.datetime(2021, 5, 1, 10))) == pd.Timestamp("2021-05-01")


def test_chunk_cache_roundtrip(tmp_path):
    path = stac.chunk_path(tmp_path, "Davao de Oro", "2020-01-01")
    assert path == tmp_path / "davao_de_oro" / "2020_01_01.parquet"
    assert stac.read_chunk(path) is None
    df = pd.DataFrame({"a": [1, 2]})
    stac.write_chunk(df, path)
    pd.testing.assert_frame_equal(stac.read_chunk(path), df)
    assert not list(path.parent.glob("*.part"))
