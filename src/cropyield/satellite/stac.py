"""Shared Planetary Computer STAC helpers: client, search, retries and a Parquet chunk cache.

Used by the MODIS and Sentinel-2 extractors. Searches return items whose asset hrefs are
already signed (`planetary_computer.sign_inplace`). Signed URLs expire after a while, so
long runs should search in short windows (e.g. one year at a time) just before loading.

Chunk cache: each completed unit of work (e.g. one province and composite date) is written
as a small Parquet file under `data/interim/cache/<name>/`. Re-runs read it instead of
downloading again. Writes are atomic (temp file + rename), so an interrupted run never
leaves a half-written chunk behind.
"""
from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar
from urllib.parse import urlparse

import os

# Long Sentinel-2/MODIS runs read thousands of byte ranges over home networks; dropped
# connections show up as GDAL "response_code=0" / "206" errors. Make GDAL retry harder
# before a read fails (user-set environment values win). Must be set before GDAL reads.
for _k, _v in {"GDAL_HTTP_MAX_RETRY": "5", "GDAL_HTTP_RETRY_DELAY": "2",
               "GDAL_HTTP_TIMEOUT": "60", "GDAL_HTTP_CONNECTTIMEOUT": "20"}.items():
    os.environ.setdefault(_k, _v)

import pandas as pd
import planetary_computer
import planetary_computer.sas as pc_sas
import pystac
import pystac_client

from cropyield.config import Config, load_config

log = logging.getLogger(__name__)

T = TypeVar("T")
Bbox = tuple[float, float, float, float]


def call_with_retry(
    fn: Callable[[], T], *, attempts: int = 3, base_delay: float = 2.0, label: str = ""
) -> T:
    """Call `fn()` and retry on any exception with exponential backoff (2 s, 4 s, ...).

    Raises the last exception once `attempts` calls have failed.
    """
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as e:  # network/IO errors come from several libraries
            if attempt == attempts - 1:
                raise
            delay = base_delay * 2 ** attempt
            log.warning("%s failed (%s: %s), retrying in %.0f s",
                        label or getattr(fn, "__name__", "call"), type(e).__name__, e, delay)
            time.sleep(delay)
    raise RuntimeError("unreachable")


# Planetary Computer SAS tokens expire (about a day). planetary_computer reuses a cached
# token until it has under 60 s left and never re-signs an already-signed URL, so a
# long run can sign assets with a nearly expired token and then get HTTP 403 mid-read.
TOKEN_MIN_TTL_S = 30 * 60
BLOB_DOMAIN = ".blob.core.windows.net"


def drop_expiring_tokens(min_ttl_s: float = TOKEN_MIN_TTL_S) -> int:
    """Remove cached SAS tokens with less than `min_ttl_s` seconds left; returns how many."""
    stale = [k for k, tok in list(pc_sas.TOKEN_CACHE.items()) if tok.ttl() < min_ttl_s]
    for k in stale:
        pc_sas.TOKEN_CACHE.pop(k, None)
    return len(stale)


def fresh_sign(items: list[pystac.Item], min_ttl_s: float = TOKEN_MIN_TTL_S) -> list[pystac.Item]:
    """Return copies of `items` re-signed with a token valid for at least `min_ttl_s`.

    Existing SAS query strings on Azure Blob asset URLs are stripped first, because
    planetary_computer skips URLs that already look signed. Call this right before reading.
    """
    drop_expiring_tokens(min_ttl_s)
    out = []
    for item in items:
        clone = item.clone()
        for asset in clone.assets.values():
            parsed = urlparse(asset.href)
            if parsed.netloc.endswith(BLOB_DOMAIN) and parsed.query:
                asset.href = parsed._replace(query="").geturl()
        out.append(planetary_computer.sign(clone))
    return out


def open_catalog(stac_url: str | None = None) -> pystac_client.Client:
    """Open the Planetary Computer STAC API with automatic asset signing.

    Args:
        stac_url: API root; defaults to `sources.stac_url` in `configs/pipeline.yaml`.
    """
    url = stac_url or load_config().pipeline.sources.stac_url
    return call_with_retry(
        lambda: pystac_client.Client.open(url, modifier=planetary_computer.sign_inplace),
        label=f"open {url}")


def search_items(
    collection: str,
    bbox: Bbox,
    start: str,
    end: str,
    query: dict | None = None,
    *,
    catalog: pystac_client.Client | None = None,
) -> list[pystac.Item]:
    """Search one collection for items intersecting `bbox` between `start` and `end`.

    Args:
        collection: STAC collection id, e.g. "modis-13Q1-061".
        bbox: (minx, miny, maxx, maxy) in EPSG:4326.
        start, end: ISO dates ("YYYY-MM-DD"), both inclusive.
        query: optional STAC query extension filter, e.g. {"eo:cloud_cover": {"lt": 80}}.
        catalog: an open client to reuse; a new one is opened if omitted.

    Returns:
        Signed items (the whole result is fetched, with retries, before returning).
    """
    catalog = catalog or open_catalog()

    def run() -> list[pystac.Item]:
        drop_expiring_tokens()  # results get signed with a token that won't lapse mid-read
        search = catalog.search(collections=[collection], bbox=list(bbox),
                                datetime=f"{start}/{end}", query=query)
        return list(search.items())

    return call_with_retry(run, label=f"search {collection} {start}/{end}")


def item_date(item: pystac.Item) -> pd.Timestamp:
    """Acquisition/composite date of an item: `start_datetime` if present, else `datetime`."""
    value = item.properties.get("start_datetime") or item.datetime
    if value is None:
        raise ValueError(f"item {item.id} has no datetime or start_datetime")
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts.normalize()


def item_end_date(item: pystac.Item) -> pd.Timestamp:
    """Last day an item covers: `end_datetime` if present, else `datetime`.

    Use this for composites (e.g. MODIS 16-day) so a period cutoff never admits imagery
    acquired after the cutoff.
    """
    value = item.properties.get("end_datetime") or item.datetime
    if value is None:
        raise ValueError(f"item {item.id} has no datetime or end_datetime")
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts.normalize()


# ------------------------------------------------------------------------- chunk cache
def slug(name: str) -> str:
    """File-name-safe lowercase slug, e.g. "Davao de Oro" -> "davao_de_oro"."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).casefold()).strip("_")


def cache_dir(cfg: Config, name: str) -> Path:
    """Cache folder for one extractor: `<paths.interim>/cache/<name>/`."""
    return cfg.pipeline.paths.interim / "cache" / name


def chunk_path(root: Path, *parts: str) -> Path:
    """Path of one cached chunk: `root/<slug(part1)>/.../<slug(last)>.parquet`."""
    *dirs, last = [slug(p) for p in parts]
    return root.joinpath(*dirs, f"{last}.parquet")


def read_chunk(path: Path) -> pd.DataFrame | None:
    """Return a cached chunk, or None if it is missing."""
    return pd.read_parquet(path) if path.exists() else None


def write_chunk(df: pd.DataFrame, path: Path) -> Path:
    """Write a chunk atomically so an interrupted run never leaves a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.part")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
    return path
