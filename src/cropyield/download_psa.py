"""Download PSA OpenSTAT palay/corn production and area harvested via the PXWeb API.

The OpenSTAT web pages sit behind a Cloudflare check, but the PXWeb JSON API at
/PXWeb/api/v1/ answers directly. This script asks the API for each table's metadata,
picks the wanted provinces/crops/periods by their labels (so it keeps working if PSA
renumbers the codes), downloads the data as JSON-stat (v1) and saves tidy CSVs.

Usage:
    python -m cropyield.download_psa            # Davao Region, all years, quarters
    python -m cropyield.download_psa --out data/raw/psa
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import pandas as pd
import requests

API = "https://openstat.psa.gov.ph/PXWeb/api/v1/en/DB/2E/CS/"
TABLES = {
    "production": ("0012E4EVCP0.px", "production_mt"),
    "area": ("0022E4EAHC0.px", "area_ha"),
}
# Labels as they appear in OpenSTAT (leading dots are stripped before matching).
GEOS = [
    "REGION XI (DAVAO REGION)",
    "Davao de Oro (Compostela Valley)",
    "Davao del Norte",
    "Davao del Sur",
    "Davao Occidental",
    "Davao Oriental",
    "City of Davao",
]
CROPS = ["Irrigated Palay", "Rainfed Palay", "Palay", "White Corn", "Yellow Corn", "Corn"]
PERIODS = ["Quarter 1", "Quarter 2", "Quarter 3", "Quarter 4", "Annual"]

HEADERS = {"User-Agent": "cropyield-student-project/0.1 (+python-requests)"}


def _request(session: requests.Session, method: str, url: str, **kw) -> requests.Response:
    """HTTP request with retries on rate limits and server errors."""
    kw.setdefault("timeout", (10, 60))  # 10 s to connect, 60 s to receive
    for attempt in range(3):
        print(f"  {method} {url.rsplit('/', 1)[-1]} (attempt {attempt + 1}/3) ...", flush=True)
        try:
            r = session.request(method, url, **kw)
        except requests.exceptions.RequestException as e:
            print(f"  -> network error: {type(e).__name__}: {e}", flush=True)
            if attempt < 2:
                time.sleep(3)
                continue
            raise RuntimeError(
                "Could not reach openstat.psa.gov.ph. Check that the site opens in your "
                "browser, or try another network (e.g. phone hotspot).") from e
        print(f"  -> HTTP {r.status_code}, {len(r.content):,} bytes", flush=True)
        if r.status_code in (429, 500, 502, 503, 504) and attempt < 2:
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 403:
            raise RuntimeError(
                f"403 from {url}. Cloudflare may be blocking this connection; "
                "try again later or from another network.")
        r.raise_for_status()
        return r
    raise RuntimeError("unreachable")


def _codes(variable: dict, wanted: list[str] | None) -> list[str]:
    """Map wanted labels to value codes; None means all values."""
    if wanted is None:
        return list(variable["values"])
    labels = {t.lstrip(".").strip(): c for c, t in zip(variable["values"], variable["valueTexts"])}
    missing = [w for w in wanted if w not in labels]
    if missing:
        raise ValueError(f"{variable['code']}: labels not found in OpenSTAT metadata: {missing}")
    return [labels[w] for w in wanted]


def build_query(meta: dict) -> dict:
    wanted = {"Geolocation": GEOS, "Ecosystem/Croptype": CROPS, "Period": PERIODS, "Year": None}
    query = []
    for var in meta["variables"]:
        if var["code"] not in wanted:
            raise ValueError(f"Unexpected variable in table: {var['code']}")
        query.append({
            "code": var["code"],
            "selection": {"filter": "item", "values": _codes(var, wanted[var["code"]])},
        })
    # OpenSTAT's json-stat2 output is broken: correct size but "value": [1.0].
    # json-stat (v1) returns the full cube; download() unwraps its "dataset" key.
    return {"query": query, "response": {"format": "json-stat"}}


def jsonstat2_to_frame(ds: dict, value_name: str) -> pd.DataFrame:
    """Convert a JSON-stat2 dataset into a long DataFrame with one column per dimension."""
    ids, sizes = ds["id"], ds["size"]
    labels = []
    for dim_id in ids:
        cat = ds["dimension"][dim_id]["category"]
        index = cat["index"]
        order = sorted(index, key=index.get) if isinstance(index, dict) else list(index)
        labels.append([cat["label"][code].lstrip(".").strip() for code in order])
    values = ds["value"]
    if isinstance(values, dict):  # sparse form: {"position": value}
        dense = [None] * math.prod(sizes)
        for pos, v in values.items():
            dense[int(pos)] = v
        values = dense
    rows = [combo + (v,) for combo, v in zip(itertools.product(*labels), values)]
    return pd.DataFrame(rows, columns=[*ids, value_name])


def tidy(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    df = df.rename(columns={"Geolocation": "geolocation", "Ecosystem/Croptype": "croptype",
                            "Period": "period", "Year": "year"})
    df["year"] = df["year"].astype(int)
    df[value_name] = pd.to_numeric(df[value_name], errors="coerce")
    return df[["geolocation", "croptype", "year", "period", value_name]].sort_values(
        ["geolocation", "croptype", "year", "period"]).reset_index(drop=True)


def check_cube(ds: dict, query: dict) -> None:
    """Raise if the server returned a different shape than we asked for."""
    expected = math.prod(len(q["selection"]["values"]) for q in query["query"])
    values = ds.get("value")
    n_values = len(values) if isinstance(values, list) else None
    got = math.prod(ds["size"]) if "size" in ds else None
    if got != expected or (n_values is not None and n_values != expected):
        raise RuntimeError(
            f"Expected {expected:,} cells but the API returned size={ds.get('size')} "
            f"with {n_values} values. The raw response was saved next to the CSVs; "
            "send it to Claude to diagnose.")


def download(out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    with requests.Session() as s:
        s.headers.update(HEADERS)
        for name, (table, value_name) in TABLES.items():
            print(f"[{name}] fetching table info ...", flush=True)
            meta = _request(s, "GET", API + table).json()
            print(f"[{name}] {meta['title']}")
            query = build_query(meta)
            print(f"[{name}] downloading data ...", flush=True)
            r = _request(s, "POST", API + table, data=json.dumps(query),
                         headers={"Content-Type": "application/json"}, timeout=(10, 120))
            raw = out_dir / f"_raw_{name}_response.json"
            raw.write_bytes(r.content)
            ds = json.loads(r.content.decode("utf-8-sig"))
            if "dataset" in ds:  # JSON-stat v1 wrapper: id/size live under "dimension"
                ds = ds["dataset"]
                ds.setdefault("id", ds["dimension"].get("id"))
                ds.setdefault("size", ds["dimension"].get("size"))
            check_cube(ds, query)
            df = tidy(jsonstat2_to_frame(ds, value_name), value_name)
            path = out_dir / f"palay_corn_{name}.csv"
            df.to_csv(path, index=False)
            print(f"[{name}] {len(df):,} rows, {df[value_name].isna().sum():,} missing -> {path}")
            written[name] = path
            time.sleep(1)  # be polite to the server
    return written


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=Path("data/raw/psa"))
    download(p.parse_args().out)


if __name__ == "__main__":
    main()
