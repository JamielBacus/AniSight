import pytest

from cropyield import download_psa as d


def _meta():
    return {
        "title": "test",
        "variables": [
            {"code": "Ecosystem/Croptype", "values": [str(i) for i in range(6)],
             "valueTexts": d.CROPS},
            {"code": "Geolocation", "values": ["0", "83", "84", "85", "86", "87", "88", "89"],
             "valueTexts": ["PHILIPPINES", "..REGION XI (DAVAO REGION)",
                            "....Davao de Oro (Compostela Valley)", "....Davao del Norte",
                            "....Davao del Sur", "....Davao Occidental", "....Davao Oriental",
                            "....City of Davao"]},
            {"code": "Year", "values": ["0", "1"], "valueTexts": ["2024", "2025"]},
            {"code": "Period", "values": [str(i) for i in range(7)],
             "valueTexts": ["Quarter 1", "Quarter 2", "Semester 1", "Quarter 3",
                            "Quarter 4", "Semester 2", "Annual"]},
        ],
    }


def test_build_query_selects_by_label():
    q = {v["code"]: v["selection"]["values"] for v in d.build_query(_meta())["query"]}
    assert q["Geolocation"] == ["83", "84", "85", "86", "87", "88", "89"]
    assert q["Period"] == ["0", "1", "3", "4", "6"]
    assert q["Year"] == ["0", "1"]


def test_build_query_fails_loudly_on_missing_label():
    meta = _meta()
    meta["variables"][1]["valueTexts"][3] = "....Renamed Province"
    with pytest.raises(ValueError, match="Davao del Norte"):
        d.build_query(meta)


def test_check_cube_rejects_collapsed_response():
    query = d.build_query(_meta())  # 7 geos x 6 crops x 2 years x 5 periods = 420
    with pytest.raises(RuntimeError, match="Expected 420 cells"):
        d.check_cube({"size": [1, 1, 1, 1], "value": [1.0]}, query)
    d.check_cube({"size": [6, 7, 2, 5], "value": [0.0] * 420}, query)  # passes


def test_jsonstat2_to_tidy_frame():
    ds = {
        "id": ["Geolocation", "Ecosystem/Croptype", "Year", "Period"],
        "size": [2, 1, 2, 1],
        "dimension": {
            "Geolocation": {"category": {"index": {"84": 0, "85": 1},
                                         "label": {"84": "....Davao de Oro (Compostela Valley)",
                                                   "85": "....Davao del Norte"}}},
            "Ecosystem/Croptype": {"category": {"index": {"2": 0}, "label": {"2": "Palay"}}},
            "Year": {"category": {"index": {"0": 0, "1": 1}, "label": {"0": "2024", "1": "2025"}}},
            "Period": {"category": {"index": {"0": 0}, "label": {"0": "Quarter 1"}}},
        },
        "value": [100.5, None, 200.0, 210.0],
    }
    df = d.tidy(d.jsonstat2_to_frame(ds, "production_mt"), "production_mt")
    assert list(df.columns) == ["geolocation", "croptype", "year", "period", "production_mt"]
    assert len(df) == 4
    row = df[(df.geolocation == "Davao del Norte") & (df.year == 2025)].iloc[0]
    assert row.production_mt == 210.0
    assert df.production_mt.isna().sum() == 1
