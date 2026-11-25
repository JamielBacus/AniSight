from datetime import date
from pathlib import Path

import pytest

from cropyield.config import (DEFAULT_CONFIG_DIR, ReportingWindow, load_config,
                              load_pipeline, load_provinces)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_load_sample_config():
    cfg = load_config()
    p = cfg.pipeline
    assert p.crops == ("palay", "corn")
    assert p.dates.modis_start == date(2001, 1, 1)
    assert p.dates.sentinel2_start == date(2017, 4, 1)
    assert p.sources.modis_collection == "modis-13Q1-061"
    assert p.cropland_class == 40
    assert p.s2_cloud_scl_classes == (3, 8, 9, 10)
    assert p.random_seed == 42
    assert p.paths.raw == REPO_ROOT / "data" / "raw"

    prov = cfg.provinces
    assert prov.region == "Davao Region"
    assert len(prov.names) == 6
    assert "Davao de Oro" in prov.names
    oro = next(x for x in prov.provinces if x.name == "Davao de Oro")
    assert "Compostela Valley" in oro.aliases
    city = next(x for x in prov.provinces if x.name == "Davao City")
    assert city.aliases == ("City of Davao",) and city.part_of == "Davao del Sur"
    assert oro.part_of is None


def test_load_pipeline_rejects_non_mapping(tmp_path):
    bad = tmp_path / "pipeline.yaml"
    bad.write_text("- just\n- a list\n")
    with pytest.raises(ValueError, match="mapping"):
        load_pipeline(bad)


def test_default_config_dir_is_repo_configs():
    assert DEFAULT_CONFIG_DIR == REPO_ROOT / "configs"


def test_reporting_units_parsed():
    prov = load_config().provinces
    assert prov.reporting_units == (
        ReportingWindow("Davao de Oro", "Davao del Norte", (1987, 1), (2001, 4)),
        ReportingWindow("Davao Occidental", "Davao del Sur", (1987, 1), (2018, 4)),
        ReportingWindow("Davao City", "Davao del Sur", (2021, 1), (2025, 1)),
    )
    assert prov.host_of("Davao City", 2025, 1) == "Davao del Sur"
    assert prov.host_of("Davao City", 2025, 2) is None
    assert prov.combined_reporting_units() == {
        "Davao del Norte+Davao de Oro": ("Davao del Norte", "Davao de Oro"),
        "Davao del Sur+Davao Occidental": ("Davao del Sur", "Davao Occidental"),
        "Davao del Sur+Davao City": ("Davao del Sur", "Davao City"),
    }


def _provinces_yaml(tmp_path, window: str) -> Path:
    path = tmp_path / "provinces.yaml"
    path.write_text("region: R\nprovinces:\n  - name: A\n  - name: B\n"
                    "reporting_units:\n" + window)
    return path


@pytest.mark.parametrize("window, match", [
    ("  - {unit: A, host: Nowhere, first: 2000-Q1, last: 2001-Q4}\n", "host 'Nowhere'"),
    ("  - {unit: Zed, host: B, first: 2000-Q1, last: 2001-Q4}\n", "unit 'Zed'"),
    ("  - {unit: A, host: B, first: 2000Q5, last: 2001-Q4}\n", "YYYY-Qn"),
    ("  - {unit: A, host: B, first: 2002-Q1, last: 2001-Q4}\n", "after last"),
])
def test_reporting_units_rejects_bad_entries(tmp_path, window, match):
    with pytest.raises(ValueError, match=match):
        load_provinces(_provinces_yaml(tmp_path, window))


def test_reporting_units_optional(tmp_path):
    path = tmp_path / "provinces.yaml"
    path.write_text("region: R\nprovinces:\n  - name: A\n")
    assert load_provinces(path).reporting_units == ()
