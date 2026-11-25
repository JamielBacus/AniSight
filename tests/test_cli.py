"""CLI wiring: each command forwards its options to the step module's main(argv)."""
from __future__ import annotations

import pytest
from typer.testing import CliRunner

import cropyield.boundaries
import cropyield.features
import cropyield.ground_truth
import cropyield.models.evaluate
import cropyield.rainfall
import cropyield.satellite.modis
import cropyield.satellite.sentinel2
from cropyield.cli import app

runner = CliRunner()

MODULES = {
    "boundaries": cropyield.boundaries,
    "ground-truth": cropyield.ground_truth,
    "modis": cropyield.satellite.modis,
    "sentinel2": cropyield.satellite.sentinel2,
    "features": cropyield.features,
    "evaluate": cropyield.models.evaluate,
    "rainfall": cropyield.rainfall,
}


@pytest.fixture
def calls(monkeypatch):
    """Replace every step's main() with a recorder; no step logic runs."""
    seen: dict[str, list[str]] = {}
    for cmd, mod in MODULES.items():
        monkeypatch.setattr(mod, "main", lambda argv=None, cmd=cmd: seen.__setitem__(cmd, argv))
    return seen


def test_help_lists_all_commands():
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0
    for cmd in [*MODULES, "version"]:
        assert cmd in res.output


@pytest.mark.parametrize("cmd", list(MODULES))
def test_command_help(cmd):
    res = runner.invoke(app, [cmd, "--help"])
    assert res.exit_code == 0
    assert "--config-dir" in res.output


@pytest.mark.parametrize("cmd", ["ground-truth", "sentinel2", "features"])
def test_defaults_pass_no_flags(calls, cmd):
    res = runner.invoke(app, [cmd])
    assert res.exit_code == 0, res.output
    assert calls == {cmd: []}


def test_boundaries_defaults_to_all(calls):
    res = runner.invoke(app, ["boundaries"])
    assert res.exit_code == 0, res.output
    assert calls == {"boundaries": ["--step", "all"]}


def test_boundaries_options(calls):
    res = runner.invoke(app, ["boundaries", "--config-dir", "cfg", "--step", "cropland"])
    assert res.exit_code == 0, res.output
    assert calls["boundaries"] == ["--config-dir", "cfg", "--step", "cropland"]


def test_boundaries_rejects_unknown_step(calls):
    res = runner.invoke(app, ["boundaries", "--step", "bogus"])
    assert res.exit_code != 0
    assert calls == {}


def test_modis_options(calls):
    res = runner.invoke(app, ["modis", "--config-dir", "cfg", "--start", "2010-01-01",
                              "--end", "2010-12-31", "--min-cropland-frac", "0.3"])
    assert res.exit_code == 0, res.output
    assert calls["modis"] == ["--config-dir", "cfg", "--start", "2010-01-01",
                              "--end", "2010-12-31", "--min-cropland-frac", "0.3"]


def test_modis_defaults_pass_no_flags(calls):
    res = runner.invoke(app, ["modis"])
    assert res.exit_code == 0, res.output
    assert calls == {"modis": []}


def test_sentinel2_options(calls):
    res = runner.invoke(app, ["sentinel2", "--start", "2019-01-01", "--end", "2019-06-30"])
    assert res.exit_code == 0, res.output
    assert calls["sentinel2"] == ["--start", "2019-01-01", "--end", "2019-06-30"]


@pytest.mark.parametrize("cmd", ["ground-truth", "features"])
def test_config_dir_forwarded(calls, cmd):
    res = runner.invoke(app, [cmd, "--config-dir", "cfg"])
    assert res.exit_code == 0, res.output
    assert calls[cmd] == ["--config-dir", "cfg"]


def test_evaluate_options(calls):
    res = runner.invoke(app, ["evaluate"])
    assert res.exit_code == 0, res.output
    assert calls["evaluate"] == ["--era", "all"]
    res = runner.invoke(app, ["evaluate", "--config-dir", "cfg", "--era", "s2"])
    assert res.exit_code == 0, res.output
    assert calls["evaluate"] == ["--config-dir", "cfg", "--era", "s2"]
    assert runner.invoke(app, ["evaluate", "--era", "landsat"]).exit_code != 0
    res = runner.invoke(app, ["evaluate", "--models", "all"])
    assert res.exit_code == 0, res.output
    assert calls["evaluate"] == ["--era", "all", "--models", "all"]
    assert runner.invoke(app, ["evaluate", "--models", "deep"]).exit_code != 0


def test_rainfall_options(calls):
    res = runner.invoke(app, ["rainfall"])
    assert res.exit_code == 0, res.output
    assert calls["rainfall"] == ["--step", "all"]
    res = runner.invoke(app, ["rainfall", "--start", "2024-01", "--end", "2024-03",
                              "--step", "aggregate"])
    assert res.exit_code == 0, res.output
    assert calls["rainfall"] == ["--start", "2024-01", "--end", "2024-03", "--step", "aggregate"]
    assert runner.invoke(app, ["rainfall", "--step", "bogus"]).exit_code != 0


def test_version():
    res = runner.invoke(app, ["version"])
    assert res.exit_code == 0
    assert "cropyield 0.1.0" in res.output
