"""`cropyield` command line: one command per pipeline step.

Each command forwards its options to the step module's `main(argv)`, so
`cropyield <cmd>` and `python -m cropyield.<module>` behave the same. Step modules
are imported inside the commands to keep `--help` fast. The PSA download stays
`python -m cropyield.download_psa`.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

import typer

app = typer.Typer(help="Crop yield estimator pipeline")

CONFIG_HELP = "folder with pipeline.yaml and provinces.yaml (default: repo configs/)"


class Step(str, Enum):
    boundaries = "boundaries"
    cropland = "cropland"
    all = "all"


def _argv(**opts: object) -> list[str]:
    """Turn the options that were given into argparse flags (`config_dir` -> `--config-dir`)."""
    argv: list[str] = []
    for name, value in opts.items():
        if value is not None:
            argv += [f"--{name.replace('_', '-')}", str(value)]
    return argv


@app.command()
def version() -> None:
    """Print the package version."""
    print("cropyield 0.1.0")


@app.command()
def boundaries(
    config_dir: Optional[str] = typer.Option(None, help=CONFIG_HELP),
    step: Step = typer.Option(Step.all, help="'cropland' reuses data/interim/boundaries.parquet"),
) -> None:
    """Fetch province boundaries and/or WorldCover cropland masks into data/interim."""
    from cropyield import boundaries as mod
    mod.main(_argv(config_dir=config_dir, step=step.value))


@app.command("ground-truth")
def ground_truth(config_dir: Optional[str] = typer.Option(None, help=CONFIG_HELP)) -> None:
    """Build data/interim/ground_truth.parquet from the PSA CSVs in data/raw/psa."""
    from cropyield import ground_truth as mod
    mod.main(_argv(config_dir=config_dir))


@app.command()
def modis(
    config_dir: Optional[str] = typer.Option(None, help=CONFIG_HELP),
    start: Optional[str] = typer.Option(
        None, help="first composite end date, YYYY-MM-DD (default: dates.modis_start)"),
    end: Optional[str] = typer.Option(
        None, help="last composite end date, YYYY-MM-DD (default: dates.end)"),
    min_cropland_frac: Optional[float] = typer.Option(
        None, help="share of a 231 m cell that must be cropland (default: 0.5)"),
) -> None:
    """Extract MODIS NDVI/EVI per province into data/interim/modis_vi.parquet."""
    from cropyield.satellite import modis as mod
    mod.main(_argv(config_dir=config_dir, start=start, end=end,
                   min_cropland_frac=min_cropland_frac))


@app.command()
def sentinel2(
    config_dir: Optional[str] = typer.Option(None, help=CONFIG_HELP),
    start: Optional[str] = typer.Option(
        None, help="first date, YYYY-MM-DD (default: dates.sentinel2_start)"),
    end: Optional[str] = typer.Option(None, help="last date, YYYY-MM-DD (default: dates.end)"),
    coarsen: Optional[int] = typer.Option(
        None, help="read on an N-times coarser grid than the 10 m mask (default: 6, ~60 m)"),
) -> None:
    """Extract monthly Sentinel-2 NDVI/EVI per province into data/interim/s2_vi.parquet."""
    from cropyield.satellite import sentinel2 as mod
    mod.main(_argv(config_dir=config_dir, start=start, end=end, coarsen=coarsen))


@app.command()
def features(config_dir: Optional[str] = typer.Option(None, help=CONFIG_HELP)) -> None:
    """Build data/processed/features.parquet and reports/data_quality.md."""
    from cropyield import features as mod
    mod.main(_argv(config_dir=config_dir))


class RainStep(str, Enum):
    download = "download"
    aggregate = "aggregate"
    all = "all"


@app.command()
def rainfall(
    config_dir: Optional[str] = typer.Option(None, help=CONFIG_HELP),
    start: Optional[str] = typer.Option(
        None, help="first month, YYYY-MM (default: dates.rainfall_start)"),
    end: Optional[str] = typer.Option(None, help="last month, YYYY-MM (default: dates.end)"),
    step: RainStep = typer.Option(
        RainStep.all, help="'download' caches Open-Meteo JSON only; 'aggregate' uses the "
                           "cache only"),
    province: Optional[str] = typer.Option(
        None, help="download only this province (smoke tests); aggregate still uses all"),
) -> None:
    """Cache Open-Meteo ERA5 daily rainfall and write data/interim/rainfall.parquet."""
    from cropyield import rainfall as mod
    mod.main(_argv(config_dir=config_dir, start=start, end=end, step=step.value,
                   province=province))


class EvalEra(str, Enum):
    modis = "modis"
    s2 = "s2"
    all = "all"


class EvalModels(str, Enum):
    baselines = "baselines"
    satellite = "satellite"
    all = "all"


@app.command()
def evaluate(
    config_dir: Optional[str] = typer.Option(None, help=CONFIG_HELP),
    era: EvalEra = typer.Option(EvalEra.all, help="modeling era (modeling.eras) or all"),
    models: Optional[EvalModels] = typer.Option(
        None, help="baselines (default), satellite (ridge + boosting per feature set "
                   "vs B2), or all"),
    feature_sets: Optional[str] = typer.Option(
        None, help="satellite models only: comma-separated feature sets, 'name' or "
                   "'name@era' (e.g. rain,modis_rain,s2_rain,rain@s2); default: all sets"),
) -> None:
    """Rolling-origin evaluation of baselines and satellite models; appends to
    reports/experiments.csv."""
    from cropyield.models import evaluate as mod
    mod.main(_argv(config_dir=config_dir, era=era.value,
                   models=models.value if models else None, feature_sets=feature_sets))


@app.command()
def select(config_dir: Optional[str] = typer.Option(None, help=CONFIG_HELP)) -> None:
    """Select a model per crop (MODIS era), write models/<crop>.joblib,
    data/processed/predictions.parquet and reports/phase2_results.md."""
    from cropyield.models import select as mod
    mod.main(_argv(config_dir=config_dir))


@app.command()
def forecast(
    quarter: str = typer.Option(..., help="target quarter, YYYYQn (e.g. 2026Q4)"),
    backfill: bool = typer.Option(
        False, "--backfill", help="allow a quarter PSA already published; rows are "
                                  "marked backfill=True"),
    config_dir: Optional[str] = typer.Option(None, help=CONFIG_HELP),
) -> None:
    """Forecast a quarter per reporting unit and crop; append to reports/forecasts.csv."""
    from cropyield.models import forecast as mod
    mod.main(_argv(config_dir=config_dir, quarter=quarter)
             + (["--backfill"] if backfill else []))


@app.command()
def score(config_dir: Optional[str] = typer.Option(None, help=CONFIG_HELP)) -> None:
    """Score logged forecasts against PSA actuals (first-seen and latest); write
    reports/live_accuracy.csv and reports/live_accuracy_summary.csv."""
    from cropyield.models import forecast as mod
    mod.score_main(_argv(config_dir=config_dir))


@app.command("export")
def export_cmd(
    config_dir: Optional[str] = typer.Option(None, help=CONFIG_HELP),
    app_dir: Optional[str] = typer.Option(None, help="site folder (default: <repo>/app)"),
) -> None:
    """Write app/data/*.json, app/data/units.geojson and app/assets/figures/ for the site."""
    from cropyield import export as mod
    mod.main(_argv(config_dir=config_dir, app_dir=app_dir))


if __name__ == "__main__":
    app()
