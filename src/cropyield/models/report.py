"""Phase 2 results report: `reports/phase2_results.md` and `reports/figures/` (Req. 6.4).

Every number in the report is computed here from `reports/experiments.csv`, the
selection run (`cropyield.models.select.run_selection`, also written to
`reports/selection.csv` and `data/processed/predictions.parquet`), the modeling table,
or `data/interim/rainfall_points.csv` (cropland area). The ridge alpha sweep is
recomputed on each run as an exploratory check: it is not logged to experiments.csv and
not used for selection.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cropyield.config import Config  # noqa: E402
from cropyield.models.baselines import MeanPlusTrend  # noqa: E402
from cropyield.models.data import era_rows  # noqa: E402
from cropyield.models.evaluate import (bootstrap_mae_diff, cross_validate,  # noqa: E402
                                       experiments_path, metrics)
from cropyield.models.satellite import FEATURE_SETS, AnomalyRidge  # noqa: E402
from cropyield.models.select import (ANOMALY_SOURCE_FEATURES, CONDITIONS_COLUMNS,  # noqa: E402
                                     INTERVAL_QUANTILES, MIN_ANOMALY_YEARS,
                                     MIN_INTERVAL_YEARS, NAIVE, PREDICTION_COLUMNS,
                                     REFERENCE, SATELLITE_CANDIDATE_MODELS,
                                     SATELLITE_CANDIDATE_SETS, SelectionResult,
                                     models_dir, predictions_path, selection_path)

log = logging.getLogger(__name__)

ALPHA_SWEEP_SETS: list[tuple[str, str]] = [("rain", "modis"), ("modis_rain", "modis"),
                                           ("s2_rain", "s2"), ("rain", "s2")]
ALPHA_SWEEP_ALPHAS: tuple[float, ...] = (1.0, 100.0, 1000.0)


def report_path(cfg: Config) -> Path:
    return cfg.pipeline.paths.reports / "phase2_results.md"


def figures_dir(cfg: Config) -> Path:
    return cfg.pipeline.paths.reports / "figures"


# --- formatting helpers -------------------------------------------------------------

def _md(df: pd.DataFrame) -> str:
    """GitHub markdown table (no tabulate dependency)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v))
                                       else str(v) for v in row) + " |")
    return "\n".join(lines)


def _ci(d: float, lo: float, hi: float) -> str:
    if not np.isfinite(lo):
        return "(reference)"
    return f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]"


def _rel(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


# --- computations -------------------------------------------------------------------

def alpha_sweep(cfg: Config, df: pd.DataFrame,
                sets: Sequence[tuple[str, str]] = ALPHA_SWEEP_SETS,
                alphas: Sequence[float] = ALPHA_SWEEP_ALPHAS) -> pd.DataFrame:
    """Exploratory: AnomalyRidge MAE - MAE(B2) [95% CI] per feature set, era, alpha, crop.

    Not logged and not used for selection (the selected models use alpha = 1.0 as
    logged). Columns: feature_set, era, alpha, crop, d_b2, lo, hi.
    """
    eras = cfg.pipeline.eras
    seed = cfg.pipeline.random_seed
    b2 = {e: cross_validate(MeanPlusTrend, df, e, eras=eras) for e in {e for _, e in sets}}
    rows = []
    for fs, era in sets:
        for a in alphas:
            oof = cross_validate(lambda: AnomalyRidge(FEATURE_SETS[fs], alpha=a), df, era,
                                 eras=eras)
            for crop in sorted(oof["crop"].unique()):
                bs = bootstrap_mae_diff(oof.loc[oof["crop"] == crop],
                                        b2[era].loc[b2[era]["crop"] == crop], seed=seed)
                rows.append({"feature_set": fs, "era": era, "alpha": a, "crop": crop,
                             "d_b2": bs.diff, "lo": bs.ci_low, "hi": bs.ci_high})
    return pd.DataFrame(rows)


def per_unit_errors(res: SelectionResult) -> pd.DataFrame:
    """Selected model's MAE/RMSE/MAPE per crop and reporting unit, next to B0 and B2."""
    pred = res.predictions
    sel = pred.rename(columns={"actual_yield_t_ha": "yield_t_ha"})
    m = metrics(sel, by=["crop", "reporting_unit"])
    for ref, col in ((NAIVE, "b0_pred_yield_t_ha"), (REFERENCE, "b2_pred_yield_t_ha")):
        r = metrics(sel.assign(pred_yield_t_ha=sel[col]), by=["crop", "reporting_unit"])
        m[f"mae_{ref}"] = r["mae"].to_numpy()
    m.insert(1, "model", m["crop"].map({c: s[0] for c, s in res.selected.items()}))
    return m


# --- figures ------------------------------------------------------------------------

def _fig_mae_by_model(exp: pd.DataFrame, path: Path) -> None:
    combos = [(e, c) for e in ("modis", "s2") for c in sorted(exp["crop"].unique())
              if ((exp["era"] == e) & (exp["crop"] == c)).any()]
    fig, axes = plt.subplots(1, len(combos), figsize=(4.2 * len(combos), 6), squeeze=False)
    for ax, (era, crop) in zip(axes[0], combos):
        g = exp.loc[(exp["era"] == era) & (exp["crop"] == crop)].sort_values("mae")
        b2 = float(g.loc[g["model"] == REFERENCE, "mae"].iloc[0])
        colors = ["tab:gray" if ":" not in m else "tab:green" for m in g["model"]]
        ax.barh(g["model"], g["mae"], color=colors)
        ax.axvline(b2, color="k", ls="--", lw=1, label=f"B2 = {b2:.3f}")
        ax.invert_yaxis()
        ax.set_title(f"{crop}, {era} era")
        ax.set_xlabel("MAE (t/ha)")
        ax.tick_params(axis="y", labelsize=7)
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Out-of-fold MAE by model (grey: baselines, green: satellite/rainfall)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _fig_observed_vs_predicted(pred: pd.DataFrame, crop: str, path: Path) -> None:
    g = pred.loc[pred["crop"] == crop]
    units = sorted(g["reporting_unit"].unique())
    ncol = 3
    nrow = int(np.ceil(len(units) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(14, 3.0 * nrow), squeeze=False,
                             sharex=True)
    for ax, unit in zip(axes.flat, units):
        u = g.loc[g["reporting_unit"] == unit].sort_values(["year", "quarter"])
        u = u.assign(t=u["year"] + (u["quarter"] - 0.5) / 4)
        # one row per quarter so lines and bands break where the unit was not reported
        grid = np.round(np.arange(u["t"].min(), u["t"].max() + 0.01, 0.25), 3)
        u = u.assign(t=u["t"].round(3)).set_index("t").reindex(grid)
        t = u.index.to_numpy()
        ax.fill_between(t, u["pred_lo_t_ha"], u["pred_hi_t_ha"], color="tab:blue",
                        alpha=0.2, lw=0, label="80% interval")
        ax.plot(t, u["pred_yield_t_ha"], "-", color="tab:blue", lw=1, label="predicted")
        ax.plot(t, u["actual_yield_t_ha"], "o", color="k", ms=2.5, label="PSA actual")
        ax.set_title(unit, fontsize=9)
        ax.set_ylabel("t/ha", fontsize=8)
    for ax in list(axes.flat)[len(units):]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(f"{crop}: PSA yield vs out-of-fold prediction of "
                 f"{g['model'].iloc[0]} (rolling origin)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _fig_residuals_by_year(pred: pd.DataFrame, path: Path) -> None:
    crops = sorted(pred["crop"].unique())
    fig, axes = plt.subplots(1, len(crops), figsize=(6 * len(crops), 4), squeeze=False)
    for ax, crop in zip(axes[0], crops):
        g = pred.loc[pred["crop"] == crop]
        r = np.log(g["actual_yield_t_ha"]) - np.log(g["pred_yield_t_ha"])
        years = sorted(g["year"].unique())
        ax.boxplot([r[g["year"] == y] for y in years], tick_labels=[str(y) for y in years])
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"{crop} ({g['model'].iloc[0]})")
        ax.set_ylabel("log(actual / predicted)")
        ax.tick_params(axis="x", rotation=90, labelsize=7)
    fig.suptitle("Out-of-fold log-residuals by test year")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --- report -------------------------------------------------------------------------

def write_report(cfg: Config, df: pd.DataFrame, res: SelectionResult,
                 sweep: pd.DataFrame | None = None) -> Path:
    """Write `reports/phase2_results.md` and its figures; return the report path.

    `sweep` defaults to `alpha_sweep(cfg, df)` (exploratory, recomputed).
    """
    root = cfg.pipeline.paths.reports.parent
    out = report_path(cfg)
    figs = figures_dir(cfg)
    figs.mkdir(parents=True, exist_ok=True)
    exp = pd.read_csv(experiments_path(cfg))
    sweep = alpha_sweep(cfg, df) if sweep is None else sweep
    pred = res.predictions
    crops = sorted(res.selected)

    fig_paths = {"mae": figs / "mae_by_model.png", "resid": figs / "residuals_by_year.png",
                 **{f"obs_{c}": figs / f"observed_vs_predicted_{c}.png" for c in crops}}
    _fig_mae_by_model(exp, fig_paths["mae"])
    _fig_residuals_by_year(pred, fig_paths["resid"])
    for c in crops:
        _fig_observed_vs_predicted(pred, c, fig_paths[f"obs_{c}"])

    sat = exp.loc[exp["model"].str.contains(":")]
    n_sat = len(sat)
    n_better = int((sat["mae_diff_ci_high"] < 0).sum())
    n_worse = int((sat["mae_diff_ci_low"] > 0).sum())
    best_sat_rows = sat.sort_values("mae").groupby(["era", "crop"]).head(1)

    L: list[str] = []
    w = L.append
    w("# Phase 2 results: modeling and evaluation")
    w("")
    w(f"Generated by `cropyield select` (`src/cropyield/models/select.py`, "
      f"`src/cropyield/models/report.py`) on {res.timestamp.isoformat(timespec='seconds')}. "
      "Model metrics come from `reports/experiments.csv`; the selection table is also in "
      "`reports/selection.csv`; per-row predictions are in "
      f"`{_rel(predictions_path(cfg), root)}`.")
    w("")
    w("## Summary")
    w("")
    if n_better == 0:
        w(f"**Satellite and rainfall features did not improve on the baselines.** None of "
          f"the {n_sat} satellite/rainfall model runs (AnomalyRidge and BoostedModel on "
          "every feature set, both eras, both crops) has an MAE below B2 with a 95% "
          f"bootstrap CI that excludes zero; {n_worse} of them are significantly worse "
          "than B2.")
    else:
        w(f"{n_better} of {n_sat} satellite/rainfall model runs beat B2 with a 95% CI "
          f"excluding zero; {n_worse} are significantly worse. See the tables below.")
    w("")
    for c in crops:
        lab, reason = res.selected[c]
        note = (" B3 was added after seeing the B0/B2 test-fold results, so this CI is "
                "optimistic." if lab == "B3" else "")
        w(f"- **{c}: {lab}.** {reason}.{note}")
    w("")
    w("Best satellite/rainfall run per crop and era (point MAE, Δ vs B2 [95% CI]):")
    w("")
    for r in best_sat_rows.sort_values(["era", "crop"]).itertuples():
        w(f"- {r.crop}, {r.era} era: {r.model} MAE {r.mae:.4f} t/ha, Δ "
          f"{_ci(r.mae_diff, r.mae_diff_ci_low, r.mae_diff_ci_high)}")
    w("")

    w("## Setup")
    w("")
    f = res.folds
    w("- Target: log(yield_t_ha); metrics on yield_t_ha after exp(). MAE and RMSE in t/ha, "
      "MAPE in %. Δ = MAE(model) − MAE(reference) in t/ha (negative = model better); 95% "
      "CI from a paired bootstrap resampling whole test years (2,000 draws, seed "
      f"{cfg.pipeline.random_seed}).")
    w(f"- Rolling origin, expanding window, one test year per fold. MODIS era: rows from "
      f"{cfg.pipeline.eras['modis'].start_year}, test years {f['test_year'].min()}–"
      f"{f['test_year'].max()} ({len(f)} folds, {f['n_train_years'].min()}–"
      f"{f['n_train_years'].max()} training years per fold). Sentinel-2 era: rows from "
      f"{cfg.pipeline.eras['s2'].start_year}, test years from "
      f"{cfg.pipeline.eras['s2'].first_test_year}.")
    w("- Baselines: B0 same quarter last year (falls back to B1 when last year is "
      "missing); B1 unit-quarter mean; B2 B1 plus a per-crop linear year trend; B3 mean "
      "of the B0 and B2 log-predictions (post hoc, see caveats).")
    w("")
    w(_md(f.rename(columns={"test_year": "test year", "n_train_years": "train years",
                            "train_first_year": "train from", "n_train": "train rows",
                            "n_test": "test rows"})))
    w("")

    w("## Model selection (MODIS era decides)")
    w("")
    w("Rule: per crop, among B0, B2, B3 and the best satellite/rainfall model (lowest "
      f"MODIS-era MAE among {' / '.join(SATELLITE_CANDIDATE_MODELS)} × "
      f"{{{', '.join(SATELLITE_CANDIDATE_SETS)}}}, as logged with ridge alpha = 1), pick "
      "the lowest-MAE candidate whose MAE − MAE(B2) 95% CI lies entirely below zero; if "
      "none qualifies, pick B2. Sentinel-2-era results are shown below but not used.")
    w("")
    w("**Caveats.** B3 and the \"best satellite model\" were chosen after seeing the "
      "test-fold results, so their CIs are optimistic. Several comparisons are made per "
      "crop (each candidate vs B2 and vs B0) without multiple-comparison correction. The "
      "CIs resample only "
      f"{len(f)} test years.")
    w("")
    for c in crops:
        t = res.candidates.loc[res.candidates["crop"] == c]
        tab = pd.DataFrame({
            "candidate": t["candidate"], "role": t["role"], "n": t["n"],
            "MAE": t["mae"].map("{:.4f}".format), "RMSE": t["rmse"].map("{:.4f}".format),
            "MAPE %": t["mape"].map("{:.2f}".format),
            "Δ vs B2 [95% CI]": [_ci(*v) for v in t[["d_b2", "d_b2_lo", "d_b2_hi"]].values],
            "Δ vs B0 [95% CI]": [_ci(*v) for v in t[["d_b0", "d_b0_lo", "d_b0_hi"]].values],
            "CI vs B2 < 0": t["qualifies"].map({True: "yes", False: "no"}),
            "selected": t["selected"].map({True: "**yes**", False: ""})})
        w(f"### {c}")
        w("")
        w(_md(tab))
        w("")
        w(f"Selected: **{res.selected[c][0]}**. {res.selected[c][1]}.")
        w("")

    w("## Final models")
    w("")
    for c in crops:
        lab = res.selected[c][0]
        p = res.model_paths.get(c, models_dir(cfg) / f"{c}.joblib")
        w(f"- `{_rel(p, root)}`: {lab}, fitted on every MODIS-era row "
          f"({int(era_rows(df, 'modis', cfg.pipeline.eras)['year'].min())}–"
          f"{int(df['year'].max())}) of both crops, as in cross-validation.")
    if any(res.selected[c][0] in ("B0", "B3") for c in crops):
        w("- A model that uses B0 needs last year's same-quarter PSA yield at forecast "
          "time: refit with the latest PSA year before forecasting. Where that value is "
          "missing (reporting-unit regroupings), B0 falls back to B1, the unit-quarter "
          "mean.")
    w("- The saved bundle holds the fitted model, its feature list, training period and "
      "the 80% interval offsets from all out-of-fold residuals of that crop.")
    w("")

    w("## All models by crop and era")
    w("")
    w("From `reports/experiments.csv` (latest row per model, era and crop). B3 was run on "
      "the MODIS era only. Δ vs B2 is the paired bootstrap difference.")
    w("")
    for era in ("modis", "s2"):
        for c in crops:
            g = exp.loc[(exp["era"] == era) & (exp["crop"] == c)].sort_values("mae")
            if g.empty:
                continue
            tab = pd.DataFrame({
                "model": g["model"], "test years": g["test_years"], "n": g["n_test"],
                "MAE": g["mae"].map("{:.4f}".format), "RMSE": g["rmse"].map("{:.4f}".format),
                "MAPE %": g["mape"].map("{:.2f}".format),
                "MAE no outliers": g["mae_no_outliers"].map("{:.4f}".format),
                "Δ vs B2 [95% CI]": [_ci(*v) for v in g[["mae_diff", "mae_diff_ci_low",
                                                          "mae_diff_ci_high"]].fillna(
                    {"mae_diff": 0.0}).values]})
            w(f"### {c}, {era} era")
            w("")
            w(_md(tab))
            w("")
    w(f"![MAE by model]({_rel(fig_paths['mae'], out.parent)})")
    w("")

    w("## Errors per reporting unit (selected models)")
    w("")
    w("Out-of-fold MODIS-era test rows. MAE of B0 and B2 on the same rows for reference.")
    w("")
    pu = per_unit_errors(res)
    w(_md(pd.DataFrame({
        "crop": pu["crop"], "model": pu["model"], "reporting unit": pu["reporting_unit"],
        "n": pu["n"], "MAE": pu["mae"].map("{:.4f}".format),
        "RMSE": pu["rmse"].map("{:.4f}".format), "MAPE %": pu["mape"].map("{:.2f}".format),
        "MAE B0": pu["mae_B0"].map("{:.4f}".format),
        "MAE B2": pu["mae_B2"].map("{:.4f}".format)})))
    w("")
    for c in crops:
        w(f"![{c} observed vs predicted]({_rel(fig_paths[f'obs_{c}'], out.parent)})")
        w("")
    w(f"![Residuals by year]({_rel(fig_paths['resid'], out.parent)})")
    w("")

    w("## 80% prediction intervals")
    w("")
    lo_q, hi_q = INTERVAL_QUANTILES
    w(f"For crop c and test year Y, the interval is exp(prediction + q) with q the "
      f"{lo_q:.0%} and {hi_q:.0%} empirical quantiles of the selected model's out-of-fold "
      "log-residuals of crop c in test years before Y only, so no residual from year Y "
      f"or later is used. The first {MIN_INTERVAL_YEARS} test years have fewer than "
      f"{MIN_INTERVAL_YEARS} earlier years of residuals and get no interval (NaN). The "
      "saved final models use the quantiles of all out-of-fold residuals of their crop "
      "for forecasts. Empirical coverage on the rows that have an interval:")
    w("")
    cv = res.coverage
    w(_md(pd.DataFrame({"crop": cv["crop"], "model": cv["model"], "years": cv["years"],
                        "rows with interval": cv["n_with_interval"],
                        "coverage": cv["coverage"].map("{:.1%}".format),
                        "below": cv["below"].map("{:.1%}".format),
                        "above": cv["above"].map("{:.1%}".format),
                        "mean width t/ha": cv["mean_width_t_ha"].map("{:.3f}".format)})))
    w("")

    w("## Conditions vs normal (app columns, not model inputs)")
    w("")
    w("`predictions.parquet` keeps these per row although the selected models do not use "
      "them:")
    w("")
    for col, label in CONDITIONS_COLUMNS.items():
        w(f"- `{col}` (\"{label}\"): {PREDICTION_COLUMNS[col]}.")
    w("")
    m_era = era_rows(df, "modis", cfg.pipeline.eras)
    w(f"The NDVI/EVI anomalies need at least {MIN_ANOMALY_YEARS} earlier values of the same "
      "unit, crop and quarter, so units created by a regrouping have no anomaly for their "
      f"first {MIN_ANOMALY_YEARS} years. Source features: "
      f"{', '.join(ANOMALY_SOURCE_FEATURES)}.")
    w("")
    w(f"`rain_anom` is shifted: its mean is {df['rain_anom'].mean():+.0f} mm over all "
      f"modeling-table rows ({int(df['year'].min())}–{int(df['year'].max())}) and "
      f"{m_era['rain_anom'].mean():+.0f} mm over MODIS-era rows "
      f"({int(m_era['year'].min())}–{int(m_era['year'].max())}), i.e. recent windows are "
      "wetter than the 1981–2000 ERA5 climatology. This may be a real trend or drift in "
      "the ERA5 reanalysis; the app should not read a positive value as unusual without "
      "this context.")
    w("")

    w("## Exploratory: ridge alpha sweep")
    w("")
    w("Exploratory check, recomputed by `report.alpha_sweep`; not logged to "
      "experiments.csv and not used for selection (the logged models use alpha = 1). "
      "AnomalyRidge Δ vs B2 [95% CI]:")
    w("")
    sw = sweep.copy()
    sw["cell"] = [_ci(d, lo, hi) for d, lo, hi in sw[["d_b2", "lo", "hi"]].values]
    piv = sw.pivot_table(index=["feature_set", "era", "crop"], columns="alpha",
                         values="cell", aggfunc="first", sort=False).reset_index()
    piv.columns = [f"α = {c:g}" if isinstance(c, float) else c for c in piv.columns]
    w(_md(piv))
    w("")
    a_lo, a_hi = min(sweep["alpha"]), max(sweep["alpha"])
    k = ["feature_set", "era", "crop"]
    s_lo = sweep.loc[sweep["alpha"] == a_lo].set_index(k)
    s_hi = sweep.loc[sweep["alpha"] == a_hi].set_index(k).loc[s_lo.index]
    shrinks = int((s_hi["d_b2"].abs() <= s_lo["d_b2"].abs()).sum())
    gains = int((sweep["hi"] < 0).sum())
    w(f"In {shrinks} of {len(s_lo)} rows |Δ| at alpha = {a_hi:g} is no larger than at "
      f"alpha = {a_lo:g}, and {gains} of {len(sweep)} cells have a CI entirely below "
      "zero.")
    if shrinks == len(s_lo) and gains == 0:
        w("Stronger shrinkage pulls every ridge towards B2 (Δ → 0) and never produces a "
          "gain, so the ridge losses at alpha = 1 come from over-fitting noise, not from "
          "a signal that better tuning would recover.")
    w("")

    w("## Why satellites and rainfall did not help (likely reasons)")
    w("")
    w("- The ESA WorldCover cropland class mixes rice and corn with plantations (coconut, "
      "banana and other tree crops), so province-mean NDVI/EVI mostly tracks "
      "non-target vegetation.")
    pts_path = cfg.pipeline.paths.interim / "rainfall_points.csv"
    if pts_path.exists():
        pts = pd.read_csv(pts_path, usecols=["unit", "cropland_area"])
        w("- Mapped cropland is small in places: "
          + ", ".join(f"{r.unit} {r.cropland_area:,.0f} ha"
                      for r in pts.sort_values("cropland_area").itertuples())
          + " (WorldCover cropland, `data/interim/rainfall_points.csv`).")
    w("- Quarterly PSA survey estimates are noisy, and at province-quarter level that "
      "noise can swamp a weather or greenness signal.")
    s2_years = exp.loc[(exp["era"] == "s2") & (exp["model"] == REFERENCE), "test_years"]
    if len(s2_years):
        a, _, b = str(s2_years.iloc[0]).partition("-")
        w(f"- The Sentinel-2 era has only {int(b or a) - int(a) + 1} test years "
          f"({s2_years.iloc[0]}), so its CIs are wide.")
    w("- BoostedModel has no year input and cannot follow the yield trend that B2 "
      "captures.")
    w("")

    w("## Other limitations")
    w("")
    fb = pred.loc[pred["b0_fallback"]].groupby(["reporting_unit", "year"]).size()
    w("- Reporting-unit regroupings break B0 (no same-unit value last year) and shorten "
      "B1/B2 histories, so these rows use crude fallbacks. Rows where B0 fell back to B1 "
      "(unit, year: rows): "
      + "; ".join(f"{u} {y}: {n}" for (u, y), n in fb.items()) + ".")
    last = int(df["year"].max())
    qs = sorted(int(q) for q in df.loc[df["year"] == last, "quarter"].unique())
    w(f"- {last} covers quarters {', '.join(f'Q{q}' for q in qs)} only, and recent PSA "
      "figures are likely provisional.")
    w("- The cropland mask is a single 2020/21 WorldCover snapshot applied to every year.")
    w("- Rainfall is ERA5 reanalysis on a ~0.25° (~28 km) grid at one point per province, "
      "not gauge-corrected.")
    w("")

    w("## Next steps")
    w("")
    w("- A rice-paddy mask (an existing paddy map, or Sentinel-1/2 phenology-based "
      "mapping) so indices average over rice fields only; a separate corn mask.")
    w("- Sentinel-1 SAR flood/inundation detection for rice transplanting timing and "
      "planted area (cloud-independent).")
    w("- Re-evaluate with more Sentinel-2 years as they accumulate.")
    w("")

    w("## predictions.parquet columns")
    w("")
    w(_md(pd.DataFrame({"column": list(PREDICTION_COLUMNS),
                        "definition": list(PREDICTION_COLUMNS.values())})))
    w("")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    return out
