from __future__ import annotations

import json
import logging
from pathlib import Path

from microgrid.config import STUDY_DATES, RESULTS_DIR, make_mg
from microgrid.preprocessing import preprocess_date, compute_dataset_characteristics
from microgrid.scenario_generation import generate_scenarios
from microgrid.scheduler import run_schedule
from microgrid.plotting import generate_date_figures, generate_combined_figures

logger = logging.getLogger(__name__)


def _save_results(result, chars: dict, variant: str | None = None) -> Path:
    folder_name = f"{result.date}_{variant}" if variant else result.date
    rd = RESULTS_DIR / "by_date" / folder_name
    tables = rd / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    result.df_compare.to_csv(tables / "df_compare.csv", index=False)
    result.df_schedule.to_csv(tables / "df_schedule.csv", index=False)
    if not result.df_diagnostics.empty:
        result.df_diagnostics.to_csv(tables / "df_diagnostics.csv", index=False)

    with open(tables / "metadata.json", "w") as f:
        json.dump({**result.metadata, **chars, "variant": variant or "simulator"}, f, indent=2)

    return rd


def run_single_date(
    date: str,
    force_preprocess: bool = False,
    enable_qaoa: bool = True,
    enable_sqa: bool = True,
    enable_exact: bool = False,
    bqm_baselines: list[str] | None = None,
    ibm_backend: str | None = None,
    ibm_shots: int = 4096,
    qaoa_p_layers: list | None = None,
) -> Path:
    backend_label = ibm_backend or "simulator"
    p_label = f"_p{''.join(str(p) for p in qaoa_p_layers)}" if qaoa_p_layers else ""
    logger.info("=" * 60)
    logger.info(f"  Date: {date}  QAOA={enable_qaoa}  SQA={enable_sqa}  "
                f"EXACT={enable_exact}  backend={backend_label}{p_label}")
    logger.info("=" * 60)

    logger.info("[1/4] Preprocessing")
    pday = preprocess_date(date, force=force_preprocess)
    logger.info(f"  load_n={pday.load_n.shape}  solar_ec_n={pday.solar_ec_n.shape}")

    logger.info("[2/4] Generating scenarios")
    scenarios = generate_scenarios(pday)

    logger.info("[3/4] Running rolling scheduling")
    mg = make_mg()
    result = run_schedule(
        pday, scenarios, mg,
        enable_qaoa=enable_qaoa, enable_sqa=enable_sqa,
        enable_exact=enable_exact, bqm_baselines=bqm_baselines,
        ibm_backend=ibm_backend, ibm_shots=ibm_shots,
        qaoa_p_layers=qaoa_p_layers,
    )

    df_cmp = result.df_compare
    for solver in df_cmp["solver"].unique():
        sub = df_cmp[df_cmp["solver"] == solver]
        logger.info(
            f"  {solver:12s}: {int(sub['is_feasible'].sum())}/{len(sub)} feasible  "
            f"total_cost={sub['cost'].sum():.4f}  mean_gap={sub['gap_pct'].mean():.1f}%"
        )

    logger.info("[4/4] Saving results and generating figures")
    chars = compute_dataset_characteristics(pday)
    base_variant = ibm_backend or None
    variant = f"{base_variant}{p_label}" if base_variant and p_label else base_variant
    rd = _save_results(result, chars, variant=variant)
    generate_date_figures(date, result.df_compare, result.df_schedule, None, mg=mg,
                          fig_dir=rd / "figures")

    logger.info(f"  Results -> {rd}")
    logger.info(f"  Time: {result.metadata.get('total_time_s', 0):.1f} s")
    return rd


def run_multiple_dates(
    dates: list[str],
    force_preprocess: bool = False,
    enable_qaoa: bool = True,
    enable_sqa: bool = True,
    enable_exact: bool = False,
    bqm_baselines: list[str] | None = None,
    skip_existing: bool = True,
    ibm_backend: str | None = None,
    ibm_shots: int = 4096,
) -> None:
    completed, failed = [], []
    variant = ibm_backend

    for i, date in enumerate(dates):
        logger.info(f"[{i + 1}/{len(dates)}] {date}")

        if skip_existing and not force_preprocess:
            folder = f"{date}_{variant}" if variant else date
            existing = RESULTS_DIR / "by_date" / folder / "tables" / "df_compare.csv"
            if existing.exists():
                logger.info(f"  Skipping {date} (results exist; use --force to re-run)")
                completed.append(date)
                continue

        try:
            run_single_date(
                date=date, force_preprocess=force_preprocess,
                enable_qaoa=enable_qaoa, enable_sqa=enable_sqa,
                enable_exact=enable_exact, bqm_baselines=bqm_baselines,
                ibm_backend=ibm_backend, ibm_shots=ibm_shots,
            )
            completed.append(date)
        except Exception as exc:
            logger.error(f"  FAILED for {date}: {exc}", exc_info=True)
            failed.append(date)

    logger.info(f"Completed: {len(completed)}/{len(dates)}  Failed: {len(failed)}")
    if failed:
        logger.warning(f"Failed dates: {failed}")

    if completed:
        try:
            generate_combined_figures(completed)
        except Exception as exc:
            logger.warning(f"Combined figures failed: {exc}")
