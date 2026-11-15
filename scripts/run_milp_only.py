"""Recompute the MILP baseline across the study days without touching stored results.

Reproduces the rolling MILP exactly as the scheduler drives it -- same day-ahead SOC
reference, same battery policy, same per-window terminal-SOC floor -- so the total is
directly comparable to the committed ``milp`` rows in results/by_date. Nothing is
written to results/by_date, so existing multi-solver runs are preserved.

Usage:
    python scripts/run_milp_only.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from microgrid.config import (
    APPLY_STEPS, ETA_C, MILP_LAM_TRACK, N_STEPS, P_BATT, PRICE_VEC, RESULTS_DIR,
    SCHEDULING_HORIZON, SOC_CAP, SOC_INIT, SOC_MAX, SOC_MIN, STUDY_DATES, DT, make_mg,
)
from microgrid.optimization_milp import milp_daily, milp_schedule_window
from microgrid.preprocessing import preprocess_date
from microgrid.scenario_generation import generate_scenarios, scenarios_to_tuples

MAX_SOC_GAIN = P_BATT * ETA_C * APPLY_STEPS * DT / SOC_CAP


def battery_policy(hour: int, net_hourly: np.ndarray, low: float, high: float,
                   peak_hour: int, charge_target: float, preserve_target: float):
    """Per-window terminal-SOC floor and reserve target, mirroring the scheduler."""
    if hour < peak_hour and net_hourly[hour] <= low:
        return 0.55, charge_target
    if hour < peak_hour:
        return 0.45, preserve_target
    if net_hourly[hour] >= high:
        return 0.15, 0.20
    return 0.25, 0.40


def run_date(date: str, mg) -> dict:
    pday = preprocess_date(date)
    scen = scenarios_to_tuples(generate_scenarios(pday))

    load_fc = np.asarray(pday.load_fc_n, float)[:N_STEPS]
    solar_fc = np.asarray(pday.solar_ec_n, float).mean(axis=0)[:N_STEPS]
    wind_fc = np.asarray(pday.wind_ec_n, float).mean(axis=0)[:N_STEPS]
    soc_ref = milp_daily(load_fc, solar_fc, wind_fc, mg,
                         price_vec=PRICE_VEC, soc0=SOC_INIT)

    net_hourly = (load_fc - solar_fc - wind_fc).reshape(24, APPLY_STEPS).mean(axis=1)
    low, high = np.percentile(net_hourly, 30), np.percentile(net_hourly, 75)
    peak_hour = int(np.argmax(net_hourly))
    stress = net_hourly.max() / (net_hourly.mean() + 1e-9)
    high_stress = stress > 1.25 or pd.to_datetime(date).month in (6, 7, 8, 9)
    charge_target, preserve_target = (0.80, 0.70) if high_stress else (0.75, 0.65)

    soc, total, n_feasible = SOC_INIT, 0.0, 0

    for step in range(0, N_STEPS, APPLY_STEPS):
        t0, t1 = step, min(step + SCHEDULING_HORIZON, N_STEPS)
        hour = step // APPLY_STEPS
        load = pday.load_n[t0:t1]
        wsc = [(w, s[t0:t1], v[t0:t1]) for w, s, v in scen]
        price = PRICE_VEC[t0:t1]
        ref_win = soc_ref[t0:t1]

        min_soc_end, _ = battery_policy(hour, net_hourly, low, high, peak_hour,
                                        charge_target, preserve_target)
        if hour < peak_hour or net_hourly[hour] < high:
            min_soc_end = max(min_soc_end, float(ref_win[-1]) - 0.15)
        if hour >= peak_hour and net_hourly[hour] >= high:
            min_soc_end = max(SOC_MIN - 0.05, min_soc_end * 0.5)
        min_soc_end = min(min_soc_end, min(SOC_MAX, soc + MAX_SOC_GAIN) - 1e-6)

        res = milp_schedule_window(load, wsc, mg, soc, price,
                                   min_soc_end=min_soc_end,
                                   soc_ref_end=float(ref_win[-1]),
                                   lam_track=MILP_LAM_TRACK)
        if res is None:
            continue

        n_feasible += 1
        total += sum(
            DT * float(price[t]) * (
                mg.C_grid * res["g"][t]
                + mg.C_cycle * mg.P_B * (res["u_c"][t] + res["u_d"][t])
                + mg.C_shed * res["s"][t]
            )
            for t in range(APPLY_STEPS)
        )
        soc = float(np.clip(res["soc"][APPLY_STEPS - 1], mg.soc_min, mg.soc_max))

    return {"date": date, "cost": total, "n_feasible": n_feasible}


def stored_totals() -> pd.Series:
    rows = {}
    for d in STUDY_DATES:
        f = RESULTS_DIR / "by_date" / d / "tables" / "df_compare.csv"
        if f.exists():
            c = pd.read_csv(f)
            rows[d] = c[c.solver == "milp"].cost.sum()
    return pd.Series(rows)


def main() -> None:
    mg = make_mg()
    print(f"MILP_LAM_TRACK = {MILP_LAM_TRACK}  (terminal-only tracking)\n", flush=True)

    rows = []
    for date in STUDY_DATES:
        r = run_date(date, mg)
        rows.append(r)
        print(f"  {date}  cost={r['cost']:8.4f}  windows={r['n_feasible']}/24", flush=True)

    df = pd.DataFrame(rows).set_index("date")
    stored = stored_totals()

    print()
    if len(stored):
        cmp = pd.DataFrame({"recomputed": df.cost, "on_disk": stored})
        cmp["diff"] = cmp.recomputed - cmp.on_disk
        print(cmp.round(4).to_string())
        print()
        print(f"TOTAL recomputed = {df.cost.sum():.4f}")
        print(f"TOTAL on disk    = {stored.sum():.4f}")
        agree = abs(df.cost.sum() - stored.sum()) < 1e-3
        print(f"\n{'MATCH' if agree else 'MISMATCH'}: recomputed vs stored MILP baseline")
    else:
        print(f"TOTAL = {df.cost.sum():.4f}")


if __name__ == "__main__":
    main()
