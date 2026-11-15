"""MILP baseline with a grid-aware terminal-SOC floor and no load-shedding slack.

The scheduler caps the day-ahead SOC floor at soc + P_B*eta_c*APPLY_STEPS*DT/SOC_CAP,
the most the battery can absorb in one window. Charging at P_B while serving a residual
demand r_t needs grid import r_t + P_B, so a substep can charge only when

    r_t + P_B <= P_G .

On 2018-07-23 this fails in every substep of hour 16: the floor (0.7480) is 0.118 above
the reachable SOC (0.6300), and the MILP with a shed slack meets it by shedding
0.044 pu of load. This script caps the floor at the grid-reachable SOC, removes the shed
slack, and reruns the rolling MILP. It reports, per date, the cost, the reference MILP
cost from results/by_date, and the number of windows where the cap binds.

Output: results/grid_aware_clamp.csv

Usage:
    python scripts/grid_aware_soc_clamp.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from microgrid.config import (
    APPLY_STEPS, ETA_C, MILP_LAM_TRACK, N_STEPS, P_BATT, P_GRID, PRICE_VEC, RESULTS_DIR,
    SCHEDULING_HORIZON, SOC_CAP, SOC_INIT, SOC_MAX, SOC_MIN, STUDY_DATES, DT, make_mg,
)
from microgrid.optimization_milp import milp_daily, milp_schedule_window
from microgrid.preprocessing import preprocess_date
from microgrid.scenario_generation import generate_scenarios, scenarios_to_tuples

BATT_GAIN = P_BATT * ETA_C * APPLY_STEPS * DT / SOC_CAP   # battery-rate clamp
STEP_GAIN = P_BATT * ETA_C * DT / SOC_CAP                 # per-substep SOC gain

# Reference MILP (with shed slack) per date, from results/by_date/<date>/tables/df_compare.csv
BASELINE = {"2018-03-03": 16.3422, "2018-04-13": 15.3257, "2018-04-14": 11.1622,
            "2018-06-27": 33.5136, "2018-07-23": 38.1232, "2018-10-26": 23.0723}


def policy(h, net, low, high, peak, ct, pt):
    """Mirrors scheduler._battery_policy: (floor, reserve_target, mode)."""
    if h < peak and net[h] <= low:
        return 0.55, ct, "charge"
    if h < peak:
        return 0.45, pt, "preserve"
    if net[h] >= high:
        return 0.15, 0.20, "discharge"
    return 0.25, 0.40, "recover"


def reachable_gain(resid: np.ndarray) -> float:
    """SOC gain actually attainable: only substeps with grid headroom can charge."""
    can_charge = (resid + P_BATT) <= P_GRID + 1e-12
    return float(can_charge[:APPLY_STEPS].sum()) * STEP_GAIN


def run_date(date, mg, grid_aware: bool):
    p = preprocess_date(date)
    sc = scenarios_to_tuples(generate_scenarios(p))

    lf = np.asarray(p.load_fc_n, float)[:N_STEPS]
    sf = np.asarray(p.solar_ec_n, float).mean(axis=0)[:N_STEPS]
    wf = np.asarray(p.wind_ec_n, float).mean(axis=0)[:N_STEPS]
    ref = milp_daily(lf, sf, wf, mg, price_vec=PRICE_VEC, soc0=SOC_INIT)

    net = (lf - sf - wf).reshape(24, APPLY_STEPS).mean(axis=1)
    low, high = np.percentile(net, 30), np.percentile(net, 75)
    peak = int(np.argmax(net))
    stress = net.max() / (net.mean() + 1e-9)
    hot = stress > 1.25 or pd.to_datetime(date).month in (6, 7, 8, 9)
    ct, pt = (0.80, 0.70) if hot else (0.75, 0.65)

    soc, total = SOC_INIT, 0.0
    n_ok = n_clamped = 0

    for step in range(0, N_STEPS, APPLY_STEPS):
        t0, t1 = step, min(step + SCHEDULING_HORIZON, N_STEPS)
        h = step // APPLY_STEPS
        wl = p.load_n[t0:t1]
        wsc = [(w, a[t0:t1], b[t0:t1]) for w, a, b in sc]
        pw, rw = PRICE_VEC[t0:t1], ref[t0:t1]

        floor, _, mode = policy(h, net, low, high, peak, ct, pt)
        if mode != "discharge":
            floor = max(floor, float(rw[-1]) - 0.15)
        else:
            floor = max(SOC_MIN - 0.05, floor * 0.5)

        cap = min(SOC_MAX, soc + BATT_GAIN) - 1e-6
        if grid_aware:
            mean_re = sum(w * (a + b) for w, a, b in wsc)
            resid = np.asarray(wl, float) - np.asarray(mean_re, float)
            cap_g = min(SOC_MAX, soc + reachable_gain(resid)) - 1e-6
            if min(floor, cap_g) < min(floor, cap) - 1e-9:
                n_clamped += 1
            cap = cap_g

        r = milp_schedule_window(wl, wsc, mg, soc, pw, min_soc_end=min(floor, cap),
                                 soc_ref_end=float(rw[-1]), lam_track=MILP_LAM_TRACK)
        if r is None:
            continue
        n_ok += 1
        total += sum(DT * pw[t] * (mg.C_grid * r["g"][t]
                     + mg.C_cycle * mg.P_B * (r["u_c"][t] + r["u_d"][t])
                     + mg.C_shed * r["s"][t]) for t in range(APPLY_STEPS))
        soc = float(np.clip(r["soc"][APPLY_STEPS - 1], mg.soc_min, mg.soc_max))

    return {"date": date, "cost": total, "windows": n_ok, "clamped": n_clamped}


def main() -> None:
    mg = make_mg()
    mg.shed_frac = 0.00

    print("MILP, no shed slack, grid-aware terminal-SOC clamp\n")
    rows = []
    for d in STUDY_DATES:
        r = run_date(d, mg, grid_aware=True)
        r["baseline"] = BASELINE[d]
        r["diff"] = r["cost"] - r["baseline"]
        rows.append(r)
        print(f"  {d}  cost={r['cost']:8.4f}  baseline={r['baseline']:8.4f}  "
              f"diff={r['diff']:+8.4f}  windows={r['windows']}/24  "
              f"clamp bound in {r['clamped']}")

    df = pd.DataFrame(rows)
    print(f"\nTOTAL      = {df.cost.sum():.4f}")
    print(f"baseline   = {df.baseline.sum():.4f}")
    print(f"difference = {df.cost.sum() - df.baseline.sum():+.4f} "
          f"({100*(df.cost.sum()-df.baseline.sum())/df.baseline.sum():+.3f}%)")
    print(f"windows solved = {int(df.windows.sum())}/144")
    print(f"windows where the grid-aware clamp binds = {int(df.clamped.sum())}/144")

    out = RESULTS_DIR / "grid_aware_clamp.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
