"""Sensitivity of the dispatch cost to the number of renewable scenarios.

Scenarios enter ``build_qubo`` as weighted terms of the power-balance penalty, so the
QUBO keeps n = BITS_PER_STEP * T binary variables for any scenario count.

For k in {1, 3, 5, 10, 51}, every dispatch window of every study day is solved with
the exact enumeration solver, so differences between k values come from the scenario
set only. k=1 is a single medoid (point forecast); k=51 uses every ECMWF member.

Outputs to results/scenario_sensitivity/:
    scenario_k_sweep.csv    per-date, per-k totals (cost, feasible windows)
    block_scaling.csv       per-unit vs. per-block standardized clustering at k=3
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from microgrid.config import (
    APPLY_STEPS, N_STEPS, PRICE_VEC, RESULTS_DIR, SCHEDULING_HORIZON,
    SOC_INIT, STUDY_DATES, make_mg,
)
from microgrid.preprocessing import preprocess_date
from microgrid.scenario_generation import generate_scenarios, scenarios_to_tuples
from microgrid.optimization_qubo import build_qubo
from microgrid.optimization_exact import run_exact_window
from microgrid.utils import decode, soc_trajectory

OUT = RESULTS_DIR / "scenario_sensitivity"
K_VALUES = [1, 3, 5, 10, 51]


def all_members(pday) -> list:
    """Every ensemble member as its own equally weighted scenario (the k=51 reference)."""
    n = pday.solar_ec_n.shape[0]
    return [(1.0 / n, pday.solar_ec_n[i], pday.wind_ec_n[i]) for i in range(n)]


def roll_day(pday, scenario_tuples: list, mg) -> dict:
    """Solve one day's rolling windows exactly under a fixed scenario set."""
    soc = SOC_INIT
    total, n_feas, n_win = 0.0, 0, 0

    for step in range(0, N_STEPS, APPLY_STEPS):
        t0, t1 = step, min(step + SCHEDULING_HORIZON, N_STEPS)
        T = t1 - t0
        wl = pday.load_n[t0:t1]
        wsc = [(w, s[t0:t1], v[t0:t1]) for w, s, v in scenario_tuples]
        pw = PRICE_VEC[t0:t1]

        Q, n, off = build_qubo(wl, wsc, mg, soc0=soc, price_vec=pw)
        x, cost, _ = run_exact_window(Q, n, off, T, wl, wsc, mg, soc, pw)

        n_win += 1
        if x is None:
            continue
        n_feas += 1
        total += cost
        _, c, d = decode(x, T)
        soc = float(np.clip(
            soc_trajectory(c[:APPLY_STEPS], d[:APPLY_STEPS], mg, soc),
            mg.soc_min, mg.soc_max)[-1])

    return {"cost": total, "n_feasible": n_feas, "n_windows": n_win}


def k_sweep() -> pd.DataFrame:
    mg = make_mg()
    rows = []
    for date in STUDY_DATES:
        pday = preprocess_date(date)
        for k in K_VALUES:
            tuples = (all_members(pday) if k >= pday.solar_ec_n.shape[0]
                      else scenarios_to_tuples(generate_scenarios(pday, n_scenarios=k)))
            res = roll_day(pday, tuples, mg)
            rows.append({"date": date, "k": k, "n_terms": len(tuples), **res})
            print(f"  {date}  k={k:2d}  cost={res['cost']:8.4f}  "
                  f"feasible={res['n_feasible']}/{res['n_windows']}")
    return pd.DataFrame(rows)


def block_scaling() -> pd.DataFrame:
    """Default (per-unit) vs. per-block standardized clustering at k=3."""
    mg = make_mg()
    rows = []
    for date in STUDY_DATES:
        pday = preprocess_date(date)
        for label, std in (("per-unit", False), ("standardized", True)):
            sc = generate_scenarios(pday, n_scenarios=3, standardize_blocks=std)
            res = roll_day(pday, scenarios_to_tuples(sc), mg)
            rows.append({
                "date": date, "clustering": label, "cost": res["cost"],
                "weights": ",".join(f"{s.weight:.3f}" for s in sc),
                "n_feasible": res["n_feasible"],
            })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    print("Scenario-count sweep (exact solver):")
    sweep = k_sweep()
    sweep.to_csv(OUT / "scenario_k_sweep.csv", index=False)

    print("\nBlock-scaling sensitivity at k=3:")
    blocks = block_scaling()
    blocks.to_csv(OUT / "block_scaling.csv", index=False)

    tot = sweep.groupby("k")[["cost"]].sum()
    ref = float(tot.loc[51, "cost"])
    print(f"\n{'k':>3s} {'cost':>10s} {'d_51 %':>8s}")
    for k in K_VALUES:
        c = float(tot.loc[k, "cost"])
        print(f"{k:3d} {c:10.4f} {100*(c-ref)/ref:+8.3f}")

    piv = blocks.pivot(index="date", columns="clustering", values="cost")
    piv["diff_%"] = 100 * (piv["standardized"] - piv["per-unit"]) / piv["per-unit"]
    print("\nBlock scaling (k=3):")
    print(piv.round(4).to_string())
    print(f"\nWrote {OUT}/")


if __name__ == "__main__":
    main()
