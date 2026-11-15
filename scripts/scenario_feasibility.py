"""Count dispatches that balance under the mean forecast and under every scenario.

With the quadratic power-balance penalty,

    sum_k w_k (A + b_k)^2 = (A + b_bar)^2 + Var(b),

where A holds the decision terms and b_k = R_k - L. Var(b) is a constant, so the
scenario set does not move the minimiser of the objective. It does change which
assignments balance. For every assignment of every dispatch window this script counts:

    expected   assignments that balance under the weighted-mean forecast
    scenario   assignments that balance under each scenario individually

Committed schedules are selected on expected-value feasibility for every solver, so
these counts describe the ensemble, not a constraint on the committed dispatch.

Output: results/scenario_feasibility/scenario_feasibility.csv
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
from microgrid.optimization_exact import run_exact_window
from microgrid.optimization_qubo import build_qubo
from microgrid.preprocessing import preprocess_date
from microgrid.scenario_generation import generate_scenarios, scenarios_to_tuples
from microgrid.utils import decode, soc_trajectory

OUT = RESULTS_DIR / "scenario_feasibility"


def run_date(date: str, mg) -> dict:
    """Exhaustive feasibility counts over all dispatch windows of one day."""
    pday = preprocess_date(date)
    scen = scenarios_to_tuples(generate_scenarios(pday))

    soc = SOC_INIT
    expected = scenario = 0

    for step in range(0, N_STEPS, APPLY_STEPS):
        t0, t1 = step, min(step + SCHEDULING_HORIZON, N_STEPS)
        T = t1 - t0
        load = pday.load_n[t0:t1]
        wsc = [(w, s[t0:t1], v[t0:t1]) for w, s, v in scen]
        price = PRICE_VEC[t0:t1]

        Q, n, off = build_qubo(load, wsc, mg, soc0=soc, price_vec=price)
        x, _, diag = run_exact_window(Q, n, off, T, load, wsc, mg, soc, price)

        expected += diag["n_expected_feasible"]
        scenario += diag["n_scenario_feasible"]

        # Advance the SOC state so later windows are evaluated from a realistic level.
        if x is not None:
            _, c, d = decode(x, T)
            soc = float(np.clip(
                soc_trajectory(c[:APPLY_STEPS], d[:APPLY_STEPS], mg, soc),
                mg.soc_min, mg.soc_max)[-1])

    ruled_out = 100.0 * (expected - scenario) / expected if expected else 0.0
    return {"date": date, "expected": expected, "scenario": scenario,
            "ruled_out_pct": ruled_out}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mg = make_mg()

    rows = []
    for date in STUDY_DATES:
        row = run_date(date, mg)
        rows.append(row)
        print(f"  {date}  mean={row['expected']:>9,}  all-scenarios={row['scenario']:>9,}"
              f"  ruled out {row['ruled_out_pct']:5.1f}%", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "scenario_feasibility.csv", index=False)

    tot_e, tot_s = int(df.expected.sum()), int(df.scenario.sum())
    print(f"\nTOTAL  mean={tot_e:,}  all-scenarios={tot_s:,}  "
          f"ruled out {100*(tot_e-tot_s)/tot_e:.1f}%")
    print(f"worst day: {df.loc[df.ruled_out_pct.idxmax(), 'date']} "
          f"({df.ruled_out_pct.max():.1f}%)")
    print(f"\nWrote {OUT}/")


if __name__ == "__main__":
    main()
