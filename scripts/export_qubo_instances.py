"""Export the 144 dispatch-window QUBOs of the paper as benchmark instances.

Each instance is built with the paper's model and settings from the window inputs of the
exact enumeration run (results/by_date/<date>, solver "exact"): load window, renewable
scenarios, prices, start SOC, battery mode, reserve target and the day-ahead SOC reference.
The reference solution is the schedule that run committed. Its energy and the ground-state
energy (by enumeration of all 2^20 assignments) are computed for the QUBO in the file.

Output: results/qubo_instances/
    index.csv                       one row per window with the reference values
    <date>/window_<hh>.json         QUBO and window data (hh = 01..24)

JSON fields
    n, offset                       number of binary variables (20) and constant term
    linear                          {"i": h_i}, diagonal terms
    quadratic                       [[i, j, J_ij], ...] with i < j
    variables                       names of the 20 variables, index order
    window                          load, scenario weights and renewables, prices (pu),
                                    start SOC, terminal SOC floor, battery mode
    reference                       committed schedule of the exact solver (cheapest feasible
                                    dispatch the encoding admits): assignment, energy and
                                    operating cost; ground-state energy and assignment of the
                                    QUBO; MILP cost of the same window

Energy of an assignment x in {0,1}^n:
    E(x) = offset + sum_i h_i x_i + sum_{i<j} J_ij x_i x_j

Usage:
    python scripts/export_qubo_instances.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from microgrid import config as C
from microgrid.optimization_milp import milp_daily
from microgrid.optimization_qubo import build_qubo
from microgrid.preprocessing import preprocess_date
from microgrid.scenario_generation import generate_scenarios, scenarios_to_tuples

OUT = C.RESULTS_DIR / "qubo_instances"
T = C.SCHEDULING_HORIZON
NAMES = [f"{v}_t{t}" for t in range(T) for v in ("g0", "g1", "g2", "c", "d")]


def stored(date: str, name: str, solver: str) -> pd.DataFrame:
    df = pd.read_csv(C.RESULTS_DIR / "by_date" / date / "tables" / name)
    return df[df.solver == solver]


def assignment(sched_hour: pd.DataFrame) -> np.ndarray:
    """Binary vector of one window from the stored 15-minute schedule."""
    x = []
    for _, r in sched_hour.sort_values("substep").iterrows():
        k = int(round(float(r.grid) * 7))
        x += [k & 1, (k >> 1) & 1, (k >> 2) & 1, int(r.charge), int(r.discharge)]
    return np.asarray(x, dtype=np.float64)


def energy(Q: dict, offset: float, x: np.ndarray) -> float:
    return float(offset + sum(v * x[i] * x[j] for (i, j), v in Q.items()))


def ground_state(Q: dict, n: int, offset: float, chunk: int = 1 << 16):
    """Lowest energy and its assignment, by enumeration of all 2^n assignments."""
    M = np.zeros((n, n))
    for (i, j), v in Q.items():
        M[i, j] += v
    best_e, best_x = np.inf, None
    for lo in range(0, 1 << n, chunk):
        idx = np.arange(lo, min(lo + chunk, 1 << n))
        X = ((idx[:, None] >> np.arange(n)) & 1).astype(np.float64)
        e = offset + np.einsum("ki,ij,kj->k", X, M, X)
        k = int(np.argmin(e))
        if e[k] < best_e:
            best_e, best_x = float(e[k]), X[k].astype(int)
    return best_e, best_x


def qubo_to_json(Q: dict, n: int, offset: float) -> dict:
    lin = {str(i): 0.0 for i in range(n)}
    quad: dict = {}
    for (i, j), v in Q.items():
        if i == j:
            lin[str(i)] += v
        else:
            a, b = min(i, j), max(i, j)
            quad[(a, b)] = quad.get((a, b), 0.0) + v
    return {"n": n, "offset": offset, "linear": lin,
            "quadratic": [[a, b, v] for (a, b), v in sorted(quad.items()) if v != 0.0]}


def main() -> None:
    mg = C.make_mg()
    rows = []
    for date in C.STUDY_DATES:
        pday = preprocess_date(date)
        sc = scenarios_to_tuples(generate_scenarios(pday))
        load_fc = np.asarray(pday.load_fc_n, float)[:C.N_STEPS]
        solar_fc = np.asarray(pday.solar_ec_n, float).mean(axis=0)[:C.N_STEPS]
        wind_fc = np.asarray(pday.wind_ec_n, float).mean(axis=0)[:C.N_STEPS]
        soc_ref = milp_daily(load_fc, solar_fc, wind_fc, mg, price_vec=C.PRICE_VEC, soc0=C.SOC_INIT)

        diag = stored(date, "df_diagnostics.csv", "exact").set_index("hour")
        sched = stored(date, "df_schedule.csv", "exact")
        cmp_ex = stored(date, "df_compare.csv", "exact").set_index("hour").cost
        cmp_ml = stored(date, "df_compare.csv", "milp").set_index("hour").cost
        out_dir = OUT / date
        out_dir.mkdir(parents=True, exist_ok=True)

        soc = C.SOC_INIT
        for hour in range(1, 25):
            t0, t1 = (hour - 1) * C.APPLY_STEPS, (hour - 1) * C.APPLY_STEPS + T
            d = diag.loc[hour]
            wl = pday.load_n[t0:t1]
            wsc = [(float(w), s[t0:t1], v[t0:t1]) for w, s, v in sc]
            price = C.PRICE_VEC[t0:t1]
            Q, n, off = build_qubo(wl, wsc, mg, soc0=soc, price_vec=price,
                                   soc_ref=soc_ref[t0:t1],
                                   soc_reserve_target=float(d.soc_target),
                                   battery_mode=str(d.battery_mode))
            sh = sched[sched.hour == hour]
            x = assignment(sh)
            e = energy(Q, off, x)
            e0, x0 = ground_state(Q, n, off)

            inst = qubo_to_json(Q, n, off)
            inst["variables"] = NAMES
            inst["window"] = {
                "date": date, "hour": hour, "steps": T, "dt_hours": C.DT,
                "load_pu": [float(v) for v in wl],
                "scenarios": [{"weight": w, "solar_pu": [float(v) for v in s],
                               "wind_pu": [float(v) for v in u]} for w, s, u in wsc],
                "price": [float(v) for v in price],
                "soc_start": soc, "soc_floor_end": float(d.min_soc_end),
                "battery_mode": str(d.battery_mode),
            }
            inst["reference"] = {
                "assignment": [int(v) for v in x],
                "energy": e,
                "operating_cost": float(cmp_ex[hour]),
                "ground_state_energy": e0,
                "ground_state_assignment": [int(v) for v in x0],
                "milp_cost": float(cmp_ml[hour]),
            }
            (out_dir / f"window_{hour:02d}.json").write_text(json.dumps(inst, indent=1))
            rows.append({"date": date, "hour": hour, "n": n, "quadratic_terms": len(inst["quadratic"]),
                         "offset": off, "reference_energy": e,
                         "ground_state_energy": e0,
                         "reference_cost": float(cmp_ex[hour]), "milp_cost": float(cmp_ml[hour]),
                         "file": f"{date}/window_{hour:02d}.json"})
            soc = float(sh.sort_values("substep").soc.iloc[-1])

    pd.DataFrame(rows).to_csv(OUT / "index.csv", index=False)
    print(f"exported {len(rows)} instances to {OUT.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
