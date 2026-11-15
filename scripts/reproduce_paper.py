"""Rebuild every table, number and figure of the paper from the stored results.

No solver is called and no hardware is used. The script reads the per-window result
files under results/ and writes:

    results/paper/table2_daily_cost.csv     Table II, per date and total
    results/paper/solver_totals.csv         totals, gaps to MILP and to the exact QUBO
    results/paper/runtime.csv               mean wall-clock time per dispatch window
    results/paper/feasibility_repair.csv    raw feasibility and repair counts
    results/paper/scenarios.csv             scenario-count sweep and scenario feasibility
    results/paper/soc_2018-07-23.csv        hourly SOC on the peak-load day
    results/paper/paper_numbers.csv         every number in the paper, with its source
    results/paper/dwave_problem_ids.csv     D-Wave problem ID and QPU access time per window
    results/paper/study_days.csv            load and renewable profile of the six days
    results/paper/by_date/<date>/           per-date summary, hourly cost, SOC, grid import,
                                            15-minute schedule
    figures/fig2_daily_cost.png             Fig. 2
    figures/fig3_soc_2018-07-23.png         Fig. 3
    figures/by_date/<date>/                 SOC, grid import and hourly cost for each date

MILP reference. 137.54 is the rolling MILP in results/by_date. 137.49 uses the
grid-aware MILP of scripts/grid_aware_soc_clamp.py (results/grid_aware_clamp.csv) for
2018-07-23. The MILP column of Table II is the grid-aware MILP; the Table II
percentages are relative to the rolling MILP of each date. Gaps against both totals
are written to solver_totals.csv.

Usage:
    python scripts/reproduce_paper.py
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from microgrid import config as C
from microgrid.preprocessing import (compute_dataset_characteristics, compute_global_scale,
                                     preprocess_date)

R = C.RESULTS_DIR
OUT = R / "paper"
FIG = REPO_ROOT / "figures"
DATES = list(C.STUDY_DATES)
PEAK_DAY = "2018-07-23"
SEASON = {"2018-03-03": "Winter", "2018-04-13": "Spring", "2018-04-14": "Spring",
          "2018-06-27": "Summer", "2018-07-23": "Summer", "2018-10-26": "Fall"}


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def _table(folder: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(folder / "tables" / name)


def _runs(pattern: str) -> list[Path]:
    return sorted(p for p in R.glob(pattern) if (p / "tables").is_dir())


def compare_rows(folder_fmt: str, solver_match) -> pd.DataFrame:
    """Per-window rows of one solver, over the six study dates."""
    frames = []
    for d in DATES:
        f = R / folder_fmt.format(date=d) / "tables" / "df_compare.csv"
        df = pd.read_csv(f)
        df = df[df.solver.astype(str).map(solver_match)].copy()
        df["date"] = d
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_all() -> dict[str, pd.DataFrame]:
    base = compare_rows("by_date/{date}", lambda s: True)
    out = {s: g.copy() for s, g in base.groupby("solver")}
    out["qaoa_p2"] = compare_rows("by_date/{date}_ibm_kingston_p2", lambda s: s == "qaoa_p2")
    out["qaoa_p3"] = compare_rows("by_date/{date}_ibm_kingston_p3", lambda s: s == "qaoa_p3")
    out["dwave_qpu"] = compare_rows("dwave_qa/by_date/{date}", lambda s: s == "dwave_qa")
    out["stride"] = compare_rows("stride_native/by_date/{date}", lambda s: s == "stride_native")
    return out


def per_date(df: pd.DataFrame) -> pd.Series:
    return df.groupby("date").cost.sum().reindex(DATES)


def milp_grid_aware() -> pd.Series:
    """MILP column of Table II: stored MILP, with the grid-aware rerun where it differs."""
    g = pd.read_csv(R / "grid_aware_clamp.csv").set_index("date").cost.reindex(DATES)
    return g


def pct(a: float, b: float) -> float:
    return 100.0 * (a - b) / b


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

def table2(runs) -> pd.DataFrame:
    ref = per_date(runs["milp"])
    shown = milp_grid_aware()
    cols = {"exact": "exact_qubo", "qaoa_p2": "ibm_qaoa_p2_p3",
            "dwave_qpu": "dwave_qpu", "stride": "dwave_stride"}
    rows = []
    for d in DATES + ["Total"]:
        r_ref = ref.sum() if d == "Total" else ref[d]
        row = {"date": d,
               "milp_table_column": shown.sum() if d == "Total" else shown[d],
               "milp_reference": r_ref}
        for key, name in cols.items():
            s = per_date(runs[key])
            v = s.sum() if d == "Total" else s[d]
            row[f"{name}_cost"] = v
            row[f"{name}_delta_pct"] = pct(v, r_ref)
        rows.append(row)
    return pd.DataFrame(rows)


def solver_totals(runs) -> pd.DataFrame:
    ref = runs["milp"].cost.sum()
    ref_ga = milp_grid_aware().sum()
    exact = runs["exact"].cost.sum()
    order = [("milp", "MILP (CBC) rolling horizon"),
             ("exact", "Exact QUBO optimum (enumeration)"),
             ("qaoa_p2", "IBM Kingston QAOA p=2"),
             ("qaoa_p3", "IBM Kingston QAOA p=3"),
             ("dwave_qpu", "D-Wave Advantage2_system1 QPU"),
             ("stride", "D-Wave Stride (hybrid nonlinear)"),
             ("sqa", "Simulated quantum annealing (same QUBO)"),
             ("sa", "Simulated annealing (same QUBO)"),
             ("tabu", "Tabu search (same QUBO)"),
             ("greedy", "Steepest descent (same QUBO)"),
             ("random", "Random sampling control (same QUBO)")]
    rows = [{"solver": "milp_grid_aware", "description": "MILP with grid-aware SOC floor on 2018-07-23",
             "total_cost": ref_ga, "windows": 144, "feasible_windows": 144,
             "gap_vs_milp_137_54_pct": pct(ref_ga, ref), "gap_vs_milp_137_49_pct": 0.0,
             "gap_vs_exact_pct": pct(ref_ga, exact)}]
    for key, desc in order:
        df = runs[key]
        v = df.cost.sum()
        rows.append({"solver": key, "description": desc, "total_cost": v,
                     "windows": len(df), "feasible_windows": int(df.is_feasible.sum()),
                     "gap_vs_milp_137_54_pct": pct(v, ref),
                     "gap_vs_milp_137_49_pct": pct(v, ref_ga),
                     "gap_vs_exact_pct": pct(v, exact)})
    return pd.DataFrame(rows)


def dwave_timing() -> pd.DataFrame:
    frames = []
    for d in DATES:
        df = _table(R / "dwave_qa" / "by_date" / d, "df_diagnostics.csv")
        df = df[df.solver == "dwave_qa"].copy()
        t = df.timing.map(ast.literal_eval)
        df["qpu_access_s"] = t.map(lambda x: x["qpu_access_time"] / 1e6)
        df["problem_id"] = t.map(lambda x: x.get("problem_id"))
        df["date"] = d
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def runtime(runs) -> pd.DataFrame:
    """Mean wall-clock time per dispatch window (Section IV-C)."""
    t = pd.read_csv(R / "runtime" / "runtime_by_date.csv", index_col=0)
    src = "results/runtime/runtime_by_date.csv"
    rows = [
        ("MILP (CBC)", runs["milp"].runtime_s.mean(), "results/by_date"),
        ("IBM Kingston QAOA p=2", runs["qaoa_p2"].runtime_s.mean(), "results/by_date/*_ibm_kingston_p2"),
        ("IBM Kingston QAOA p=3", runs["qaoa_p3"].runtime_s.mean(), "results/by_date/*_ibm_kingston_p3"),
        ("D-Wave Advantage_system6.4", float(t.loc["MEAN"].iloc[-1]), src),
    ]
    return pd.DataFrame(rows, columns=["solver", "mean_seconds_per_window", "source"])


def _diag(pattern: str, solver_prefix: str) -> pd.DataFrame:
    frames = []
    for run in _runs(pattern):
        df = _table(run, "df_diagnostics.csv")
        df = df[df.solver.astype(str).str.startswith(solver_prefix)].copy()
        df["run"] = run.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _flag(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().eq("true")


def feasibility_repair(runs) -> pd.DataFrame:
    ibm_k2 = _diag("by_date/2018-??-??_ibm_kingston_p2", "qaoa")
    dw = _diag("dwave_qa/by_date/2018-??-??", "dwave_qa")
    rows = []
    for name, d in [("IBM Kingston QAOA p=2", ibm_k2),
                    ("D-Wave Advantage2_system1", dw)]:
        rep = _flag(d.repair_used)
        ham = (d.loc[rep, "hamming_raw_to_repaired"].mean()
               if "hamming_raw_to_repaired" in d else np.nan)
        rows.append({"platform": name, "windows": len(d),
                     "raw_feasible": int(_flag(d.solver_raw_feasible).sum()),
                     "repair_used": int(rep.sum()),
                     "mean_bits_changed_when_repaired": ham,
                     "fallback_used": int(_flag(d.fallback_used).sum()),
                     "final_feasible": int(_flag(d.is_feasible).sum())})
    rnd = runs["random"]
    rows.append({"platform": "Random sampling control (same repair)", "windows": len(rnd),
                 "raw_feasible": np.nan, "repair_used": np.nan,
                 "mean_bits_changed_when_repaired": np.nan, "fallback_used": np.nan,
                 "final_feasible": int(rnd.is_feasible.sum())})
    return pd.DataFrame(rows)


def scenarios() -> tuple[pd.DataFrame, dict]:
    k = pd.read_csv(R / "scenario_sensitivity" / "scenario_k_sweep.csv")
    tot = k.groupby("k").agg(total_cost=("cost", "sum"), feasible=("n_feasible", "sum"),
                             windows=("n_windows", "sum")).reset_index()
    ref = float(tot.loc[tot.k == 51, "total_cost"].iloc[0])
    tot["delta_vs_k51_pct"] = 100 * (tot.total_cost - ref) / ref
    spread = 100 * (tot.total_cost.max() - tot.total_cost.min()) / ref
    sf = pd.read_csv(R / "scenario_feasibility" / "scenario_feasibility.csv")
    ruled = 100 * (1 - sf.scenario.sum() / sf.expected.sum())
    tot["note"] = ""
    extra = pd.DataFrame([
        {"k": "all", "total_cost": np.nan, "feasible": np.nan, "windows": np.nan,
         "delta_vs_k51_pct": spread, "note": "spread of total cost over k in {1,3,5,10,51}, % of k=51"},
        {"k": "all", "total_cost": np.nan, "feasible": int(sf.scenario.sum()),
         "windows": int(sf.expected.sum()), "delta_vs_k51_pct": ruled,
         "note": "assignments balancing under every scenario vs under the mean; % ruled out"},
    ])
    return pd.concat([tot, extra], ignore_index=True), {"spread": spread, "ruled_out": ruled}


def hourly_soc(folder: Path, solver: str) -> pd.Series:
    df = _table(folder, "df_schedule.csv")
    df = df[df.solver == solver].sort_values(["hour", "substep"])
    s = df.groupby("hour").soc.last()
    return pd.concat([pd.Series({0: C.SOC_INIT}), s])


# Result folder and solver label of each method in the paper, per date
METHODS = {
    "milp": ("by_date/{d}", "milp"),
    "exact_qubo": ("by_date/{d}", "exact"),
    "ibm_qaoa_p2": ("by_date/{d}_ibm_kingston_p2", "qaoa_p2"),
    "ibm_qaoa_p3": ("by_date/{d}_ibm_kingston_p3", "qaoa_p3"),
    "dwave_qpu": ("dwave_qa/by_date/{d}", "dwave_qa"),
    "dwave_stride": ("stride_native/by_date/{d}", "stride_native"),
}
SAMPLERS = {"sqa": "SQA", "sa": "Simulated annealing", "tabu": "Tabu search",
            "greedy": "Steepest descent", "random": "Random sampling control"}


def soc_day(date: str) -> pd.DataFrame:
    """End-of-hour SOC of every method on one date (hour 0 = initial SOC)."""
    df = pd.DataFrame({k: hourly_soc(R / f.format(d=date), s) for k, (f, s) in METHODS.items()})
    df.index.name = "hour"
    return df


def soc_peak_day() -> pd.DataFrame:
    df = soc_day(PEAK_DAY).drop(columns="ibm_qaoa_p3")
    return df.rename(columns={"ibm_qaoa_p2": "ibm_qaoa_p2_p3"})


def schedule_day(date: str) -> pd.DataFrame:
    """15-minute schedule of every method: grid import (fraction of P_G), battery actions, SOC."""
    frames = []
    for key, (f, s) in METHODS.items():
        df = _table(R / f.format(d=date), "df_schedule.csv")
        df = df[df.solver == s].sort_values(["hour", "substep"])
        df = df[["hour", "substep", "global_step", "grid", "charge", "discharge", "soc"]].copy()
        df.insert(0, "method", key)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def hourly_cost_day(date: str) -> pd.DataFrame:
    cols = {}
    for key, (f, s) in METHODS.items():
        df = _table(R / f.format(d=date), "df_compare.csv")
        cols[key] = df[df.solver == s].set_index("hour").cost
    df = pd.DataFrame(cols)
    df.index.name = "hour"
    return df


def summary_day(date: str) -> pd.DataFrame:
    rows = []
    base = _table(R / "by_date" / date, "df_compare.csv")
    ref = base[base.solver == "milp"].cost.sum()
    ga = float(milp_grid_aware()[date])
    rows.append({"method": "milp_grid_aware", "cost": ga, "gap_vs_milp_pct": pct(ga, ref),
                 "feasible_windows": 24, "windows": 24})
    entries = [(k, R / f.format(d=date), s) for k, (f, s) in METHODS.items()]
    entries += [(k, R / "by_date" / date, k) for k in SAMPLERS]
    for key, folder, s in entries:
        df = _table(folder, "df_compare.csv")
        df = df[df.solver == s]
        rows.append({"method": key, "cost": df.cost.sum(), "gap_vs_milp_pct": pct(df.cost.sum(), ref),
                     "feasible_windows": int(df.is_feasible.sum()), "windows": len(df)})
    return pd.DataFrame(rows)


def study_days() -> pd.DataFrame:
    keys = ["peak_load_mw", "mean_load_mw", "load_range_mw",
            "net_load_min_mw", "net_load_max_mw", "net_load_mean_mw", "solar_penetration",
            "wind_penetration", "renewable_penetration", "renewable_variability", "peak_hour",
            "min_net_load_hour"]
    rows = []
    for d in DATES:
        m = compute_dataset_characteristics(preprocess_date(d))
        rows.append({"date": d, "season": SEASON[d], **{k: m.get(k) for k in keys}})
    return pd.DataFrame(rows).round(4)


# ---------------------------------------------------------------------------
# paper number check
# ---------------------------------------------------------------------------

def paper_numbers(t2, tot, rt, rep, sc, soc) -> pd.DataFrame:
    T = tot.set_index("solver")
    g = lambda s, c: float(T.loc[s, c])
    rt = rt.set_index("solver").mean_seconds_per_window
    rep = rep.set_index("platform")
    dwr = rep.loc["D-Wave Advantage2_system1"]
    k2 = rep.loc["IBM Kingston QAOA p=2"]
    rnd = rep.loc["Random sampling control (same repair)"]
    meta = ["sqa", "sa", "tabu", "greedy"]
    meta_gap = [g(s, "gap_vs_exact_pct") for s in meta]
    finals = soc.iloc[-1]
    prices = sorted(set(np.round(C.PRICE_VEC, 3)))

    rows = []

    def add(section, quantity, paper, value, fmt, source, note=""):
        if isinstance(value, str):
            text = value
        else:
            text = fmt.format(value)
        status = "match" if text == paper else "differs"
        rows.append({"section": section, "quantity": quantity, "paper_value": paper,
                     "repository_value": text, "status": status, "source": source,
                     "note": note})

    # Abstract (MILP basis 137.49)
    add("Abstract", "IBM QAOA gap to MILP (%)", "13.8", g("qaoa_p2", "gap_vs_milp_137_49_pct"), "{:.1f}",
        "results/paper/solver_totals.csv", "basis: MILP 137.49")
    add("Abstract", "D-Wave Advantage2 gap to MILP (%)", "17.9", g("dwave_qpu", "gap_vs_milp_137_49_pct"), "{:.1f}",
        "results/paper/solver_totals.csv", "basis: MILP 137.49")
    add("Abstract", "D-Wave Stride gap to MILP (%)", "3.8", g("stride", "gap_vs_milp_137_49_pct"), "{:.1f}",
        "results/paper/solver_totals.csv", "basis: MILP 137.49")
    add("Abstract", "Encoding share of the gap (%)", "13.73", g("exact", "gap_vs_milp_137_49_pct"), "{:.2f}",
        "results/paper/solver_totals.csv", "exact QUBO optimum vs MILP 137.49")
    add("Abstract", "IBM QAOA above exact QUBO optimum (%)", "0.05", g("qaoa_p2", "gap_vs_exact_pct"), "{:.2f}",
        "results/paper/solver_totals.csv")

    # Section III
    add("III-B", "Cost change over k in {1,3,5,10,51} (%)", "0.09", sc["spread"], "{:.2f}",
        "results/paper/scenarios.csv")
    add("III-B", "Fewer dispatches when balance is required in every scenario (%)", "18.3",
        sc["ruled_out"], "{:.1f}", "results/paper/scenarios.csv")
    add("III-D", "Hamming anchor lambda_H (IBM Kingston runs)", "0.10",
        C.IBM_RUN_REPAIR["LAM_REPAIR_HAMMING"], "{:.2f}", "microgrid/config.py (IBM_RUN_REPAIR)")
    add("III-D", "QAOA windows needing no repair beyond stage S1", "141 of 144",
        f"{int(k2.windows - k2.repair_used)} of {int(k2.windows)}", "",
        "results/paper/feasibility_repair.csv", "IBM Kingston p=2")
    add("III-D", "Windows with feasible dispatch, no MILP fallback", "144",
        str(int(k2.final_feasible)), "", "results/paper/feasibility_repair.csv", "IBM Kingston p=2")

    # QUBO and model parameters (Sections III and IV)
    for q, p, v, src in [
        ("Binary variables per window", "20", C.BITS_PER_STEP * C.SCHEDULING_HORIZON, "BITS_PER_STEP*T"),
        ("lambda_b", "150", C.LAM_BALANCE, "LAM_BALANCE"), ("lambda_m", "120", C.LAM_MUTEX, "LAM_MUTEX"),
        ("lambda_s", "250", C.LAM_SOC, "LAM_SOC"), ("lambda_v", "0.5", C.LAM_RESERVE, "LAM_RESERVE"),
        ("lambda_k", "6", C.LAM_TRACK, "LAM_TRACK"),
        ("COBYLA max iterations", "60", C.QAOA_MAX_ITER, "QAOA_MAX_ITER"),
        ("COBYLA restarts", "8", C.QAOA_RESTARTS, "QAOA_RESTARTS"),
        ("Scenarios (k-means medoids)", "3", C.N_SCENARIOS, "N_SCENARIOS"),
        ("P_G (pu)", "1.0", C.P_GRID, "P_GRID"), ("P_B (pu)", "0.15", C.P_BATT, "P_BATT"),
        ("SOC_cap (pu h)", "0.60", C.SOC_CAP, "SOC_CAP"), ("eta_c = eta_d", "0.95", C.ETA_C, "ETA_C"),
        ("SOC_min", "0.10", C.SOC_MIN, "SOC_MIN"), ("SOC_max", "0.90", C.SOC_MAX, "SOC_MAX"),
        ("SOC_0", "0.50", C.SOC_INIT, "SOC_INIT"), ("C_grid", "2.0", C.C_GRID, "C_GRID"),
        ("C_cycle", "0.018", C.C_CYCLE, "C_CYCLE"), ("C_shed", "8.0", C.C_SHED, "C_SHED"),
        ("CBC time limit per window (s)", "30", C.MILP_TIME_LIMIT, "MILP_TIME_LIMIT"),
        ("CBC relative gap", "1e-06", C.MILP_GAP_REL, "MILP_GAP_REL"),
    ]:
        fmt = "{:g}" if p in ("20", "150", "120", "250", "6", "60", "8", "3", "30") else (
            "{:.0e}" if p == "1e-06" else "{:.%df}" % (len(p.split(".")[1]) if "." in p else 0))
        if p == "1e-06":
            text = f"{v:.0e}".replace("e-6", "e-06")
            add("III/IV", q, p, text, "", f"microgrid/config.py ({src})")
        else:
            add("III/IV", q, p, v, fmt, f"microgrid/config.py ({src})")
    add("IV", "TOU price levels", "0.7, 1.0, 1.4, 1.6", ", ".join(f"{x:.1f}" for x in prices), "",
        "microgrid/config.py (PRICE_VEC)")
    scale = compute_global_scale()
    add("IV", "P_scale (MW)", "73,262", f"{scale:,.0f}", "", "data/ (microgrid/preprocessing.py)",
        "maximum hourly mean load of 2018")
    add("IV-D", "Grid level as share of capacity (%)", "14", 100 * C.GRID_COEFFS[0], "{:.0f}",
        "microgrid/config.py (GRID_COEFFS)", "one level = P_G/7")

    # Table I
    add("Table I", "IBM hardware windows (p=2 + p=3)", "288",
        str(len(runs_cache["qaoa_p2"]) + len(runs_cache["qaoa_p3"])), "",
        "results/by_date/*_ibm_kingston_p2, *_p3")
    add("Table I", "Logical variables per window", "20", C.BITS_PER_STEP * C.SCHEDULING_HORIZON, "{:d}",
        "microgrid/config.py (BITS_PER_STEP*T)")
    add("Table I", "D-Wave hardware windows", "144", str(len(runs_cache["dwave_qpu"])), "", "results/dwave_qa")
    ibm, dw6 = C.IBM_HARDWARE, C.DWAVE_HARDWARE["runtime_study"]
    src_hw = "microgrid/config.py (IBM_HARDWARE, DWAVE_HARDWARE)"
    add("Table I", "IBM processor", "Heron r2", ibm["processor"], "", src_hw)
    add("Table I", "IBM topology", "Heavy-hex", ibm["topology"].capitalize(), "", src_hw)
    add("Table I", "IBM working qubits", "156", ibm["qubits"], "{:d}", src_hw)
    add("Table I", "IBM median T1 (us)", "173", ibm["median_t1_us"], "{:d}", src_hw)
    add("Table I", "IBM CZ error", "2.1e-03", ibm["cz_error"], "{:.1e}", src_hw)
    add("Table I", "IBM readout error", "1.5e-02", ibm["readout_error"], "{:.1e}", src_hw)
    add("Table I", "IBM shots per window", "16,384", ibm["shots_per_window"], "{:,d}", src_hw)
    add("Table I", "IBM transpile optimization level", "3", ibm["transpile_optimization_level"], "{:d}", src_hw)
    add("Table I", "D-Wave processor (runtime study)", "Advantage_system6.4", dw6["solver"], "", src_hw)
    add("Table I", "D-Wave topology (runtime study)", "Pegasus", dw6["topology"], "", src_hw)
    add("Table I", "D-Wave working qubits (runtime study)", "5,627", dw6["qubits"], "{:,d}", src_hw)
    add("Table I", "D-Wave reads per window (runtime study)", "4,000", dw6["num_reads"], "{:,d}", src_hw)
    add("Table I", "D-Wave anneal time, us (runtime study)", "100", dw6["annealing_time_us"], "{:d}", src_hw)
    add("Table I", "D-Wave embedding", "minorminer", C.DWAVE_HARDWARE["embedding"], "", src_hw)
    add("Table I", "D-Wave chain strength", "uniform_torque_compensation",
        C.DWAVE_HARDWARE["chain_strength"], "", src_hw)

    # Section IV-A and Table II
    add("IV-A", "MILP total cost", "137.54", g("milp", "total_cost"), "{:.2f}", "results/paper/solver_totals.csv")
    add("IV-A", "IBM Kingston total cost", "156.43", g("qaoa_p2", "total_cost"), "{:.2f}", "results/paper/solver_totals.csv")
    add("IV-A", "IBM Kingston gap to MILP (%)", "13.7", g("qaoa_p2", "gap_vs_milp_137_54_pct"), "{:.1f}",
        "results/paper/solver_totals.csv", "basis: MILP 137.54")
    add("IV-A", "D-Wave QPU total cost", "162.04", g("dwave_qpu", "total_cost"), "{:.2f}", "results/paper/solver_totals.csv")
    add("IV-A", "D-Wave QPU gap to MILP (%)", "17.8", g("dwave_qpu", "gap_vs_milp_137_54_pct"), "{:.1f}",
        "results/paper/solver_totals.csv", "basis: MILP 137.54")
    add("IV-A", "D-Wave Stride total cost", "142.69", g("stride", "total_cost"), "{:.2f}", "results/paper/solver_totals.csv")
    add("IV-A", "D-Wave Stride gap to MILP (%)", "3.7", g("stride", "gap_vs_milp_137_54_pct"), "{:.1f}",
        "results/paper/solver_totals.csv", "basis: MILP 137.54")
    t = t2.set_index("date")
    add("IV-A", "Largest gap, 2018-04-14 (%)", "34", t.loc["2018-04-14", "exact_qubo_delta_pct"], "{:.0f}",
        "results/paper/table2_daily_cost.csv")
    paper_t2 = {
        "2018-03-03": ("16.34", "19.58", "+19.80", "19.58", "+19.80", "20.73", "+26.87", "17.55", "+7.40"),
        "2018-04-13": ("15.33", "18.73", "+22.21", "18.73", "+22.21", "19.56", "+27.63", "16.20", "+5.68"),
        "2018-04-14": ("11.16", "14.96", "+34.01", "14.96", "+34.01", "15.56", "+39.44", "12.25", "+9.78"),
        "2018-06-27": ("33.51", "36.19", "+7.98", "36.19", "+7.98", "37.68", "+12.44", "34.52", "+3.00"),
        "2018-07-23": ("38.07", "40.37", "+5.90", "40.44", "+6.08", "41.60", "+9.11", "38.80", "+1.77"),
        "2018-10-26": ("23.07", "26.53", "+15.00", "26.53", "+15.00", "26.90", "+16.59", "23.37", "+1.29"),
        "Total": ("137.49", "156.36", "+13.68", "156.43", "+13.7", "162.04", "+17.81", "142.69", "+3.7"),
    }
    names = ["MILP", "Exact QUBO cost", "Exact QUBO delta (%)", "IBM cost", "IBM delta (%)",
             "D-Wave QPU cost", "D-Wave QPU delta (%)", "D-Wave Stride cost", "D-Wave Stride delta (%)"]
    keys = ["milp_table_column", "exact_qubo_cost", "exact_qubo_delta_pct", "ibm_qaoa_p2_p3_cost",
            "ibm_qaoa_p2_p3_delta_pct", "dwave_qpu_cost", "dwave_qpu_delta_pct",
            "dwave_stride_cost", "dwave_stride_delta_pct"]
    for d, vals in paper_t2.items():
        for name, key, p in zip(names, keys, vals):
            if "delta" in key:
                dec = len(p.split(".")[1])
                fmt = "{:+.%df}" % dec
            else:
                fmt = "{:.2f}"
            note = "relative to the rolling MILP of the same date (results/by_date)" if "delta" in key else ""
            if key == "milp_table_column":
                note = "grid-aware MILP on 2018-07-23, stored MILP on the other dates"
            add("Table II", f"{d} {name}", p, float(t.loc[d, key]), fmt,
                "results/paper/table2_daily_cost.csv", note)
    p2 = runs_cache["qaoa_p2"].set_index(["date", "hour"]).cost
    p3 = runs_cache["qaoa_p3"].set_index(["date", "hour"]).cost
    diff = float((p2 - p3).abs().max())
    add("Table II", "IBM p=2 vs p=3 per-window cost difference", "< 1e-9",
        "< 1e-9" if diff < 1e-9 else f"{diff:.1e}", "", "results/by_date/*_ibm_kingston_p2, *_p3",
        f"maximum absolute difference {diff:g} over {len(p2)} paired windows")

    # Section IV-B
    add("IV-B", "Terminal SOC, MILP (2018-07-23)", "0.162", finals["milp"], "{:.3f}",
        "results/paper/soc_2018-07-23.csv")
    add("IV-B", "Terminal SOC, IBM (2018-07-23)", "0.128", finals["ibm_qaoa_p2_p3"], "{:.3f}",
        "results/paper/soc_2018-07-23.csv")
    add("IV-B", "D-Wave raw dispatch feasible", "144/144", f"{int(dwr.raw_feasible)}/{int(dwr.windows)}", "",
        "results/paper/feasibility_repair.csv")
    add("IV-B", "D-Wave windows where repair fired", "9/144", f"{int(dwr.repair_used)}/{int(dwr.windows)}", "",
        "results/paper/feasibility_repair.csv")
    add("IV-B", "Bits changed by repair (of 20), mean", "2.4", dwr.mean_bits_changed_when_repaired, "{:.1f}",
        "results/paper/feasibility_repair.csv", "D-Wave windows where repair fired")
    add("IV-B", "Random sampling feasible windows", "58/144",
        f"{int(rnd.final_feasible)}/{int(rnd.windows)}", "", "results/paper/feasibility_repair.csv")

    # Section IV-C
    add("IV-C", "IBM Kingston p=2 mean time per window (s)", "332.1", rt["IBM Kingston QAOA p=2"], "{:.1f}",
        "results/paper/runtime.csv")
    add("IV-C", "IBM Kingston p=3 mean time per window (s)", "663.5", rt["IBM Kingston QAOA p=3"], "{:.1f}",
        "results/paper/runtime.csv")
    add("IV-C", "D-Wave mean time per window (s)", "2.006",
        rt["D-Wave Advantage_system6.4"], "{:.3f}", "results/paper/runtime.csv")

    # Conclusion
    add("V", "Encoding share of the gap (%)", "13.68", g("exact", "gap_vs_milp_137_54_pct"), "{:.2f}",
        "results/paper/solver_totals.csv", "basis: MILP 137.54")
    add("V", "Metaheuristics above exact QUBO (%)", "3.1-3.7",
        f"{min(meta_gap):.1f}-{max(meta_gap):.1f}", "", "results/paper/solver_totals.csv",
        "SQA, simulated annealing, tabu, steepest descent")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

SERIES = [
    ("milp_table_column", "MILP (classical optimal)", "#08519C", ""),
    ("exact_qubo_cost", "Exact QUBO (enumeration)", "#3182BD", "--"),
    ("ibm_p2", "QAOA $p{=}2$  (IBM Kingston)", "#6BAED6", "//"),
    ("ibm_p3", "QAOA $p{=}3$  (IBM Kingston)", "#9ECAE1", "\\\\"),
    ("dwave_qpu_cost", "D-Wave QA  (Advantage2_system1)", "#C6DBEF", "xx"),
    ("dwave_stride_cost", "D-Wave Stride  (Leap Hybrid NLP v1)", "#DEEBF7", ".."),
]


def fig2(t2: pd.DataFrame, runs) -> Path:
    df = t2[t2.date != "Total"].set_index("date").copy()
    df["ibm_p2"] = per_date(runs["qaoa_p2"])
    df["ibm_p3"] = per_date(runs["qaoa_p3"])
    keys = [k for k, *_ in SERIES]
    x = np.arange(len(df))
    width = 0.135
    offsets = (np.arange(len(keys)) - (len(keys) - 1) / 2) * width
    fig, ax = plt.subplots(figsize=(14.5, 5.4))
    for off, (key, label, color, hatch) in zip(offsets, SERIES):
        vals = df[key].to_numpy()
        ax.bar(x + off, vals, width * 0.92, label=label, color=color, hatch=hatch,
               edgecolor="#1F3B57", linewidth=0.8, zorder=3)
        if key == "milp_table_column":
            continue
        dy = 3 if keys.index(key) % 2 else 12
        for xi, (v, m) in enumerate(zip(vals, df["milp_table_column"].to_numpy())):
            ax.annotate(f"+{100 * (v - m) / m:.0f}%", (x[xi] + off, v), textcoords="offset points",
                        xytext=(0, dy), ha="center", fontsize=7.5, color="#333333")
    half = (len(keys) * width) / 2
    for xi, m in enumerate(df["milp_table_column"].to_numpy()):
        ax.hlines(m, x[xi] - half - 0.03, x[xi] + half + 0.03, colors="#3182BD",
                  linestyles=(0, (4, 3)), linewidth=1.3, zorder=4)
    ax.set_title("Total daily operating cost: MILP, exact QUBO optimum, IBM Kingston "
                 "($p{=}2$, $p{=}3$), D-Wave QA, D-Wave Stride", fontsize=13.5,
                 fontweight="bold", pad=14)
    ax.set_xlabel("Study Date", fontsize=12)
    ax.set_ylabel("Total Daily Cost (p.u. h)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{pd.Timestamp(d).strftime('%b-%d')}\n{SEASON[d]}" for d in df.index], fontsize=11)
    ax.set_ylim(0, df[keys].max().max() * 1.22)
    ax.grid(axis="y", color="#E4E4E4", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    leg = ax.legend(ncol=3, fontsize=9.5, loc="upper left", framealpha=1.0,
                    title="MILP baseline shown as dashed line per date", borderpad=0.6,
                    labelspacing=0.5, handlelength=1.8)
    leg.get_title().set_fontsize(10)
    fig.text(0.5, 0.005, "Gap % vs MILP shown above each bar", ha="center", fontsize=10,
             style="italic", color="#555555")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    out = FIG / "fig2_daily_cost.png"
    fig.savefig(out, dpi=C.FIGURE_DPI, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    return out


TRACES = [
    ("milp", "MILP (optimal baseline)", "#1F77B4", "o", "--"),
    ("exact_qubo", "Exact QUBO (enumeration)", "#7F7F7F", None, None),
    ("dwave_qpu", "D-Wave QA (Advantage2_system1)", "#2CA02C", "v", "-"),
    ("dwave_stride", "D-Wave Stride (Leap Hybrid NLP v1)", "#D62728", "s", "-"),
    ("ibm_qaoa_p2_p3", "IBM Kingston $p{=}2,3$", "#FF7F0E", "D", "-"),
]


def _style(ax, title: str, ylabel: str, ylim) -> None:
    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Hour of Day", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlim(0, 24)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xticks(range(0, 25, 2))
    ax.grid(color="#E8E8E8", linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _plot_traces(ax, data: pd.DataFrame) -> None:
    for key, label, color, marker, ls in TRACES:
        s = data[key].dropna()
        if marker is None:
            ax.plot(s.index, s.values, "-", color=color, linewidth=5.0, alpha=0.40, label=label,
                    zorder=2, solid_capstyle="round")
        else:
            ax.plot(s.index, s.values, ls, color=color, marker=marker, markersize=5.5,
                    linewidth=1.8, label=label, zorder=3)


def _outside(ax, box_title: str | None = None, lines: list[str] | None = None) -> None:
    """Legend and an optional text box to the right of the axes, clear of the data."""
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=10, framealpha=1.0)
    if lines:
        ax.text(1.01, 0.0, box_title + "\n" + "\n".join(lines), transform=ax.transAxes,
                fontsize=9.5, va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#BBBBBB",
                          linewidth=0.8))


def _p23(df: pd.DataFrame) -> pd.DataFrame:
    """IBM p=2 and p=3 give the same dispatch; plot them as one trace."""
    out = df.copy()
    if "ibm_qaoa_p2" in out:
        out["ibm_qaoa_p2_p3"] = out["ibm_qaoa_p2"]
    return out


def fig_grid_import(sched: pd.DataFrame, date: str, out: Path) -> Path:
    data = sched.pivot_table(index="hour", columns="method", values="grid", aggfunc="mean")
    data = _p23(data)
    fig, ax = plt.subplots(figsize=(15, 5.2))
    _plot_traces(ax, data)
    _style(ax, f"Grid Import, {date}", "Grid import (fraction of $P_G$, hourly mean)", (0, 1.1))
    _outside(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=C.FIGURE_DPI, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    return out


def fig_hourly_cost(cost: pd.DataFrame, date: str, out: Path) -> Path:
    data = _p23(cost)
    fig, ax = plt.subplots(figsize=(15, 5.2))
    _plot_traces(ax, data)
    totals = [f"  {label.replace('$p{=}2,3$', 'p=2,3')}: {data[key].sum():.2f}"
              for key, label, *_ in TRACES]
    _style(ax, f"Hourly Operating Cost, {date}", "Cost per window (p.u. h)", None)
    ax.set_ylim(0, data[[k for k, *_ in TRACES]].max().max() * 1.1)
    _outside(ax, "Daily total:", totals)
    fig.tight_layout()
    fig.savefig(out, dpi=C.FIGURE_DPI, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    return out


def fig_soc(soc: pd.DataFrame, date: str, out: Path, outside: bool = False) -> Path:
    soc = _p23(soc)
    traces = TRACES
    fig, ax = plt.subplots(figsize=(15, 5.2))
    ax.axhline(C.SOC_MAX, color="#D62728", linestyle=":", linewidth=1.4,
               label=f"Operating band ({C.SOC_MIN:.2f}-{C.SOC_MAX:.2f})")
    ax.axhline(C.SOC_MIN, color="#D62728", linestyle=":", linewidth=1.4)
    ax.axhline(C.SOC_INIT, color="#4C78B8", linewidth=1.0, label=f"Initial SOC ({C.SOC_INIT:.2f})")
    finals = []
    for key, label, color, marker, ls in traces:
        s = soc[key].dropna()
        if marker is None:
            ax.plot(s.index, s.values, "-", color=color, linewidth=5.0, alpha=0.40, label=label,
                    zorder=2, solid_capstyle="round")
        else:
            ax.plot(s.index, s.values, ls, color=color, marker=marker, markersize=5.5,
                    linewidth=1.8, label=label, zorder=3)
        finals.append(f"  {label.replace('$p{=}2,3$', 'p=2,3')}: {s.iloc[-1]:.3f}")
    _style(ax, f"Battery SOC Trajectory, {date}", "State of Charge (fraction)", (0, 1.0))
    if outside:
        _outside(ax, "Final SOC:", finals)
    else:
        ax.legend(loc="upper right", fontsize=10, framealpha=1.0)
        ax.text(0.012, 0.03, "Final SOC:\n" + "\n".join(finals), transform=ax.transAxes,
                fontsize=9.5, va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#BBBBBB",
                          linewidth=0.8))
    fig.tight_layout()
    fig.savefig(out, dpi=C.FIGURE_DPI, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    return out


def per_date_outputs() -> list[Path]:
    """Tables under results/paper/by_date/<date>/ and figures under figures/by_date/<date>/."""
    written = []
    study_days().to_csv(OUT / "study_days.csv", index=False)
    for d in DATES:
        t_dir, f_dir = OUT / "by_date" / d, FIG / "by_date" / d
        t_dir.mkdir(parents=True, exist_ok=True)
        f_dir.mkdir(parents=True, exist_ok=True)
        soc, sched, cost = soc_day(d), schedule_day(d), hourly_cost_day(d)
        summary_day(d).round(4).to_csv(t_dir / "summary.csv", index=False)
        cost.round(6).to_csv(t_dir / "hourly_cost.csv")
        soc.round(4).to_csv(t_dir / "soc.csv")
        sched.pivot_table(index="hour", columns="method", values="grid", aggfunc="mean").round(4) \
            .to_csv(t_dir / "grid_import_hourly.csv")
        sched.to_csv(t_dir / "schedule_15min.csv", index=False)
        written += [fig_soc(soc, d, f_dir / "soc.png", outside=True),
                    fig_grid_import(sched, d, f_dir / "grid_import.png"),
                    fig_hourly_cost(cost, d, f_dir / "hourly_cost.png")]
    return written


# ---------------------------------------------------------------------------

runs_cache: dict[str, pd.DataFrame] = {}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    runs_cache.update(load_all())
    runs = runs_cache

    t2 = table2(runs)
    tot = solver_totals(runs)
    rt = runtime(runs)
    rep = feasibility_repair(runs)
    sc, sc_sum = scenarios()
    soc = soc_peak_day()
    nums = paper_numbers(t2, tot, rt, rep, sc_sum, soc)

    t2.round(4).to_csv(OUT / "table2_daily_cost.csv", index=False)
    tot.round(4).to_csv(OUT / "solver_totals.csv", index=False)
    rt.round(4).to_csv(OUT / "runtime.csv", index=False)
    rep.round(4).to_csv(OUT / "feasibility_repair.csv", index=False)
    sc.round(4).to_csv(OUT / "scenarios.csv", index=False)
    soc.round(4).to_csv(OUT / f"soc_{PEAK_DAY}.csv")
    nums.to_csv(OUT / "paper_numbers.csv", index=False)
    dwave_timing()[["date", "hour", "problem_id", "qpu_access_s"]].round(6).to_csv(
        OUT / "dwave_problem_ids.csv", index=False)

    f2 = fig2(t2, runs)
    f3 = fig_soc(soc, PEAK_DAY, FIG / f"fig3_soc_{PEAK_DAY}.png")
    per_date = per_date_outputs()

    counts = nums.status.value_counts()
    print(f"paper numbers checked: {len(nums)}")
    for k, v in counts.items():
        print(f"  {k:24s} {v}")
    diff = nums[nums.status != "match"]
    if len(diff):
        print("\nnot matching or not stored:")
        for r in diff.itertuples():
            print(f"  [{r.section}] {r.quantity}: paper {r.paper_value} | repository {r.repository_value or '-'}")
    print(f"\nwrote {OUT.relative_to(REPO_ROOT)}/, {f2.relative_to(REPO_ROOT)}, {f3.relative_to(REPO_ROOT)}"
          f" and {len(per_date)} per-date figures under figures/by_date/")


if __name__ == "__main__":
    main()
