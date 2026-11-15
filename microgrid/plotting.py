from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from typing import List, Optional, Tuple

from microgrid.config import SOLVER_COLORS, SOLVER_LABELS, FIGURE_DPI, RESULTS_DIR, SOC_INIT


# Style helpers

_STYLE = {
    "figure.facecolor": "white",
    "axes.grid":        True,
    "grid.alpha":       0.25,
    "grid.linewidth":   0.8,
    "font.family":      "sans-serif",
    "font.size":        10,
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "axes.linewidth":   1.2,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  10,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "gray",
    "lines.linewidth":  2.2,
    "lines.markersize": 6,
    "axes.spines.top":  False,
    "axes.spines.right": False,
}

SOLVERS_ORDERED = ["milp", "qaoa_p2", "qaoa_p3", "qaoa_p5", "sqa", "dwave_qa", "dwave_hybrid", "ibm_p2", "ibm_p3", "ibm_p5"]

_MARKERS = {"milp": "o", "qaoa_p2": "s", "qaoa_p3": "s", "qaoa_p5": "s", "sqa": "^", "dwave_qa": "v", "dwave_hybrid": "P", "ibm_p2": "D", "ibm_p3": "D", "ibm_p5": "D"}

_SEASON_FOR_DATE = {
    "2018-03-03": "Winter",
    "2018-04-13": "Spring",
    "2018-04-14": "Spring",
    "2018-06-27": "Summer",
    "2018-07-23": "Summer",
    "2018-10-26": "Fall",
}


def _style() -> None:
    plt.rcParams.update(_STYLE)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def _hours(df: pd.DataFrame, col: str = "hour") -> List[int]:
    return sorted(df[col].unique().tolist())


def _hours_display(hours: List[int]) -> np.ndarray:
    """Convert 1-indexed data hours to 0-indexed display hours for x-axis."""
    return np.array(hours) - 1


def _set_hour_axis(ax, hours: List[int] | None = None, y_from_zero: bool = True) -> None:
    """Anchor x-axis so hour 0 is at left edge; optionally anchor y bottom to 0."""
    n = 24 if hours is None else max(len(hours), 1)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_xticks(range(0, n, 2))
    ax.margins(x=0)
    if y_from_zero:
        cur = ax.get_ylim()
        if cur[1] > 0:
            ax.set_ylim(bottom=0)


def _series(df_cmp: pd.DataFrame, solver: str, col: str, hours: List[int]) -> np.ndarray:
    sub = df_cmp[df_cmp["solver"] == solver].set_index("hour")
    return np.array([float(sub.loc[h, col]) if h in sub.index else np.nan for h in hours])


def _sched(df_s: pd.DataFrame, solver: str, col: str, hours: List[int],
           substep: str = "last") -> np.ndarray:
    max_sub = df_s["substep"].max() if "substep" in df_s.columns else 0
    use_sub = max_sub if substep == "last" else 0
    mask = (df_s["solver"] == solver)
    if "substep" in df_s.columns:
        mask &= (df_s["substep"] == use_sub)
    sub = df_s[mask].set_index("hour")
    return np.array([float(sub.loc[h, col]) if h in sub.index and col in sub.columns else np.nan
                     for h in hours])


def _sched_all(df_s: pd.DataFrame, solver: str, col: str) -> Tuple[np.ndarray, np.ndarray]:
    sub = df_s[df_s["solver"] == solver].sort_values("global_step")
    if col not in sub.columns or "global_step" not in sub.columns:
        return np.array([]), np.array([])
    return sub["global_step"].values, sub[col].values


def _active_solvers(df: pd.DataFrame, backend_filter: Optional[str] = None, keep_milp: bool = True) -> List[str]:
    """Return solvers present in df, optionally filtered by execution backend.

    backend_filter:
      None -> all solvers
      "simulator" -> qaoa_p2, qaoa_p3, qaoa_p5, sqa only (+ MILP if keep_milp=True)
      "ibm_kingston" -> ibm_p2, ibm_p3, ibm_p5, sqa only (+ MILP if keep_milp=True)
    """
    present = set(df["solver"].unique())
    solvers = [s for s in SOLVERS_ORDERED if s in present]

    if backend_filter == "simulator":
        # Keep only simulator solvers (exclude IBM), optionally keep MILP baseline
        solvers = [s for s in solvers if s == "milp" and keep_milp or not s.startswith("ibm_") and s != "milp"]
    elif backend_filter == "ibm_kingston":
        # Keep IBM solvers + SQA (for comparison), optionally keep MILP baseline
        solvers = [s for s in solvers if s == "milp" and keep_milp or s.startswith("ibm_") or s == "sqa"]

    return solvers


def _shade_infeasible_hours(ax, df_cmp: pd.DataFrame, span: float = 0.4) -> None:
    """Shade infeasible hours red on an hour-axis plot (0-indexed display)."""
    if "is_feasible" not in df_cmp.columns:
        return
    infeasible_hours = df_cmp[~df_cmp["is_feasible"]]["hour"].unique()
    added = False
    for hour in sorted(infeasible_hours):
        h0 = hour - 1
        label = "Infeasible hour" if not added else None
        ax.axvspan(h0 - span, h0 + span, alpha=0.15, color="red", hatch="///", label=label)
        added = True


def _shade_infeasible_steps(ax, df_cmp: pd.DataFrame) -> None:
    """Shade infeasible hours red on a global_step-axis plot (4 steps per hour)."""
    if "is_feasible" not in df_cmp.columns:
        return
    infeasible_hours = df_cmp[~df_cmp["is_feasible"]]["hour"].unique()
    added = False
    for hour in sorted(infeasible_hours):
        step_lo = (hour - 1) * 4 - 0.5
        step_hi = hour * 4 - 0.5
        label = "Infeasible hour" if not added else None
        ax.axvspan(step_lo, step_hi, alpha=0.15, color="red", hatch="///", label=label)
        added = True


# Per-date figures (fig1-fig8, no LP)

def plot_hourly_cost(df_cmp: pd.DataFrame, date: str, out_dir: Path, backend_filter: Optional[str] = None) -> None:
    """Fig 4: hourly operating cost - Classical, QAOA, SQA + daily totals."""
    _style()
    fig, ax = plt.subplots(figsize=(12, 5))
    hours = _hours(df_cmp)

    _shade_infeasible_hours(ax, df_cmp)
    solvers = _active_solvers(df_cmp, backend_filter=backend_filter)
    daily_totals = {}
    for solver in solvers:
        vals = _series(df_cmp, solver, "cost", hours)
        daily_totals[solver] = float(np.nansum(vals))
        ax.plot(_hours_display(hours), vals,
                color=SOLVER_COLORS.get(solver, "#444"),
                marker=_MARKERS.get(solver, "o"),
                linestyle="--" if solver == "milp" else "-",
                linewidth=2,
                label=SOLVER_LABELS.get(solver, solver))

    ax.set_title(f"Hourly Operating Cost - {date}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Hour of Day", fontsize=11)
    ax.set_ylabel("Hourly Cost ($/pu)", fontsize=11)
    _set_hour_axis(ax, hours)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.3)

    # Daily totals annotation
    txt = "Daily total cost:\n"
    for solver in solvers:
        label = SOLVER_LABELS.get(solver, solver).replace(" (exact QUBO)", "").replace(" (", "\n(")
        txt += f"{label}: {daily_totals[solver]:.2f}\n"
    ax.text(0.98, 0.97, txt, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    _save(fig, out_dir / "fig4_hourly_cost_details.png")


def plot_battery_cd(df_sched: pd.DataFrame, date: str, out_dir: Path,
                    df_cmp: Optional[pd.DataFrame] = None, backend_filter: Optional[str] = None) -> None:
    """Fig 7: 15-min battery charge/discharge power - all 4 substeps per hour per solver."""
    _style()
    solvers = list(_active_solvers(df_sched, backend_filter=backend_filter))
    if not solvers:
        return

    n_rows = len(solvers)
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 3.5 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    fig.suptitle(f"15-Minute Battery Charge/Discharge Power - {date}", fontsize=13, y=0.995)

    C_CHG  = "#2ca02c"
    C_DIS  = "#d62728"
    C_IDLE = "#cccccc"

    for ax, solver in zip(axes, solvers):
        sub = df_sched[df_sched["solver"] == solver].sort_values("global_step")
        if sub.empty:
            continue

        global_steps = sub["global_step"].values
        batt_mw = sub["batt_mw"].values

        bar_colors = []
        for b in batt_mw:
            if b > 0.5:      bar_colors.append(C_CHG)
            elif b < -0.5:   bar_colors.append(C_DIS)
            else:            bar_colors.append(C_IDLE)

        ax.bar(global_steps, batt_mw, color=bar_colors,
               edgecolor="black", linewidth=0.4, width=0.8)

        if df_cmp is not None:
            _shade_infeasible_steps(ax, df_cmp)

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel(f"{SOLVER_LABELS.get(solver, solver)}\nBatt Power (MW)", fontsize=10)
        ax.grid(axis="y", alpha=0.25, linestyle="--")

    # Shared legend
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=C_CHG, edgecolor="black", label="Charging"),
                  Patch(facecolor=C_IDLE, edgecolor="black", label="Idle"),
                  Patch(facecolor=C_DIS, edgecolor="black", label="Discharging")]
    fig.legend(handles=legend_els, loc="lower center", fontsize=10, ncol=3, bbox_to_anchor=(0.5, -0.01))

    axes[-1].set_xlabel("15-Minute Step (96 total per 24 hours)", fontsize=11, fontweight="bold")
    axes[-1].xaxis.set_major_locator(ticker.MultipleLocator(4))  # 4 steps = 1 hour
    axes[-1].xaxis.set_minor_locator(ticker.MultipleLocator(1))

    # Hour gridlines
    for ax in axes:
        for h in range(1, 25):
            ax.axvline(h * 4, color="gray", linestyle=":", linewidth=1, alpha=0.5)

    _save(fig, out_dir / "fig7_battery_charge_discharge.png")


def plot_soc(df_sched: pd.DataFrame, date: str, out_dir: Path,
             df_cmp: Optional[pd.DataFrame] = None, backend_filter: Optional[str] = None) -> None:
    """Fig 5: hourly SOC trajectory - initial 0.50 reference line + operating band + final SOC."""
    _style()
    fig, ax = plt.subplots(figsize=(12, 5))
    hours = _hours(df_sched)

    if df_cmp is not None:
        _shade_infeasible_hours(ax, df_cmp)

    solvers = _active_solvers(df_sched, backend_filter=backend_filter)
    final_soc = {}

    for solver in solvers:
        soc = _sched(df_sched, solver, "soc", hours, substep="last")
        final_soc[solver] = float(soc[-1]) if len(soc) > 0 and not np.isnan(soc[-1]) else np.nan
        # x=0 shows the initial SOC; x=1..23 show end-of-hour SOC for hours 1..23.
        y_disp = np.concatenate(([SOC_INIT], soc[:-1]))
        x_disp = np.arange(len(y_disp))
        ax.plot(x_disp, y_disp,
                color=SOLVER_COLORS.get(solver, "#444"),
                marker=_MARKERS.get(solver, "o"),
                linestyle="--" if solver == "milp" else "-",
                linewidth=2,
                label=SOLVER_LABELS.get(solver, solver))

    # Initial SOC reference (0.50)
    ax.axhline(0.50, color="blue", linestyle="-", linewidth=1.5, alpha=0.6, label="Initial SOC (0.50)")

    # Operating band (0.10 - 0.90)
    ax.axhline(0.10, color="red", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.axhline(0.90, color="red", linestyle=":", linewidth=1.2, alpha=0.7, label="Operating band (0.10-0.90)")
    ax.fill_between([min(hours) - 0.5, max(hours) + 0.5], 0.10, 0.90, alpha=0.04, color="gray")

    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"Battery SOC Trajectory - {date}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Hour of Day", fontsize=11)
    ax.set_ylabel("State of Charge (fraction)", fontsize=11)
    _set_hour_axis(ax, hours)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)

    # Final SOC annotation
    txt = "Final SOC:\n"
    for solver in solvers:
        if solver in final_soc and not np.isnan(final_soc[solver]):
            label = SOLVER_LABELS.get(solver, solver).replace(" (exact QUBO)", "").replace(" (", "\n(")
            txt += f"{label}: {final_soc[solver]:.3f}\n"
    ax.text(0.02, 0.05, txt, transform=ax.transAxes, fontsize=9,
            verticalalignment="bottom", horizontalalignment="left",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8))

    _save(fig, out_dir / "fig5_battery_soc_trajectory.png")


def plot_grid_import(df_sched: pd.DataFrame, date: str, out_dir: Path,
                       df_cmp: Optional[pd.DataFrame] = None, backend_filter: Optional[str] = None) -> None:
    """Fig 6: grid import fraction per solver (no LP)."""
    _style()
    fig, ax = plt.subplots(figsize=(12, 5))
    hours = _hours(df_sched)

    if df_cmp is not None:
        _shade_infeasible_hours(ax, df_cmp)

    for solver in _active_solvers(df_sched, backend_filter=backend_filter):
        grid = _sched(df_sched, solver, "grid", hours)
        ax.plot(_hours_display(hours), grid,
                color=SOLVER_COLORS.get(solver, "#444"),
                marker=_MARKERS.get(solver, "o"),
                linestyle="--" if solver == "milp" else "-",
                label=SOLVER_LABELS.get(solver, solver))

    ax.set_ylim(-0.05, 1.1)
    ax.set_title(f"Grid Import Fraction - 15-min scheduling - {date}")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Grid Fraction (0-1)")
    _set_hour_axis(ax, hours)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
    ax.legend(loc="upper right")
    _save(fig, out_dir / "fig6_grid_import_fraction.png")


def plot_net_load(df_sched: pd.DataFrame, date: str, out_dir: Path,
                    mg=None, df_cmp: Optional[pd.DataFrame] = None, backend_filter: Optional[str] = None) -> None:
    """Fig 3: net load profile - load bars, stacked wind+solar, net load line (MW)."""
    _style()

    first_solver = _active_solvers(df_sched, backend_filter=backend_filter)[0] if _active_solvers(df_sched, backend_filter=backend_filter) else None
    if first_solver is None:
        return

    hours = _hours(df_sched)
    mask  = (df_sched["solver"] == first_solver)
    if "substep" in df_sched.columns:
        mask &= (df_sched["substep"] == 3)
    sub = df_sched[mask].set_index("hour")

    def _col(c):
        if c in sub.columns:
            return np.array([float(sub.loc[h, c]) if h in sub.index else np.nan for h in hours])
        return np.full(len(hours), np.nan)

    load  = _col("load_mw")
    solar = _col("solar_mw")
    wind  = _col("wind_mw")

    net_mw = load - solar - wind

    fig, ax = plt.subplots(figsize=(14, 6))

    if df_cmp is not None:
        _shade_infeasible_hours(ax, df_cmp)

    _hd = _hours_display(hours)
    ax.bar(_hd, load, label="Load Demand", color="#1f77b4", alpha=0.8, width=0.75, edgecolor="black", linewidth=0.5)

    # Stacked bars: wind + solar
    ax.bar(_hd, wind, label="Wind Generation", color="#2ca02c", alpha=0.75, width=0.75, edgecolor="black", linewidth=0.5)
    ax.bar(_hd, solar, bottom=wind, label="Solar Generation", color="#ff7f0e", alpha=0.75, width=0.75, edgecolor="black", linewidth=0.5)

    # Net load line
    ax.plot(_hd, net_mw, color="#d62728", marker="o", linewidth=3, markersize=8,
            label="Net Load", zorder=5, markerfacecolor="#d62728",
            markeredgecolor="white", markeredgewidth=1.5)

    ax.axhline(y=0, color="gray", linestyle="-", linewidth=1, alpha=0.6, zorder=2)
    ax.set_xlabel("Hour of Day", fontsize=13, fontweight="bold")
    ax.set_ylabel("Power (MW)", fontsize=13, fontweight="bold")
    ax.set_title(f"{date}\nLoad Demand vs. Renewable Generation Profile", fontsize=14, fontweight="bold", pad=20)
    ax.legend(fontsize=11, loc="lower right", framealpha=0.95, edgecolor="black", fancybox=True, shadow=True)
    ax.grid(axis="y", alpha=0.4, linestyle="--")
    _set_hour_axis(ax, hours, y_from_zero=False)
    ax.set_facecolor("#f8f9fa")

    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.2)

    _save(fig, out_dir / "fig3_load_renewable_profile.png")


def plot_cost_breakdown(df_sched: pd.DataFrame, df_cmp: pd.DataFrame,
                          date: str, out_dir: Path, mg=None, backend_filter: Optional[str] = None) -> None:
    """Fig 8: dual-panel hourly cost breakdown (grid top, battery bottom) per solver."""
    _style()
    solvers = list(_active_solvers(df_cmp, backend_filter=backend_filter))
    if not solvers:
        return

    from microgrid.config import PRICE_VEC, DT
    hours = _hours(df_cmp)
    n = len(solvers)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle(f"Cost Component Breakdown - {date}", fontsize=13, y=0.98)
    fig.text(0.5, 0.92, "Blue: Grid import cost (top row).  Orange: Battery cycling cost (bottom row, much smaller).",
             ha="center", fontsize=9, style="italic", color="gray")
    fig.subplots_adjust(top=0.88, hspace=0.3)

    P_G     = mg.P_G     if mg is not None else 1.0
    P_B     = mg.P_B     if mg is not None else 0.5
    C_grid  = mg.C_grid  if mg is not None else 1.0
    C_cycle = mg.C_cycle if mg is not None else 0.01
    price   = np.array(PRICE_VEC[:max(hours) + 1], dtype=float) if PRICE_VEC is not None else np.ones(max(hours) + 1)

    for col, solver in enumerate(solvers):
        grid_cost  = []
        batt_cost  = []
        for h in hours:
            g_val = _sched(df_sched, solver, "grid", [h])[0]
            sub_mask = (df_sched["hour"] == h) & (df_sched["solver"] == solver)
            c_total = df_sched.loc[sub_mask, "charge"].sum()
            d_total = df_sched.loc[sub_mask, "discharge"].sum()
            p  = price[h] if h < len(price) else 1.0
            gc = DT * p * C_grid  * P_G * (g_val if not np.isnan(g_val) else 0)
            bc = DT * p * C_cycle * P_B * ((c_total if not np.isnan(c_total) else 0)
                                           + (d_total if not np.isnan(d_total) else 0))
            grid_cost.append(gc)
            batt_cost.append(bc)

        grid_arr = np.array(grid_cost)
        batt_arr = np.array(batt_cost)

        # Top panel: Grid import cost
        ax_grid = axes[0, col]
        ax_grid.bar(_hours_display(hours), grid_arr, width=0.7, color="#4472C4", alpha=0.85)
        _shade_infeasible_hours(ax_grid, df_cmp)
        ax_grid.set_title(SOLVER_LABELS.get(solver, solver), fontsize=11)
        ax_grid.set_ylabel("Grid Cost (pu·h)", fontsize=9)
        ax_grid.xaxis.set_major_locator(ticker.MultipleLocator(4))
        ax_grid.grid(axis="y", alpha=0.3)

        # Bottom panel: Battery cycling cost
        ax_batt = axes[1, col]
        ax_batt.bar(_hours_display(hours), batt_arr, width=0.7, color="#ED7D31", alpha=0.85)
        _shade_infeasible_hours(ax_batt, df_cmp)
        ax_batt.set_ylabel("Battery Cost (pu·h)", fontsize=9)
        ax_batt.set_xlabel("Hour of Day")
        ax_batt.xaxis.set_major_locator(ticker.MultipleLocator(4))
        ax_batt.grid(axis="y", alpha=0.3)

    path = out_dir / "fig8_cost_component_breakdown.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)


def plot_solver_summary(df_cmp: pd.DataFrame, date: str, out_dir: Path, backend_filter: Optional[str] = None) -> None:
    """Fig 1: solver summary table - feasibility, cost, gap, runtime."""
    _style()

    solvers  = _active_solvers(df_cmp, backend_filter=backend_filter)
    cl_total = None
    if "milp" in df_cmp["solver"].values:
        cl_sub   = df_cmp[df_cmp["solver"] == "milp"]
        cl_total = float(cl_sub["cost"].sum())

    headers = ["Solver", "Feasible Hrs", "Feas Rate", "Total Cost",
               "Mean Hourly Cost", "Gap vs CL %", "Runtime (s)"]
    cells       = []
    cell_colors = []

    for solver in solvers:
        sub    = df_cmp[df_cmp["solver"] == solver]
        feas   = int(sub["is_feasible"].sum())
        n_hrs  = len(sub)
        rate   = f"{100 * feas / n_hrs:.1f}%" if n_hrs else "-"
        tot    = float(sub["cost"].sum())
        mean_c = float(sub["cost"].mean())
        if solver == "milp" or cl_total is None or abs(cl_total) < 1e-9:
            gap_s = "-"
        else:
            gap_pct = 100.0 * (tot - cl_total) / abs(cl_total)
            gap_s   = f"{gap_pct:+.1f}%"
        rt = "-"
        if "runtime_s" in sub.columns:
            rt_val = sub["runtime_s"].mean()
            rt = f"{rt_val:.2f}" if not np.isnan(rt_val) else "-"

        cells.append([
            SOLVER_LABELS.get(solver, solver),
            f"{feas}/{n_hrs}", rate,
            f"{tot:.4f}", f"{mean_c:.4f}",
            gap_s, rt,
        ])
        bg = "#ddeeff" if solver == "milp" else "white"
        cell_colors.append([bg] * len(headers))

    fig, ax = plt.subplots(figsize=(13, 0.55 * len(cells) + 1.2))
    ax.axis("off")
    table = ax.table(cellText=cells, colLabels=headers, cellColours=cell_colors,
                     cellLoc="center", colLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    for k, cell in table.get_celld().items():
        if k[0] == 0:
            cell.set_facecolor("#1f3a68")
            cell.set_text_props(color="white", weight="bold")
        cell.set_edgecolor("#aaa")
    ax.set_title(f"Solver Summary - {date}",
                 fontsize=11, pad=12, weight="bold")
    fig.subplots_adjust(bottom=0.12)
    fig.text(0.5, 0.03, "Results summarize 24 hourly scheduling blocks, with each block containing four committed 15-minute steps.",
             ha="center", fontsize=8, style="italic", color="gray")
    _save(fig, out_dir / "fig1_solver_summary.png")


def plot_hourly_results_table(df_cmp: pd.DataFrame, date: str, out_dir: Path, backend_filter: Optional[str] = None) -> None:
    """Fig 2: per-hour results table - CL/QAOA/SQA costs + gaps, peak hours highlighted yellow."""
    _style()

    hours   = _hours(df_cmp)
    solvers = _active_solvers(df_cmp, backend_filter=backend_filter)

    # Build cost + gap per hour per solver
    cost_by = {}
    for solver in solvers:
        cost_by[solver] = _series(df_cmp, solver, "cost", hours)

    cl_costs = cost_by.get("milp", np.full(len(hours), np.nan))

    # Identify peak hours: top 20% of MILP cost.
    threshold = np.nanpercentile(cl_costs, 80)

    # Detect infeasible hours per solver
    infeas_by_solver = {}
    for solver in solvers:
        sub = df_cmp[df_cmp["solver"] == solver]
        infeas = set(sub[~sub["is_feasible"]]["hour"].unique())
        infeas_by_solver[solver] = infeas

    # Detect if IBM or simulator run, build headers dynamically
    has_ibm = any(s.startswith("ibm_") for s in solvers)
    extra_solvers = [s for s in ("dwave_qa", "dwave_hybrid") if s in solvers]
    if has_ibm:
        # IBM run: show MILP + every IBM solver present (+ any D-Wave annealing solver)
        ibm_solvers = [s for s in solvers if s.startswith("ibm_")]
        headers = ["Hour", "Classical (Baseline)"]
        for ibm_s in ibm_solvers:
            lbl = SOLVER_LABELS.get(ibm_s, ibm_s)
            headers += [lbl, f"{lbl} Δ†"]
        for s in extra_solvers:
            lbl = SOLVER_LABELS.get(s, s)
            headers += [lbl, f"{lbl} Δ†"]
    else:
        # Simulator run: show MILP + QAOA-p2 + SQA + any D-Wave annealing solver present
        headers = ["Hour", "Classical (Baseline)", "QAOA-p2", "QAOA Δ†", "SQA", "SQA Δ†"]
        for s in extra_solvers:
            lbl = SOLVER_LABELS.get(s, s)
            headers += [lbl, f"{lbl} Δ†"]

    cells       = []
    cell_colors = []

    if has_ibm:
        ibm_solvers = [s for s in solvers if s.startswith("ibm_")]
        ibm_solver = ibm_solvers[0] if ibm_solvers else None
        qa_costs = np.full(len(hours), np.nan)  # not used
        sq_costs = np.full(len(hours), np.nan)  # not used
    else:
        qa_costs  = cost_by.get("qaoa_p2", np.full(len(hours), np.nan))
        sq_costs  = cost_by.get("sqa",     np.full(len(hours), np.nan))
        ibm_costs = np.full(len(hours), np.nan)  # not used
        ibm_solver = None

    for i, h in enumerate(hours):
        cl_c = cl_costs[i]

        cl_s = f"{cl_c:.4f}" if not np.isnan(cl_c) else "-"

        if has_ibm:
            row = [str(h), cl_s]
            ibm_infeas = False
            for ibm_s_name in ibm_solvers:
                ibm_c = cost_by.get(ibm_s_name, np.full(len(hours), np.nan))[i]
                ibm_s_str = f"{ibm_c:.4f}" if not np.isnan(ibm_c) else "-"
                if not np.isnan(cl_c) and abs(cl_c) > 1e-9 and not np.isnan(ibm_c):
                    ibm_gap = f"{100*(ibm_c - cl_c)/abs(cl_c):+.1f}%"
                else:
                    ibm_gap = "-"
                row += [ibm_s_str, ibm_gap]
                if h in infeas_by_solver.get(ibm_s_name, set()):
                    ibm_infeas = True
            extra_infeas = False
            for s in extra_solvers:
                v_c = cost_by.get(s, np.full(len(hours), np.nan))[i]
                v_s = f"{v_c:.4f}" if not np.isnan(v_c) else "-"
                if not np.isnan(cl_c) and abs(cl_c) > 1e-9 and not np.isnan(v_c):
                    v_g = f"{100*(v_c - cl_c)/abs(cl_c):+.1f}%"
                else:
                    v_g = "-"
                row += [v_s, v_g]
                if h in infeas_by_solver.get(s, set()):
                    extra_infeas = True
            cells.append(row)
            ibm_infeas = ibm_infeas or extra_infeas
            is_peak = (not np.isnan(cl_c)) and (cl_c >= threshold)
        else:
            qa_c = qa_costs[i]
            sq_c = sq_costs[i]
            qa_s = f"{qa_c:.4f}" if not np.isnan(qa_c) else "-"
            sq_s = f"{sq_c:.4f}" if not np.isnan(sq_c) else "-"

            if not np.isnan(cl_c) and abs(cl_c) > 1e-9 and not np.isnan(qa_c):
                qa_gap = f"{100*(qa_c - cl_c)/abs(cl_c):+.1f}%"
            else:
                qa_gap = "-"
            if not np.isnan(cl_c) and abs(cl_c) > 1e-9 and not np.isnan(sq_c):
                sq_gap = f"{100*(sq_c - cl_c)/abs(cl_c):+.1f}%"
            else:
                sq_gap = "-"

            row = [str(h), cl_s, qa_s, qa_gap, sq_s, sq_gap]
            extra_infeas = False
            for s in extra_solvers:
                v_c = cost_by.get(s, np.full(len(hours), np.nan))[i]
                v_s = f"{v_c:.4f}" if not np.isnan(v_c) else "-"
                if not np.isnan(cl_c) and abs(cl_c) > 1e-9 and not np.isnan(v_c):
                    v_g = f"{100*(v_c - cl_c)/abs(cl_c):+.1f}%"
                else:
                    v_g = "-"
                row += [v_s, v_g]
                if h in infeas_by_solver.get(s, set()):
                    extra_infeas = True
            cells.append(row)
            is_peak = (not np.isnan(cl_c)) and (cl_c >= threshold)
            qa_infeas = h in infeas_by_solver.get("qaoa_p2", set())
            sq_infeas = h in infeas_by_solver.get("sqa", set()) or extra_infeas

        if is_peak:
            row_bg = "#FFF2CC"  # peak hour
        elif has_ibm and ibm_infeas:
            row_bg = "#FFE6E6"  # infeasible hour (IBM)
        elif not has_ibm and (qa_infeas or sq_infeas):
            row_bg = "#FFE6E6"  # infeasible hour (simulator)
        else:
            row_bg = "white"

        cell_colors.append([row_bg] * len(headers))

    n_rows = len(cells)
    fig, ax = plt.subplots(figsize=(14, 0.32 * n_rows + 0.5))
    ax.axis("off")

    # Detect infeasible hours for caption
    infeas_caption = ""
    if has_ibm:
        ibm_infeas_hours = infeas_by_solver.get(ibm_solver, set()) if ibm_solver else set()
        if ibm_infeas_hours:
            infeas_sorted = sorted(ibm_infeas_hours)
            ibm_label = SOLVER_LABELS.get(ibm_solver, ibm_solver)
            infeas_caption = f"\n(light red = hours {', '.join(map(str, infeas_sorted))} infeasible for {ibm_label})"
    else:
        all_infeas = set()
        sim_solvers_list = ["qaoa_p2", "sqa"] + extra_solvers
        for solver in sim_solvers_list:
            all_infeas.update(infeas_by_solver.get(solver, set()))
        if all_infeas:
            infeas_sorted = sorted(all_infeas)
            label_list = "/".join(["QAOA", "SQA"] + [SOLVER_LABELS.get(s, s) for s in extra_solvers])
            infeas_caption = f"\n(light red = hours {', '.join(map(str, infeas_sorted))} infeasible for {label_list})"

    ax.set_title(f"Hourly Cost Details - {date}\n"
                 f"(yellow = peak-price hours, top 20% MILP cost){infeas_caption}\n"
                 f"† Δ reflects independent SOC trajectories per solver; per-hour gaps are scheduling "
                 f"timing differences, not solver quality. Use total daily cost for comparison.",
                 fontsize=9, pad=2, weight="bold")

    table = ax.table(cellText=cells, colLabels=headers, cellColours=cell_colors,
                     cellLoc="center", colLoc="center", loc="upper center",
                     bbox=[0, 0, 1, 0.93])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.0)
    for k, cell in table.get_celld().items():
        if k[0] == 0:
            cell.set_facecolor("#1f3a68")
            cell.set_text_props(color="white", weight="bold")
        cell.set_edgecolor("#bbb")
    _save(fig, out_dir / "fig2_hourly_operating_cost.png")


def plot_shed_hourly(df_sched: pd.DataFrame, date: str, out_dir: Path, mg=None, backend_filter: Optional[str] = None) -> None:
    """Fig 9: hourly load shed (pu·h) - 3 solvers grouped bars. Shed = max(0, load - supply)."""
    if mg is None:
        from microgrid.config import make_mg
        mg = make_mg()
    _style()
    fig, ax = plt.subplots(figsize=(12, 5))

    # Get scale from metadata
    import json
    meta_path = RESULTS_DIR / "by_date" / date / "tables" / "metadata.json"
    scale_mw = float(json.loads(meta_path.read_text()).get("scale_mw", 1.0)) if meta_path.exists() else 1.0

    solvers = _active_solvers(df_sched, backend_filter=backend_filter)
    hours = sorted(df_sched["hour"].unique())
    n_h = len(hours)
    n_s = len(solvers)
    bar_w = 0.8 / max(n_s, 1)
    x = _hours_display(hours).astype(float)

    totals = {}
    for i, solver in enumerate(solvers):
        sub = df_sched[df_sched["solver"] == solver].sort_values(["hour", "substep"])
        shed_per_hour = np.zeros(n_h)
        for hi, h in enumerate(hours):
            rows = sub[sub["hour"] == h]
            if len(rows) == 0: continue
            load_pu  = rows["load_mw"].values  / scale_mw
            solar_pu = rows["solar_mw"].values / scale_mw
            wind_pu  = rows["wind_mw"].values  / scale_mw
            grid_pu  = rows["grid"].values
            chg      = rows["charge"].values
            dis      = rows["discharge"].values
            supply   = mg.P_G * grid_pu + mg.P_B * (dis - chg) + solar_pu + wind_pu
            shed_t   = np.maximum(0.0, load_pu - supply)
            shed_per_hour[hi] = float(shed_t.sum())
        totals[solver] = float(shed_per_hour.sum())
        offset = (i - (n_s - 1) / 2) * bar_w
        ax.bar(x + offset, shed_per_hour, bar_w,
               color=SOLVER_COLORS.get(solver, "#444"),
               label=f"{SOLVER_LABELS.get(solver, solver)} (Σ={totals[solver]:.3f} pu)",
               edgecolor="black", linewidth=0.4, alpha=0.85)

    ax.set_title(f"Hourly Load Shed - {date}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Hour of Day", fontsize=11)
    ax.set_ylabel("Shed (pu·step, summed over 4 substeps)", fontsize=11)
    _set_hour_axis(ax, hours)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out_dir / "fig9_hourly_shed.png")


# Per-date convenience wrapper

def generate_date_figures(
    date: str,
    df_cmp: pd.DataFrame,
    df_sched: pd.DataFrame,
    df_rob_agg: Optional[pd.DataFrame] = None,
    mg=None,
    fig_dir=None,
    backend_filter: Optional[str] = None,
) -> None:
    if fig_dir is None:
        fig_dir = RESULTS_DIR / "by_date" / date / "figures"
    from pathlib import Path
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # If IBM backend detected in df_cmp metadata or folder name, rename qaoa_p* solvers to ibm_p
    import json
    metadata_path = fig_dir.parent / "tables" / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            meta = json.load(f)
        variant = meta.get("variant", "")
        if "ibm_" in variant:
            # Extract p-layer from variant (e.g., "ibm_kingston_p2" -> p=2)
            p_layers = [s for s in variant.split("_") if s.startswith("p")]
            if p_layers:
                p_val = p_layers[0][1:]  # strip "p"
                old_solver = f"qaoa_p{p_val}"
                new_solver = f"ibm_p{p_val}"
                if old_solver in df_cmp["solver"].values:
                    df_cmp = df_cmp.copy()
                    df_cmp["solver"] = df_cmp["solver"].replace(old_solver, new_solver)
                if old_solver in df_sched["solver"].values:
                    df_sched = df_sched.copy()
                    df_sched["solver"] = df_sched["solver"].replace(old_solver, new_solver)

    # Detect backend: if any ibm_* solver present, filter to IBM only; otherwise filter to simulator only
    if backend_filter is None:
        ibm_solvers = [s for s in df_cmp["solver"].unique() if s.startswith("ibm_")]
        backend_filter = "ibm_kingston" if ibm_solvers else "simulator"

    plot_hourly_cost(df_cmp,                    date, fig_dir, backend_filter=backend_filter)
    plot_battery_cd(df_sched,                   date, fig_dir, df_cmp=df_cmp, backend_filter=backend_filter)
    plot_soc(df_sched,                          date, fig_dir, df_cmp=df_cmp, backend_filter=backend_filter)
    plot_grid_import(df_sched,                  date, fig_dir, df_cmp=df_cmp, backend_filter=backend_filter)
    plot_net_load(df_sched,                     date, fig_dir, mg=mg, df_cmp=df_cmp, backend_filter=backend_filter)
    plot_cost_breakdown(df_sched, df_cmp, date, fig_dir, mg=mg, backend_filter=backend_filter)
    plot_solver_summary(df_cmp,        date, fig_dir, backend_filter=backend_filter)
    plot_hourly_results_table(df_cmp,  date, fig_dir, backend_filter=backend_filter)
    plot_shed_hourly(df_sched,         date, fig_dir, mg=mg, backend_filter=backend_filter)


# Combined figures (no LP, 15-min scheduling titles)

def _load_date_summary(dates: List[str]) -> pd.DataFrame:
    rows = []
    for d in dates:
        p = RESULTS_DIR / "by_date" / d / "tables" / "df_compare.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        cap_col = "cap_constrained" in df.columns
        # Total cost gap vs MILP baseline (daily total, not mean of per-hour gaps).
        # Per-hour gaps are invalid for cross-solver comparison due to independent SOC paths.
        cl_sub   = df[df["solver"] == "milp"]
        cl_total = float(cl_sub["cost"].sum()) if not cl_sub.empty else float("nan")
        for solver in df["solver"].unique():
            sub       = df[df["solver"] == solver]
            cap_hrs   = int(sub["cap_constrained"].sum()) if cap_col else 0
            tot       = float(sub["cost"].sum())
            if solver == "milp" or np.isnan(cl_total) or abs(cl_total) < 1e-9:
                total_gap_pct = 0.0
            else:
                total_gap_pct = 100.0 * (tot - cl_total) / abs(cl_total)
            rows.append(dict(
                date=d, solver=solver,
                total_cost=tot,
                total_gap_pct=total_gap_pct,   # daily total cost gap - valid cross-solver metric
                feasible_hrs=int(sub["is_feasible"].sum()),
                total_hrs=len(sub),
                cap_constrained_hrs=cap_hrs,
            ))
    return pd.DataFrame(rows)


def plot_combined_daily_cost(dates: List[str], out_dir: Path) -> None:
    """Bar chart: total daily cost per date, grouped by solver - no LP."""
    _style()
    summary = _load_date_summary(dates)
    if summary.empty:
        return

    solvers = [s for s in SOLVERS_ORDERED if s in summary["solver"].values]
    x = np.arange(len(dates))
    width = 0.8 / len(solvers)

    fig, ax = plt.subplots(figsize=(max(10, len(dates) * 2), 5))
    for i, solver in enumerate(solvers):
        sub  = summary[summary["solver"] == solver].set_index("date")
        vals = [float(sub.loc[d, "total_cost"]) if d in sub.index else np.nan for d in dates]
        ax.bar(x + (i - len(solvers) / 2 + 0.5) * width, vals, width,
               label=SOLVER_LABELS.get(solver, solver),
               color=SOLVER_COLORS.get(solver, "#888"), alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(dates, rotation=30, ha="right")
    ax.set_title("Total Daily Operating Cost Across Study Dates - 15-min scheduling")
    ax.set_ylabel("Total Cost (pu·h)")
    ax.legend()
    _save(fig, out_dir / "figA_daily_cost.png")


def plot_combined_gap(dates: List[str], out_dir: Path) -> None:
    """Bar chart: total daily cost gap % vs MILP per date."""
    _style()
    summary = _load_date_summary(dates)
    if summary.empty:
        return

    qubo_solvers = [s for s in SOLVERS_ORDERED if s != "milp" and s in summary["solver"].values]
    x = np.arange(len(dates))
    width = 0.8 / max(len(qubo_solvers), 1)

    fig, ax = plt.subplots(figsize=(max(10, len(dates) * 2), 5))
    for i, solver in enumerate(qubo_solvers):
        sub  = summary[summary["solver"] == solver].set_index("date")
        vals = [float(sub.loc[d, "total_gap_pct"]) if d in sub.index else np.nan for d in dates]
        ax.bar(x + (i - len(qubo_solvers) / 2 + 0.5) * width, vals, width,
               label=SOLVER_LABELS.get(solver, solver),
               color=SOLVER_COLORS.get(solver, "#888"), alpha=0.85, edgecolor="white")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(dates, rotation=30, ha="right")
    ax.set_title("Cost Gap vs MILP (%) Across Study Dates - 15-min scheduling")
    ax.set_ylabel("Total Cost Gap vs MILP (%)")
    ax.legend()
    _save(fig, out_dir / "figB_gap_vs_milp.png")


def plot_summary_table(dates: List[str], out_dir: Path) -> None:
    """figS_summary_table: per-date costs + gap% for all solvers incl IBM p2/p3."""
    _style()
    rows = []
    for d in dates:
        p_base = RESULTS_DIR / "by_date" / d / "tables" / "df_compare.csv"
        p_ibm2 = RESULTS_DIR / "by_date" / f"{d}_ibm_kingston_p2" / "tables" / "df_compare.csv"
        p_ibm3 = RESULTS_DIR / "by_date" / f"{d}_ibm_kingston_p3" / "tables" / "df_compare.csv"
        if not p_base.exists():
            continue
        df      = pd.read_csv(p_base)
        season  = _SEASON_FOR_DATE.get(d, "-")
        r       = {"Date": d, "Season": season}

        ml = float(df[df["solver"] == "milp"]["cost"].sum())
        qa = float(df[df["solver"] == "qaoa_p2"]["cost"].sum()) if "qaoa_p2" in df["solver"].values else float("nan")
        sq = float(df[df["solver"] == "sqa"]["cost"].sum())     if "sqa"     in df["solver"].values else float("nan")

        ibm2 = float("nan")
        if p_ibm2.exists():
            df2  = pd.read_csv(p_ibm2)
            ibm2 = float(df2[df2["solver"] == "qaoa_p2"]["cost"].sum()) if "qaoa_p2" in df2["solver"].values else float("nan")

        ibm3 = float("nan")
        if p_ibm3.exists():
            df3  = pd.read_csv(p_ibm3)
            ibm3 = float(df3[df3["solver"] == "qaoa_p3"]["cost"].sum()) if "qaoa_p3" in df3["solver"].values else float("nan")

        def _gap(v):
            if np.isnan(v) or ml <= 0:
                return "-"
            return f"{(v - ml) / ml * 100:+.2f}%"

        r["MILP"]           = f"{ml:.2f}"
        r["SQA"]            = f"{sq:.2f}"
        r["SQA Gap(%)"]     = _gap(sq)
        r["QAOA-Sim(p=2)"]  = f"{qa:.2f}"
        r["Sim Gap(%)"]     = _gap(qa)
        r["IBM p=2"]        = f"{ibm2:.2f}" if not np.isnan(ibm2) else "-"
        r["IBM p=2 Gap(%)"] = _gap(ibm2)
        r["IBM p=3"]        = f"{ibm3:.2f}" if not np.isnan(ibm3) else "-"
        r["IBM p=3 Gap(%)"] = _gap(ibm3)
        rows.append(r)

    if not rows:
        return

    cols = ["Date", "Season", "MILP",
            "SQA", "SQA Gap(%)",
            "QAOA-Sim(p=2)", "Sim Gap(%)",
            "IBM p=2", "IBM p=2 Gap(%)",
            "IBM p=3", "IBM p=3 Gap(%)"]
    data = [[r[c] for c in cols] for r in rows]

    fig, ax = plt.subplots(figsize=(18, 0.6 + 0.50 * (len(rows) + 1)))
    ax.axis("off")
    tbl = ax.table(cellText=data, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.6)

    n_cols = len(cols)
    # Header
    for j in range(n_cols):
        cell = tbl[(0, j)]
        cell.set_facecolor("#1f3a68")
        cell.set_text_props(color="white", weight="bold")
        cell.set_edgecolor("#0d2244")

    # Gap columns indices for colour coding
    gap_cols = {cols.index(c) for c in ["SQA Gap(%)", "Sim Gap(%)", "IBM p=2 Gap(%)", "IBM p=3 Gap(%)"]}

    for i in range(1, len(rows) + 1):
        for j in range(n_cols):
            cell = tbl[(i, j)]
            cell.set_edgecolor("#bbb")
            if j in gap_cols:
                val = data[i - 1][j]
                try:
                    pct = float(val.replace("%", "").replace("+", ""))
                    if pct > 20:
                        cell.set_facecolor("#ffd5d5")
                    elif pct > 10:
                        cell.set_facecolor("#fff3cd")
                    else:
                        cell.set_facecolor("#d4edda")
                except (ValueError, AttributeError):
                    cell.set_facecolor("#f9f9f9")
            else:
                cell.set_facecolor("#f9f9f9" if i % 2 == 0 else "white")

    ax.set_title("Per-Date Total Cost & Gap vs MILP - All Solvers", fontsize=13, fontweight="bold", pad=12)
    fig.text(0.5, 0.01,
             "Gap = (solver cost − MILP) / MILP × 100%.  "
             "Green <10%, yellow 10-20%, red >20%.  "
             "QAOA-Sim, IBM p=2, IBM p=3 committed identical dispatch solutions.",
             ha="center", fontsize=8, style="italic", color="#555")
    _save(fig, out_dir / "figS_summary_table.png")


def export_combined_tables(dates: List[str], out_dir: Path) -> None:
    """Export table1-table4 CSVs matching paper tables."""
    import json
    tables_dir = out_dir.parent / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # table1: overall summary
    t1_rows = []
    for d in dates:
        p = RESULTS_DIR / "by_date" / d / "tables" / "df_compare.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        season = _SEASON_FOR_DATE.get(d, "Unknown")
        regime = "Constrained" if d == "2018-10-26" else "Moderate"

        row = {"Date": d, "Regime": regime, "Season": season}
        best_cost, best_solver = float("inf"), "-"
        for solver in ("milp", "qaoa_p2", "sqa"):
            sub = df[df["solver"] == solver]
            if sub.empty:
                continue
            tc   = float(sub["cost"].sum())
            feas = int(sub["is_feasible"].sum())
            key  = {"milp": "milp", "qaoa_p2": "qaoa_p2", "sqa": "sqa"}[solver]
            row[f"TotalCost_{key}"]  = round(tc, 4)
            row[f"FeasHours_{key}"]  = feas
            if tc < best_cost:
                best_cost, best_solver = tc, solver
        row["Best_Solver"] = best_solver
        t1_rows.append(row)
    if t1_rows:
        pd.DataFrame(t1_rows).to_csv(tables_dir / "table1_overall_summary.csv", index=False)

    # table2: gap vs MILP baseline
    t2_rows = []
    for d in dates:
        p = RESULTS_DIR / "by_date" / d / "tables" / "df_compare.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        cl_total = float(df[df["solver"] == "milp"]["cost"].sum())
        qa_total = float(df[df["solver"] == "qaoa_p2"]["cost"].sum()) if "qaoa_p2" in df["solver"].values else float("nan")
        sq_total = float(df[df["solver"] == "sqa"]["cost"].sum())     if "sqa"     in df["solver"].values else float("nan")
        qa_gap   = 100.0 * (qa_total - cl_total) / abs(cl_total) if not np.isnan(qa_total) and abs(cl_total) > 1e-9 else float("nan")
        sq_gap   = 100.0 * (sq_total - cl_total) / abs(cl_total) if not np.isnan(sq_total) and abs(cl_total) > 1e-9 else float("nan")
        t2_rows.append({"Date": d,
                         "Classical_TotalCost": round(cl_total, 4),
                         "QAOA_TotalCost":      round(qa_total, 4),
                         "SQA_TotalCost":       round(sq_total, 4),
                         "QAOA_Gap_vs_Classical_%": round(qa_gap, 3),
                         "SQA_Gap_vs_Classical_%":  round(sq_gap, 3)})
    if t2_rows:
        pd.DataFrame(t2_rows).to_csv(tables_dir / "table2_gap_vs_milp.csv", index=False)

    # table3: key hours
    t3_rows = []
    THRESHOLD = 0.46
    for d in dates:
        p = RESULTS_DIR / "by_date" / d / "tables" / "df_compare.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        hours = sorted(df["hour"].unique())
        cl_s  = df[df["solver"] == "milp"].set_index("hour")
        qa_s  = df[df["solver"] == "qaoa_p2"].set_index("hour")  if "qaoa_p2" in df["solver"].values else pd.DataFrame()
        sq_s  = df[df["solver"] == "sqa"].set_index("hour")      if "sqa"     in df["solver"].values else pd.DataFrame()
        for h in hours:
            cl_c = float(cl_s.loc[h, "cost"]) if h in cl_s.index else float("nan")
            qa_c = float(qa_s.loc[h, "cost"]) if (not qa_s.empty and h in qa_s.index) else float("nan")
            sq_c = float(sq_s.loc[h, "cost"]) if (not sq_s.empty and h in sq_s.index) else float("nan")
            qa_diff = qa_c - cl_c if not (np.isnan(qa_c) or np.isnan(cl_c)) else float("nan")
            if np.isnan(qa_diff) or abs(qa_diff) < THRESHOLD:
                continue
            obs = "QAOA significantly worse" if qa_diff > 0 else "QAOA outperforms Classical"
            t3_rows.append({"Date": d, "Hour": h,
                             "Classical": round(cl_c, 6),
                             "QAOA":      round(qa_c, 6),
                             "SQA":       round(sq_c, 6),
                             "QAOA_diff": round(qa_diff, 6),
                             "Observation": obs})
    if t3_rows:
        df3 = pd.DataFrame(t3_rows).sort_values("QAOA_diff", ascending=False)
        df3.to_csv(tables_dir / "table3_key_hours.csv", index=False)

    # table4: dataset characteristics
    t4_rows = []
    for d in dates:
        meta_path  = RESULTS_DIR / "by_date" / d / "tables" / "metadata.json"
        sched_path = RESULTS_DIR / "by_date" / d / "tables" / "df_schedule.csv"
        row = {"Date": d, "Season": _SEASON_FOR_DATE.get(d, "Unknown")}
        meta: dict = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
        row["Peak_Hour"] = int(meta.get("peak_hour", 0))
        scale = float(meta.get("scale_mw", 1.0))
        row["Peak_Load_MW"] = round(float(meta.get("peak_load_mw", scale)), 1)
        if sched_path.exists():
            df = pd.read_csv(sched_path)
            fst = [s for s in SOLVERS_ORDERED if s in df["solver"].values]
            if fst:
                sub = df[df["solver"] == fst[0]]
                if "substep" in sub.columns:
                    sub = sub[sub["substep"] == 0]
                mean_load  = float(sub["load"].mean())  if "load"  in sub.columns else float("nan")
                mean_wind  = float(sub["wind"].mean())  if "wind"  in sub.columns else float("nan")
                mean_solar = float(sub["solar"].mean()) if "solar" in sub.columns else float("nan")
                row["Mean_Load_MW"]        = round(mean_load * scale, 1)
                row["Wind_Penetration"]    = round(mean_wind,  3)
                row["Solar_Penetration"]   = round(mean_solar, 3)
                row["Renewable_Penetration"] = round(mean_wind + mean_solar, 3)
                if "load" in sub.columns and "wind" in sub.columns and "solar" in sub.columns:
                    net = (sub["load"] - sub["wind"] - sub["solar"]) * scale
                    row["Net_Load_Min_MW"]       = round(float(net.min()), 1)
                    row["Renewable_Variability"] = round(float((sub["wind"] + sub["solar"]).std()), 3)
        regime = "Constrained" if d == "2018-10-26" else "Moderate"
        char_parts = []
        if not np.isnan(row.get("Wind_Penetration", float("nan"))):
            wp = row["Wind_Penetration"]
            char_parts.append("very high wind" if wp > 0.45 else ("high wind" if wp > 0.3 else ("moderate wind" if wp > 0.1 else "low wind" if wp > 0.05 else "near-zero wind")))
        char_parts.append(row["Season"].lower())
        peak = row.get("Peak_Load_MW", 0)
        char_parts.append("high load" if peak > 60000 else "moderate load")
        row["Character"] = ", ".join(char_parts)
        row["Regime"]    = regime
        t4_rows.append(row)
    if t4_rows:
        pd.DataFrame(t4_rows).to_csv(tables_dir / "table4_dataset_characteristics.csv", index=False)


def generate_combined_figures(dates: Optional[List[str]] = None) -> None:
    from microgrid.config import STUDY_DATES
    if dates is None:
        dates = STUDY_DATES
    out_dir = RESULTS_DIR / "combined" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_combined_daily_cost(dates, out_dir)
    plot_combined_gap(dates, out_dir)
    plot_summary_table(dates, out_dir)
    export_combined_tables(dates, out_dir)
