from __future__ import annotations
import time
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

from microgrid.config import (
    N_STEPS, SCHEDULING_HORIZON, APPLY_STEPS, SOC_INIT, DT,
    QAOA_P_SWEEP, PRICE_VEC, BITS_PER_STEP, GRID_COEFFS,
    SOC_TOL, MicrogridParams, MILP_LAM_TRACK, BQM_BASELINE_READS,
)
from microgrid.utils import decode, soc_trajectory, op_cost
from microgrid.feasibility import check_feasibility
from microgrid.preprocessing import ProcessedDay
from microgrid.scenario_generation import ScenarioList, scenarios_to_tuples
from microgrid.optimization_milp import milp_schedule_window, milp_daily
from microgrid.optimization_qubo import build_qubo
from microgrid.optimization_qaoa import run_qaoa_window, _repair as _qaoa_repair, _repair_pw as _qaoa_repair_pw, _repair_sqa_staged as _sqa_repair_staged
from microgrid.optimization_sqa  import run_sqa_window
from microgrid.optimization_exact import run_exact_window
from microgrid.optimization_bqm   import run_bqm_window

logger = logging.getLogger(__name__)


@dataclass
class ScheduleResult:
    date:           str
    df_compare:     pd.DataFrame
    df_schedule:    pd.DataFrame
    df_diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)
    metadata:       Dict[str, Any] = field(default_factory=dict)


# All valid discrete grid fractions from 3-bit thermometer encoding
_GRID_LEVELS = sorted(set(
    b0 * GRID_COEFFS[0] + b1 * GRID_COEFFS[1] + b2 * GRID_COEFFS[2]
    for b0 in (0, 1) for b1 in (0, 1) for b2 in (0, 1)
))
_GRID_LEVELS_ARR = np.array(_GRID_LEVELS)  # [0, 1/7, 2/7, 3/7, 4/7, 5/7, 6/7]


def _snap_grid_up(g, c, d, load_vec, renew_vec, mg, soc0: float):
    """Cover supply deficit: snap grid up first; only use battery if grid maxed. Also snap down on surplus."""
    g = g.copy()
    c = c.copy()
    d = d.copy()
    T = len(g)
    step_d = (mg.P_B / mg.SOC_CAP) * DT / mg.eta_d
    step_c = (mg.P_B / mg.SOC_CAP) * DT * mg.eta_c
    soc = soc0
    for t in range(T):
        supply  = mg.P_G * g[t] + mg.P_B * (d[t] - c[t]) + renew_vec[t]
        deficit = load_vec[t] - supply
        surplus = supply - load_vec[t]

        if deficit > 1e-4:
            g_old = g[t]
            # Fix 1: snap grid up
            needed_g = g[t] + deficit / mg.P_G
            candidates = _GRID_LEVELS_ARR[_GRID_LEVELS_ARR >= needed_g - 1e-9]
            g[t] = float(candidates[0]) if len(candidates) > 0 else _GRID_LEVELS_ARR[-1]
            supply  = mg.P_G * g[t] + mg.P_B * (d[t] - c[t]) + renew_vec[t]
            deficit = load_vec[t] - supply
            logger.debug("snap_up  t=%d: g %.4f->%.4f deficit_before=%.1f deficit_after=%.1f",
                         t, g_old, g[t], load_vec[t]-mg.P_G*g_old-mg.P_B*(d[t]-c[t])-renew_vec[t], deficit)
            # Fix 2: grid maxed and still short - discharge battery
            if deficit > 1e-4 and d[t] < 0.5 and c[t] < 0.5 and soc > mg.soc_min + SOC_TOL:
                d[t] = 1.0
                logger.debug("snap_up  t=%d: added discharge to cover remaining deficit=%.1f", t, deficit)

        elif surplus > mg.P_B + 1e-4:
            g_old = g[t]
            # Fix 3: snap grid down - keep P_B headroom for charging
            ideal_g = max(g[t] - (surplus - mg.P_B) / mg.P_G, 0.0)
            candidates = _GRID_LEVELS_ARR[_GRID_LEVELS_ARR <= ideal_g + 1e-9]
            g[t] = float(candidates[-1]) if len(candidates) > 0 else _GRID_LEVELS_ARR[0]
            supply = mg.P_G * g[t] + mg.P_B * (d[t] - c[t]) + renew_vec[t]
            logger.debug("snap_dn  t=%d: g %.4f->%.4f surplus_before=%.1f surplus_after=%.1f",
                         t, g_old, g[t], surplus, supply - load_vec[t])
            # Fix 4: still oversupplied and discharging - turn off discharge
            if supply > load_vec[t] + 1e-4 and d[t] > 0.5:
                d[t] = 0.0
                logger.debug("snap_dn  t=%d: turned off discharge (oversupplied)", t)

        # Fix 5: discharging near SOC floor - switch to grid
        if d[t] > 0.5 and soc <= mg.soc_min + 2 * SOC_TOL:
            d[t] = 0.0
            needed_g = g[t] + mg.P_B / mg.P_G
            candidates = _GRID_LEVELS_ARR[_GRID_LEVELS_ARR >= needed_g - 1e-9]
            g[t] = float(candidates[0]) if len(candidates) > 0 else _GRID_LEVELS_ARR[-1]
            logger.debug("snap_soc t=%d: SOC floor - replaced discharge with grid %.4f", t, g[t])

        # Fix 6: charging at SOC max - turn off
        if c[t] > 0.5 and soc >= mg.soc_max - SOC_TOL:
            c[t] = 0.0
            logger.debug("snap_soc t=%d: SOC max - turned off charge", t)

        # Advance SOC
        if c[t] > 0.5:
            soc = min(soc + step_c, mg.soc_max)
        elif d[t] > 0.5:
            soc = max(soc - step_d, mg.soc_min)
    return g, c, d


def _physics_guard(g, c, d, soc: float, mg: MicrogridParams, apply_t: int):
    g = np.array(g, dtype=float).copy()
    c = np.array(c, dtype=float).copy()
    d = np.array(d, dtype=float).copy()
    step_d = (mg.P_B / mg.SOC_CAP) * DT / mg.eta_d
    step_c = (mg.P_B / mg.SOC_CAP) * DT * mg.eta_c
    for t in range(apply_t):
        if d[t] > 0.5 and soc <= mg.soc_min + SOC_TOL:
            d[t] = 0.0; g[t] = min(g[t] + mg.P_B / mg.P_G, 1.0)
        if c[t] > 0.5 and soc >= mg.soc_max - SOC_TOL:
            c[t] = 0.0
        if c[t] > 0.5:
            soc = min(soc + step_c, mg.soc_max)
        elif d[t] > 0.5:
            soc = max(soc - step_d, mg.soc_min)
    return g, c, d


def _bits_str(x: np.ndarray, T_act: int) -> str:
    width = T_act * BITS_PER_STEP
    raw = "".join(str(int(b)) for b in x.astype(int).ravel())[:width].zfill(width)
    return "b" + raw


def _encode_bits(g_arr, c_arr, d_arr, T: int) -> np.ndarray:
    x = np.zeros(T * BITS_PER_STEP, dtype=np.float64)
    for t in range(T):
        base = t * BITS_PER_STEP
        k    = max(0, min(7, int(round(float(g_arr[t]) * 7))))
        x[base]     = (k >> 0) & 1
        x[base + 1] = (k >> 1) & 1
        x[base + 2] = (k >> 2) & 1
        x[base + 3] = round(float(c_arr[t]))
        x[base + 4] = round(float(d_arr[t]))
    return x


def _gap(cost: float, ref: float) -> Tuple[float, float]:
    abs_diff = cost - ref
    if abs(ref) < 1e-9:
        return abs_diff, float("nan")
    pct = 100.0 * abs_diff / abs(ref)
    return abs_diff, float(np.clip(pct, -200.0, 200.0))


def _sched_rows(hour, apply_t, global_step_start,
                wl, w_solar, w_wind, g, c, d, soc_arr, mg, scale, solver):
    return [
        dict(
            hour=hour, substep=t, global_step=global_step_start + t, solver=solver,
            load_mw=float(wl[t]) * scale, solar_mw=float(w_solar[t]) * scale,
            wind_mw=float(w_wind[t]) * scale,
            grid=round(float(g[t]), 4), charge=int(round(float(c[t]))),
            discharge=int(round(float(d[t]))),
            grid_mw=float(g[t]) * mg.P_G * scale,
            batt_mw=(float(d[t]) - float(c[t])) * mg.P_B * scale,
            soc=float(soc_arr[t]),
        )
        for t in range(apply_t)
    ]


def _post_guard_feasible(g, c, d, T_act, wl, wsc, mg, soc0) -> bool:
    x_g = _encode_bits(g, c, d, T_act)
    rep = check_feasibility(x_g, T_act, wl, wsc, mg, soc0=soc0, eval_t=min(APPLY_STEPS, T_act))
    return bool(rep["expected_feasible"])


def _milp_cost(res: Dict, apply_t: int, load_vec, mg, price_vec) -> float:
    """Compute op_cost from MILP result arrays (grid + cycle + shed)."""
    cost = 0.0
    s_arr = res.get("s", np.zeros(apply_t))
    for t in range(apply_t):
        pr = float(price_vec[t])
        cost += DT * pr * (
            mg.C_grid  * float(res["g"][t])
            + mg.C_cycle * mg.P_B * (float(res["u_c"][t]) + float(res["u_d"][t]))
            + mg.C_shed  * float(s_arr[t])
        )
    return cost


def _safe_fallback_quantum(
    T_act, wl, wsc, mg, soc0, price_win, soc_ref,
    primary_seed, min_soc_end, battery_mode, target_soc,
):
    """Repair-based fallback for QAOA/SQA.

    Does not copy bits or costs from the MILP baseline. Priority order:
    primary_seed -> dynamic seeds -> standard physics-aware seeds.
    """
    n_q = T_act * BITS_PER_STEP
    price_mean = float(price_win[:APPLY_STEPS].mean())
    is_peak_price = price_mean >= 1.4

    def _soc_ok(c_arr, d_arr):
        soc_end = np.clip(
            soc_trajectory(c_arr[:APPLY_STEPS], d_arr[:APPLY_STEPS], mg, soc0),
            mg.soc_min, mg.soc_max)[-1]
        floor = min_soc_end if min_soc_end is not None else mg.soc_min
        return float(soc_end) >= floor

    def _try_repair(seed, label):
        for repair_fn in (_qaoa_repair, _qaoa_repair_pw):
            x = repair_fn(seed.copy(), T_act, wl, wsc, mg, soc0, price_win,
                          battery_mode=battery_mode, target_soc=target_soc,
                          min_soc_end=min_soc_end)
            g, c, d = decode(x, T_act)
            g, c, d = _physics_guard(g, c, d, soc0, mg, APPLY_STEPS)
            x_pg = _encode_bits(g, c, d, T_act)
            x_pg = _qaoa_repair(x_pg, T_act, wl, wsc, mg, soc0, price_win,
                                battery_mode=battery_mode, target_soc=target_soc,
                                min_soc_end=min_soc_end)
            g, c, d = decode(x_pg, T_act)
            if _post_guard_feasible(g, c, d, T_act, wl, wsc, mg, soc0) and _soc_ok(c, d):
                return g, c, d, {"fb_source": f"repair_{label}"}
        return None

    # 1. Primary seed (solver's own bits)
    if primary_seed is not None:
        result = _try_repair(np.asarray(primary_seed, dtype=np.float64), "primary")
        if result is not None:
            return result

    # 2. Dynamic seeds from load/renewable
    def _make_seed(g_pat, c_v, d_v):
        s = np.zeros(n_q, dtype=np.float64)
        for t in range(T_act):
            s[t * BITS_PER_STEP + 0] = g_pat[0]
            s[t * BITS_PER_STEP + 1] = g_pat[1]
            s[t * BITS_PER_STEP + 2] = g_pat[2]
            s[t * BITS_PER_STEP + 3] = c_v
            s[t * BITS_PER_STEP + 4] = d_v
        return s

    wl_arr  = np.asarray(wl[:T_act], dtype=np.float64)
    avg_ren = np.zeros(T_act)
    for w, sv, wv in wsc:
        avg_ren += float(w) * (np.asarray(sv[:T_act], float) + np.asarray(wv[:T_act], float))
    net_load = wl_arr - avg_ren
    min_grid_pu = np.maximum(net_load / mg.P_G, 0.0)
    mean_min_grid = float(np.clip(np.mean(min_grid_pu[:APPLY_STEPS]), 0.0, 1.0))
    mgb = int(np.round(mean_min_grid * 7))
    min_g0 = (mgb >> 0) & 1; min_g1 = (mgb >> 1) & 1; min_g2 = (mgb >> 2) & 1

    dyn_seeds = [
        ((min_g0, min_g1, min_g2), 0, 0, "dyn_idle"),
        ((min_g0, min_g1, min_g2), 1, 0, "dyn_charge"),
        ((min_g0, min_g1, min_g2), 0, 1, "dyn_discharge"),
    ]
    std_seeds = [
        ((0, 0, 0), 0, 0, "idle"),
        ((1, 1, 1), 0, 1, "grid_full_dis"),
        ((1, 1, 0), 0, 1, "grid3_7_dis"),
        ((0, 1, 0), 1, 0, "grid2_7_chg"),
        ((1, 1, 1), 0, 0, "grid_full"),
        ((1, 0, 1), 0, 0, "grid5_7"),
        ((0, 1, 1), 0, 0, "grid6_7"),
        ((1, 0, 0), 0, 1, "grid1_7_dis"),
    ]
    if is_peak_price:
        peak_seeds = [
            ((1, 1, 1), 0, 1, "peak_full_dis"),
            ((1, 1, 0), 0, 1, "peak_6_7_dis"),
            ((1, 0, 1), 0, 1, "peak_5_7_dis"),
            ((0, 1, 1), 0, 1, "peak_6_7b_dis"),
        ]
        std_seeds = peak_seeds + std_seeds

    for g_pat, cv, dv, label in dyn_seeds + std_seeds:
        seed = _make_seed(g_pat, cv, dv)
        result = _try_repair(seed, label)
        if result is not None:
            return result

    return None


def run_schedule(
    pday: ProcessedDay,
    scenarios: ScenarioList,
    mg: MicrogridParams,
    enable_qaoa: bool = True,
    enable_sqa:  bool = True,
    enable_exact: bool = False,
    bqm_baselines: Optional[List[str]] = None,
    ibm_backend: Optional[str] = None,
    ibm_shots:   int = 4096,
    qaoa_p_layers: Optional[list] = None,
) -> ScheduleResult:
    np.random.seed(42)
    sc = scenarios_to_tuples(scenarios)
    bqm_baselines = list(bqm_baselines or [])

    p_sweep = qaoa_p_layers if qaoa_p_layers is not None else QAOA_P_SWEEP

    SOLVERS = ["milp"]
    if enable_exact:
        SOLVERS.append("exact")
    if enable_qaoa:
        SOLVERS += [f"qaoa_p{p}" for p in p_sweep]
    if enable_sqa:
        SOLVERS.append("sqa")
    SOLVERS += bqm_baselines

    soc_state: Dict[str, float] = {"milp": SOC_INIT}
    for _p in p_sweep:
        soc_state[f"qaoa_p{_p}"] = SOC_INIT
    soc_state["sqa"]   = SOC_INIT
    soc_state["exact"] = SOC_INIT
    for _bl in bqm_baselines:
        soc_state[_bl] = SOC_INIT
    warm: Dict[int, Any] = {p: None for p in p_sweep}

    # MILP-based daily SOC reference (uses binary c/d - tighter than LP)
    try:
        load_fc_96  = np.asarray(pday.load_fc_n, float)[:N_STEPS]
        solar_fc_96 = np.asarray(pday.solar_ec_n, float).mean(axis=0)[:N_STEPS]
        wind_fc_96  = np.asarray(pday.wind_ec_n,  float).mean(axis=0)[:N_STEPS]
        soc_ref_full = milp_daily(
            load_fc_96, solar_fc_96, wind_fc_96, mg,
            price_vec=PRICE_VEC, soc0=SOC_INIT,
        )
        logger.info(
            f"  Daily MILP (forecast) SOC range [{soc_ref_full.min():.3f}, {soc_ref_full.max():.3f}]"
        )
    except Exception as e:
        logger.warning(f"  Daily MILP failed ({e}); no SOC reference")
        soc_ref_full = None

    rows_cmp:   List[Dict] = []
    rows_sched: List[Dict] = []
    rows_diag:  List[Dict] = []
    t_wall = time.time()
    import os as _os
    _max_hours = int(_os.environ.get("MAX_HOURS", "24"))

    # Forecast battery policy (same logic as before, now referenced to MILP daily plan)
    net_fc_96     = np.asarray(load_fc_96, float) - np.asarray(solar_fc_96, float) - np.asarray(wind_fc_96, float)
    net_fc_hourly = net_fc_96.reshape(24, APPLY_STEPS).mean(axis=1)
    net_low_thr   = float(np.percentile(net_fc_hourly, 30))
    net_high_thr  = float(np.percentile(net_fc_hourly, 75))
    net_peak_hr0  = int(np.argmax(net_fc_hourly))
    daily_peak    = float(np.max(net_fc_hourly))
    daily_mean    = float(np.mean(net_fc_hourly))
    daily_stress  = daily_peak / (daily_mean + 1e-9)
    try:
        _month = pd.to_datetime(pday.date).month
    except Exception:
        _month = 6
    is_high_stress = (daily_stress > 1.25) or (_month in (6, 7, 8, 9))
    CHARGE_TARGET, PRESERVE_TARGET = (0.80, 0.70) if is_high_stress else (0.75, 0.65)

    def _battery_policy(hour_1idx: int):
        h = hour_1idx - 1
        if h < net_peak_hr0 and net_fc_hourly[h] <= net_low_thr:
            return 0.55, CHARGE_TARGET, "charge"
        elif h < net_peak_hr0:
            return 0.45, PRESERVE_TARGET, "preserve"
        elif net_fc_hourly[h] >= net_high_thr:
            return 0.15, 0.20, "discharge"
        else:
            return 0.25, 0.40, "recover"

    for step in range(0, N_STEPS, APPLY_STEPS):
        t_start = step
        t_end   = min(t_start + SCHEDULING_HORIZON, N_STEPS)
        T_act   = t_end - t_start
        hour    = step // APPLY_STEPS + 1
        if hour > _max_hours:
            break

        wl       = pday.load_n[t_start:t_end]
        w_solar  = pday.solar_n[t_start:t_end]
        w_wind   = pday.wind_n[t_start:t_end]
        wsc      = [(float(sc[k][0]), sc[k][1][t_start:t_end], sc[k][2][t_start:t_end])
                    for k in range(len(sc))]
        price_win = PRICE_VEC[t_start:t_end]

        milp_soc = soc_state["milp"]
        soc_ref_win = soc_ref_full[t_start:t_end] if soc_ref_full is not None else None

        min_soc_end, soc_reserve_h, batt_mode = _battery_policy(hour)
        if soc_ref_full is not None and t_end - 1 < len(soc_ref_full) and batt_mode != "discharge":
            lp_soc_end  = float(soc_ref_full[t_end - 1])
            min_soc_end = float(max(min_soc_end, lp_soc_end - 0.15))
        if batt_mode == "discharge":
            min_soc_end = max(mg.soc_min - 0.05, min_soc_end * 0.5)
        _max_soc_gain  = mg.P_B * mg.eta_c * APPLY_STEPS * DT / mg.SOC_CAP

        # Per-solver reachable clamp - each solver's min_soc_end bounded by its own soc state.
        # Prevents impossible targets when solver SOCs diverge (e.g. QAOA at floor, MILP cycling).
        _target_min_soc_end = min_soc_end
        def _reach_clamp(soc):
            reach = min(mg.soc_max, float(soc) + _max_soc_gain)
            return min(_target_min_soc_end, reach - 1e-6)

        min_soc_end_milp = _reach_clamp(milp_soc)
        _qa_soc_key = f"qaoa_p{p_sweep[0]}" if p_sweep else "qaoa_p2"
        min_soc_end_qa   = _reach_clamp(soc_state.get(_qa_soc_key, milp_soc))
        min_soc_end_sq   = _reach_clamp(soc_state.get("sqa",     milp_soc))
        min_soc_end_ex   = _reach_clamp(soc_state.get("exact",   milp_soc))
        min_soc_end      = min_soc_end_milp  # rebound per-solver below

        # MILP baseline
        t0 = time.time()
        try:
            soc_ref_end_milp = float(soc_ref_win[-1]) if soc_ref_win is not None else None
            milp_res = milp_schedule_window(
                wl, wsc, mg, milp_soc, price_win,
                min_soc_end=min_soc_end,
                soc_ref_end=soc_ref_end_milp,
                lam_track=MILP_LAM_TRACK,
            )
        except Exception as e:
            logger.warning(f"Hr{hour:02d} MILP failed: {e}")
            milp_res = None
        dt_milp = time.time() - t0

        if milp_res is not None and milp_res["ok"]:
            g_ml   = milp_res["g"]   / mg.P_G           # normalized [0,1]
            c_ml   = milp_res["u_c"]                     # binary {0,1}
            d_ml   = milp_res["u_d"]                     # binary {0,1}
            g_ml, c_ml, d_ml = _physics_guard(g_ml, c_ml, d_ml, milp_soc, mg, APPLY_STEPS)
            soc_arr_ml = np.clip(
                soc_trajectory(c_ml[:APPLY_STEPS], d_ml[:APPLY_STEPS], mg, milp_soc),
                mg.soc_min, mg.soc_max)
            cost_ml  = _milp_cost(milp_res, APPLY_STEPS, wl, mg, price_win)
            is_feas_ml = True
            soc_state["milp"] = float(soc_arr_ml[-1])
            rows_sched += _sched_rows(hour, APPLY_STEPS, t_start,
                                      wl, w_solar, w_wind,
                                      g_ml, c_ml, d_ml, soc_arr_ml,
                                      mg, pday.scale_mw, "milp")
        else:
            is_feas_ml = False
            cost_ml    = float("nan")
            soc_arr_ml = np.array([milp_soc] * APPLY_STEPS)
            g_ml = np.ones(T_act); c_ml = np.zeros(T_act); d_ml = np.zeros(T_act)

        _max_supply_ml = mg.P_G + mg.P_B + float(np.mean([
            sum(float(w)*(float(sv[t])+float(wv[t])) for w,sv,wv in wsc)
            for t in range(APPLY_STEPS)]))
        _cap_constrained_ml = float(np.mean(wl[:APPLY_STEPS])) > _max_supply_ml + 0.05

        rows_cmp.append(dict(
            solver="milp", hour=hour, cost=cost_ml,
            gap_abs=0.0, gap_pct=0.0,
            is_feasible=is_feas_ml, used_fallback=False,
            runtime_s=dt_milp, cap_constrained=_cap_constrained_ml,
        ))

        # Exact QUBO baseline. Enumerates all 2**n assignments, so it returns the
        # cheapest dispatch the encoding admits and needs no repair or fallback:
        # any repair count above zero here would indicate a bug, not solver error.
        # The QUBO ground state is recorded separately - it is the cost a perfect
        # energy minimiser would achieve, which is not the same as the cheapest
        # feasible dispatch when penalty terms dominate the objective.
        if enable_exact:
            ex_soc = soc_state["exact"]
            Q_ex, n_ex, off_ex = build_qubo(
                wl, wsc, mg, soc0=ex_soc, price_vec=price_win,
                soc_ref=soc_ref_win, soc_reserve_target=soc_reserve_h,
                battery_mode=batt_mode,
            )
            t0 = time.time()
            try:
                x_ex, cost_ex, ex_diag = run_exact_window(
                    Q_ex, n_ex, off_ex, T_act, wl, wsc, mg, ex_soc, price_win,
                    min_soc_end=min_soc_end_ex,
                )
            except Exception as e:
                logger.warning(f"Hr{hour:02d} exact failed: {e}")
                x_ex, cost_ex, ex_diag = None, float("nan"), {}
            dt_ex = time.time() - t0

            is_feas_ex = x_ex is not None
            if is_feas_ex:
                g_ex, c_ex, d_ex = decode(x_ex, T_act)
                soc_arr_ex = np.clip(
                    soc_trajectory(c_ex[:APPLY_STEPS], d_ex[:APPLY_STEPS], mg, ex_soc),
                    mg.soc_min, mg.soc_max)
                soc_state["exact"] = float(soc_arr_ex[-1])
                rows_sched += _sched_rows(hour, APPLY_STEPS, t_start,
                                          wl, w_solar, w_wind,
                                          g_ex, c_ex, d_ex, soc_arr_ex,
                                          mg, pday.scale_mw, "exact")
            else:
                cost_ex = float("nan")

            gap_abs_ex, gap_pct_ex = _gap(cost_ex, cost_ml)
            rows_cmp.append(dict(
                solver="exact", hour=hour, cost=cost_ex,
                gap_abs=gap_abs_ex, gap_pct=gap_pct_ex,
                is_feasible=is_feas_ex, used_fallback=False,
                runtime_s=dt_ex, cap_constrained=False,
            ))
            rows_diag.append(dict(
                solver="exact", hour=hour,
                load_mean=round(float(wl.mean()), 4),
                price_mean=round(float(price_win.mean()), 4),
                soc0=round(ex_soc, 4),
                is_feasible=is_feas_ex,
                battery_mode=batt_mode, min_soc_end=min_soc_end_ex,
                soc_target=soc_reserve_h,
                fb_source="", repair_used=False, fallback_used=False,
                solver_raw_cost=cost_ex,
                solver_raw_feasible=is_feas_ex,
                solver_raw_violations=0 if is_feas_ex else float("nan"),
                milp_cost=cost_ml,
                **{k: v for k, v in ex_diag.items()},
            ))

        # QAOA
        if enable_qaoa:
            min_soc_end = min_soc_end_qa  # per-solver reachable clamp
            for p in p_sweep:
                sname  = f"qaoa_p{p}"
                qa_soc = soc_state[sname]
                Q_qa, n_qa, off_qa = build_qubo(
                    wl, wsc, mg, soc0=qa_soc, price_vec=price_win,
                    soc_ref=soc_ref_win, soc_reserve_target=soc_reserve_h,
                    battery_mode=batt_mode,
                )
                t0 = time.time()
                qa_diag: Dict = {}
                try:
                    x_qa, cost_qa, params_qa, _, qa_diag = run_qaoa_window(
                        Q_qa, n_qa, off_qa, p, T_act, wl, wsc, mg, qa_soc,
                        price_win, warm_params=warm[p],
                        baseline_cost=cost_ml if not np.isnan(cost_ml) else None,
                        soc_ref=soc_ref_win,
                        min_soc_end=min_soc_end,
                        ibm_backend=ibm_backend,
                        ibm_shots=ibm_shots,
                    )
                    warm[p] = params_qa
                except Exception as e:
                    logger.warning(f"Hr{hour:02d} {sname} failed: {e}")
                    x_qa, cost_qa = None, float("nan")

                fallback_qa = (x_qa is None)
                g_qa = np.ones(T_act); c_qa = np.zeros(T_act); d_qa = np.zeros(T_act)
                soc_arr_qa = np.array([qa_soc] * APPLY_STEPS)

                qa_raw_bits = x_qa.copy() if x_qa is not None else None
                qa_post_repair_bits = None

                if x_qa is not None:
                    _raw_rep = check_feasibility(x_qa, T_act, wl, wsc, mg, soc0=qa_soc, eval_t=APPLY_STEPS)
                    solver_raw_cost_qa     = float(op_cost(x_qa, APPLY_STEPS, wl, mg, price_vec=price_win, scenarios=wsc))
                    solver_raw_feasible_qa = bool(_raw_rep["expected_feasible"])
                    solver_raw_viol_qa     = int(_raw_rep["total_violations"])
                else:
                    solver_raw_cost_qa = solver_raw_feasible_qa = solver_raw_viol_qa = float("nan")

                if x_qa is not None:
                    g_qa, c_qa, d_qa = decode(x_qa, T_act)
                    g_qa, c_qa, d_qa = _physics_guard(g_qa, c_qa, d_qa, qa_soc, mg, APPLY_STEPS)
                    x_pg = _encode_bits(g_qa, c_qa, d_qa, T_act)

                    _raw_feas = _post_guard_feasible(g_qa, c_qa, d_qa, T_act, wl, wsc, mg, qa_soc)
                    _raw_soc_end = float(np.clip(
                        soc_trajectory(c_qa[:APPLY_STEPS], d_qa[:APPLY_STEPS], mg, qa_soc),
                        mg.soc_min, mg.soc_max)[-1])
                    _raw_soc_ok = _raw_soc_end >= min_soc_end

                    if _raw_feas and _raw_soc_ok:
                        x_rerep = _qaoa_repair(x_pg, T_act, wl, wsc, mg, qa_soc, price_win,
                                               soc_ref=soc_ref_win, battery_mode=batt_mode,
                                               target_soc=soc_reserve_h, min_soc_end=min_soc_end)
                        g_rep, c_rep, d_rep = decode(x_rerep, T_act)
                        _rep_feas = _post_guard_feasible(g_rep, c_rep, d_rep, T_act, wl, wsc, mg, qa_soc)
                        _rep_cost = float(op_cost(x_rerep, APPLY_STEPS, wl, mg, price_vec=price_win, scenarios=wsc)) if _rep_feas else np.inf
                        _rep_soc_end = float(np.clip(
                            soc_trajectory(c_rep[:APPLY_STEPS], d_rep[:APPLY_STEPS], mg, qa_soc),
                            mg.soc_min, mg.soc_max)[-1]) if _rep_feas else -np.inf
                        if _rep_feas and _rep_cost < float(op_cost(x_pg, APPLY_STEPS, wl, mg, price_vec=price_win, scenarios=wsc)) and _rep_soc_end >= min_soc_end:
                            g_qa, c_qa, d_qa = g_rep, c_rep, d_rep
                        else:
                            g_qa, c_qa, d_qa = decode(x_pg, T_act)
                        is_feas_qa = True
                    else:
                        x_rerep = _qaoa_repair(x_pg, T_act, wl, wsc, mg, qa_soc, price_win,
                                               soc_ref=soc_ref_win, battery_mode=batt_mode,
                                               target_soc=soc_reserve_h, min_soc_end=min_soc_end)
                        g_qa, c_qa, d_qa = decode(x_rerep, T_act)
                        is_feas_qa = _post_guard_feasible(g_qa, c_qa, d_qa, T_act, wl, wsc, mg, qa_soc)
                        if not is_feas_qa:
                            x_rerep_pw = _qaoa_repair_pw(x_pg, T_act, wl, wsc, mg, qa_soc, price_win,
                                                         soc_ref=soc_ref_win, battery_mode=batt_mode,
                                                         target_soc=soc_reserve_h, min_soc_end=min_soc_end)
                            g_qa, c_qa, d_qa = decode(x_rerep_pw, T_act)
                            is_feas_qa = _post_guard_feasible(g_qa, c_qa, d_qa, T_act, wl, wsc, mg, qa_soc)

                    qa_post_repair_bits = _encode_bits(g_qa, c_qa, d_qa, T_act)
                    repair_used_qa = not np.array_equal(qa_raw_bits, qa_post_repair_bits)

                    if is_feas_qa:
                        _soc_end_qa = np.clip(
                            soc_trajectory(c_qa[:APPLY_STEPS], d_qa[:APPLY_STEPS], mg, qa_soc),
                            mg.soc_min, mg.soc_max)[-1]
                        if float(_soc_end_qa) < min_soc_end:
                            is_feas_qa = False

                    qa_fb_source = ""
                    if not is_feas_qa:
                        sf = None
                        for _qa_relax in [0.0, -0.025, -0.05, -0.075, -0.10, -0.15, -0.20, -0.30]:
                            _fb_min = min_soc_end + _qa_relax
                            sf = _safe_fallback_quantum(
                                T_act, wl, wsc, mg, qa_soc, price_win, soc_ref_win,
                                primary_seed=x_qa, min_soc_end=_fb_min,
                                battery_mode=batt_mode, target_soc=soc_reserve_h)
                            if sf is not None:
                                break
                        # Last-resort fallback: relax the floor to the physically reachable bound from soc0.
                        if sf is None:
                            _step_d = (mg.P_B / mg.SOC_CAP) * DT / mg.eta_d
                            _reach_floor = max(mg.soc_min, qa_soc - APPLY_STEPS * _step_d - 1e-6)
                            sf = _safe_fallback_quantum(
                                T_act, wl, wsc, mg, qa_soc, price_win, soc_ref_win,
                                primary_seed=x_qa, min_soc_end=_reach_floor,
                                battery_mode=batt_mode, target_soc=soc_reserve_h)
                        if sf is not None:
                            g_qa, c_qa, d_qa, fb_info = sf
                            qa_fb_source = fb_info.get("fb_source", "")
                            is_feas_qa = True; fallback_qa = True
                        else:
                            qa_fb_source = "solver_infeasible"

                else:
                    repair_used_qa = False; qa_fb_source = "solver_failed"
                    sf = None
                    for _qa_relax in [0.0, -0.025, -0.05, -0.075, -0.10, -0.15, -0.20, -0.30]:
                        _fb_min = min_soc_end + _qa_relax
                        sf = _safe_fallback_quantum(
                            T_act, wl, wsc, mg, qa_soc, price_win, soc_ref_win,
                            primary_seed=None, min_soc_end=_fb_min,
                            battery_mode=batt_mode, target_soc=soc_reserve_h)
                        if sf is not None:
                            break
                    # Last-resort fallback: relax floor to the physically reachable bound from soc0.
                    if sf is None:
                        _step_d = (mg.P_B / mg.SOC_CAP) * DT / mg.eta_d
                        _reach_floor = max(mg.soc_min, qa_soc - APPLY_STEPS * _step_d - 1e-6)
                        sf = _safe_fallback_quantum(
                            T_act, wl, wsc, mg, qa_soc, price_win, soc_ref_win,
                            primary_seed=None, min_soc_end=_reach_floor,
                            battery_mode=batt_mode, target_soc=soc_reserve_h)
                    if sf is not None:
                        g_qa, c_qa, d_qa, fb_info = sf
                        qa_fb_source = "solver_failed_" + fb_info.get("fb_source", "")
                        is_feas_qa = True; fallback_qa = True
                    else:
                        is_feas_qa = False

                if is_feas_qa:
                    cost_qa = float(op_cost(
                        _encode_bits(g_qa, c_qa, d_qa, T_act),
                        APPLY_STEPS, wl, mg, price_vec=price_win, scenarios=wsc))
                    soc_arr_qa = np.clip(
                        soc_trajectory(c_qa[:APPLY_STEPS], d_qa[:APPLY_STEPS], mg, qa_soc),
                        mg.soc_min, mg.soc_max)
                    soc_state[sname] = float(soc_arr_qa[-1])
                    rows_sched += _sched_rows(hour, APPLY_STEPS, t_start,
                                              wl, w_solar, w_wind,
                                              g_qa, c_qa, d_qa, soc_arr_qa,
                                              mg, pday.scale_mw, sname)
                else:
                    cost_qa = float("nan")

                gap_abs_qa, gap_pct_qa = _gap(cost_qa, cost_ml)
                rows_cmp.append(dict(
                    solver=sname, hour=hour, cost=cost_qa,
                    gap_abs=gap_abs_qa, gap_pct=gap_pct_qa,
                    is_feasible=is_feas_qa, used_fallback=fallback_qa,
                    runtime_s=time.time() - t0,
                    cap_constrained=False,
                ))
                rows_diag.append(dict(
                    solver=sname, hour=hour,
                    load_mean=round(float(wl.mean()), 4),
                    price_mean=round(float(price_win.mean()), 4),
                    soc0=round(qa_soc, 4),
                    is_feasible=is_feas_qa,
                    battery_mode=batt_mode, min_soc_end=min_soc_end, soc_target=soc_reserve_h,
                    fb_source=qa_fb_source, repair_used=repair_used_qa, fallback_used=fallback_qa,
                    solver_raw_cost=solver_raw_cost_qa,
                    solver_raw_feasible=solver_raw_feasible_qa,
                    solver_raw_violations=solver_raw_viol_qa,
                    ibm_backend=ibm_backend or "simulator",
                    **{k: v for k, v in qa_diag.items()},
                ))

        # SQA
        # Sampler-based solvers. SQA and the classical baselines traverse one shared
        # path: same QUBO, same physics guard, same staged repair, same fallback ladder,
        # same diagnostics. Only the sampler differs, so any cost difference between
        # them is attributable to the sampler and not to the surrounding pipeline -
        # which is the premise the classical-baseline comparison rests on.
        def _run_sampler_solver(name: str, solve, soc_floor: float) -> None:
            soc0_s = soc_state[name]
            Q_s, n_s, off_s = build_qubo(
                wl, wsc, mg, soc0=soc0_s, price_vec=price_win,
                soc_ref=soc_ref_win, soc_reserve_target=soc_reserve_h,
                battery_mode=batt_mode)

            t0 = time.time()
            diag_s: Dict = {}
            try:
                x_s, cost_s, diag_s = solve(Q_s, n_s, off_s, soc0_s, soc_floor)
            except Exception as e:
                logger.warning(f"Hr{hour:02d} {name} failed: {e}")
                x_s, cost_s = None, float("nan")

            fallback_s = (x_s is None)
            g_s = np.ones(T_act); c_s = np.zeros(T_act); d_s = np.zeros(T_act)
            soc_arr_s = np.array([soc0_s] * APPLY_STEPS)
            raw_bits = x_s.copy() if x_s is not None else None

            if x_s is not None:
                _raw = check_feasibility(x_s, T_act, wl, wsc, mg,
                                         soc0=soc0_s, eval_t=APPLY_STEPS)
                raw_cost = float(op_cost(x_s, APPLY_STEPS, wl, mg,
                                         price_vec=price_win, scenarios=wsc))
                raw_feas = bool(_raw["expected_feasible"])
                raw_viol = int(_raw["total_violations"])
            else:
                raw_cost = raw_feas = raw_viol = float("nan")

            if x_s is not None:
                g_s, c_s, d_s = decode(x_s, T_act)
                g_s, c_s, d_s = _physics_guard(g_s, c_s, d_s, soc0_s, mg, APPLY_STEPS)
                x_pg = _encode_bits(g_s, c_s, d_s, T_act)

                # Feasibility here means the physical constraints only: power balance
                # within tolerance, SOC inside [soc_min, soc_max], and no simultaneous
                # charge/discharge. The terminal floor `soc_floor` is derived from the
                # daily reference trajectory and is a *tracking preference*, not a
                # physical limit - a schedule that balances power and respects the SOC
                # bounds is operationally valid even if it ends below the reference.
                # Treating a reference miss as infeasibility previously forced the
                # staged repair to run on schedules that were already valid, and that
                # repair optimises toward the reference at the expense of balance: it
                # rewrote 16 of 20 bits and turned zero violations into sixteen. The
                # resulting failure also froze the solver's SOC state, so the next
                # window inherited an unreachable target and the whole day cascaded.
                # Reference shortfall is now recorded as a diagnostic instead.
                def _acceptable(bits) -> bool:
                    gg, cc, ddx = decode(bits, T_act)
                    return _post_guard_feasible(gg, cc, ddx, T_act, wl, wsc, mg, soc0_s)

                if _acceptable(x_pg):
                    g_s, c_s, d_s = decode(x_pg, T_act)
                    is_feas_s = True
                    repair_stage = ""
                else:
                    x_rerep, repair_stage = _sqa_repair_staged(
                        x_pg, T_act, wl, wsc, mg, soc0_s, price_win,
                        soc_ref=soc_ref_win, battery_mode=batt_mode,
                        target_soc=soc_reserve_h, min_soc_end=soc_floor)
                    # Keep the repaired schedule only if it is an improvement; the
                    # staged repair is not guaranteed to be monotone.
                    if _acceptable(x_rerep):
                        g_s, c_s, d_s = decode(x_rerep, T_act)
                        is_feas_s = True
                    else:
                        g_s, c_s, d_s = decode(x_rerep, T_act)
                        is_feas_s = False

                post_bits = _encode_bits(g_s, c_s, d_s, T_act)
                repair_used_s = not np.array_equal(raw_bits, post_bits)
                hamming_s = int(np.sum(raw_bits != post_bits))
                fb_source_s = repair_stage if repair_used_s else ""
                if not is_feas_s:
                    sf = None
                    for relax in [0.0, -0.025, -0.05, -0.075, -0.10, -0.15, -0.20, -0.30]:
                        sf = _safe_fallback_quantum(
                            T_act, wl, wsc, mg, soc0_s, price_win, soc_ref_win,
                            primary_seed=x_s, min_soc_end=soc_floor + relax,
                            battery_mode=batt_mode, target_soc=soc_reserve_h)
                        if sf is not None:
                            break
                    # Last resort: relax the floor to the bound reachable from soc0.
                    if sf is None:
                        _step_d = (mg.P_B / mg.SOC_CAP) * DT / mg.eta_d
                        _reach_floor = max(mg.soc_min,
                                           soc0_s - APPLY_STEPS * _step_d - 1e-6)
                        sf = _safe_fallback_quantum(
                            T_act, wl, wsc, mg, soc0_s, price_win, soc_ref_win,
                            primary_seed=x_s, min_soc_end=_reach_floor,
                            battery_mode=batt_mode, target_soc=soc_reserve_h)
                    if sf is not None:
                        g_s, c_s, d_s, fb_info = sf
                        fb_source_s = fb_info.get("fb_source", "")
                        is_feas_s = True; fallback_s = True
                    else:
                        fb_source_s = "solver_infeasible"

            else:
                repair_used_s = False; hamming_s = -1
                fb_source_s = "solver_failed"
                sf = _safe_fallback_quantum(
                    T_act, wl, wsc, mg, soc0_s, price_win, soc_ref_win,
                    primary_seed=None, min_soc_end=soc_floor,
                    battery_mode=batt_mode, target_soc=soc_reserve_h)
                if sf is None:
                    _step_d = (mg.P_B / mg.SOC_CAP) * DT / mg.eta_d
                    _reach_floor = max(mg.soc_min, soc0_s - APPLY_STEPS * _step_d - 1e-6)
                    sf = _safe_fallback_quantum(
                        T_act, wl, wsc, mg, soc0_s, price_win, soc_ref_win,
                        primary_seed=None, min_soc_end=_reach_floor,
                        battery_mode=batt_mode, target_soc=soc_reserve_h)
                if sf is not None:
                    g_s, c_s, d_s, fb_info = sf
                    fb_source_s = "solver_failed_" + fb_info.get("fb_source", "")
                    is_feas_s = True; fallback_s = True
                else:
                    is_feas_s = False

            if is_feas_s:
                # Close any residual deficit against the *forecast* renewables, the
                # same information the QUBO and the MILP baseline were given. Using
                # realized generation here would hand this solver a hindsight
                # correction that neither MILP nor the exact baseline receives, and
                # would make the cost comparison unattributable.
                _renew_fc = np.zeros(T_act, dtype=float)
                for _w, _sv, _wv in wsc:
                    _renew_fc += float(_w) * (np.asarray(_sv[:T_act], float)
                                              + np.asarray(_wv[:T_act], float))
                g_s, c_s, d_s = _snap_grid_up(g_s, c_s, d_s, wl[:T_act],
                                              _renew_fc, mg, soc0_s)
                cost_s = float(op_cost(
                    _encode_bits(g_s, c_s, d_s, T_act),
                    APPLY_STEPS, wl, mg, price_vec=price_win, scenarios=wsc))
                soc_arr_s = np.clip(
                    soc_trajectory(c_s[:APPLY_STEPS], d_s[:APPLY_STEPS], mg, soc0_s),
                    mg.soc_min, mg.soc_max)
                soc_state[name] = float(soc_arr_s[-1])
                soc_shortfall_s = max(0.0, soc_floor - float(soc_arr_s[-1]))
                rows_sched.extend(_sched_rows(hour, APPLY_STEPS, t_start,
                                             wl, w_solar, w_wind,
                                             g_s, c_s, d_s, soc_arr_s,
                                             mg, pday.scale_mw, name))
            else:
                cost_s = float("nan")
                soc_shortfall_s = float("nan")

            gap_abs_s, gap_pct_s = _gap(cost_s, cost_ml)
            rows_cmp.append(dict(
                solver=name, hour=hour, cost=cost_s,
                gap_abs=gap_abs_s, gap_pct=gap_pct_s,
                is_feasible=is_feas_s, used_fallback=fallback_s,
                runtime_s=time.time() - t0,
                cap_constrained=False,
            ))
            rows_diag.append(dict(
                solver=name, hour=hour,
                load_mean=round(float(wl.mean()), 4),
                price_mean=round(float(price_win.mean()), 4),
                soc0=round(soc0_s, 4),
                is_feasible=is_feas_s,
                battery_mode=batt_mode, min_soc_end=soc_floor,
                soc_target=soc_reserve_h,
                fb_source=fb_source_s, repair_used=repair_used_s,
                hamming_raw_to_repaired=hamming_s,
                soc_ref_shortfall=soc_shortfall_s,
                fallback_used=fallback_s,
                solver_raw_cost=raw_cost,
                solver_raw_feasible=raw_feas,
                solver_raw_violations=raw_viol,
                milp_cost=cost_ml,
                **{k: v for k, v in diag_s.items()},
            ))

        if enable_sqa:
            _run_sampler_solver(
                "sqa",
                lambda Q, n_, off, s0, fl: run_sqa_window(
                    Q, n_, off, T_act, wl, wsc, mg, s0, price_win, min_soc_end=fl),
                min_soc_end_sq,
            )

        for _bl in bqm_baselines:
            _run_sampler_solver(
                _bl,
                lambda Q, n_, off, s0, fl, _nm=_bl: run_bqm_window(
                    _nm, Q, n_, off, T_act, wl, wsc, mg, s0, price_win,
                    num_reads=BQM_BASELINE_READS, min_soc_end=fl),
                _reach_clamp(soc_state[_bl]),
            )

        # Per-hour log
        ml_row = next((r for r in rows_cmp if r["solver"] == "milp" and r["hour"] == hour), {})
        parts  = [f"Hr{hour:02d}", f"MILP={ml_row.get('cost', float('nan')):.4f}(feas={ml_row.get('is_feasible')})"]
        if enable_qaoa:
            qa_row = next((r for r in reversed(rows_cmp) if r["solver"].startswith("qaoa") and r["hour"] == hour), {})
            parts.append(f"QAOA={qa_row.get('cost', float('nan')):.4f}(Δ={qa_row.get('gap_pct', float('nan')):.1f}%)")
        if enable_sqa:
            sq_row = next((r for r in reversed(rows_cmp) if r["solver"] == "sqa" and r["hour"] == hour), {})
            parts.append(f"SQA={sq_row.get('cost', float('nan')):.4f}(Δ={sq_row.get('gap_pct', float('nan')):.1f}%)")
        parts.append(f"t={time.time()-t_wall:.1f}s")
        logger.info(" | ".join(parts))

    return ScheduleResult(
        date=pday.date,
        df_compare=pd.DataFrame(rows_cmp),
        df_schedule=pd.DataFrame(rows_sched),
        df_diagnostics=pd.DataFrame(rows_diag),
        metadata={
            "date":           pday.date,
            "scale_mw":       pday.scale_mw,
            "total_time_s":   time.time() - t_wall,
            "solvers":        SOLVERS,
            "baseline":       "milp",
            "benchmark_mode": "independent_soc",
            "ibm_backend":    ibm_backend or "simulator",
            "bits_per_step":  BITS_PER_STEP,
        },
    )
