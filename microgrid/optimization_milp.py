"""
MILP baseline for one microgrid scheduling window.

Variables per step t (continuous unless noted):
  g[t]  ∈ [0, P_G]          grid import power (pu)
  bc[t] ∈ {0, P_B}          battery charge power  - BINARY × P_B
  bd[t] ∈ {0, P_B}          battery discharge power - BINARY × P_B
  u_c[t] ∈ {0,1}            charge binary
  u_d[t] ∈ {0,1}            discharge binary
  s[t]  ∈ [0, shed_frac*load]  load shed (relaxed for feasibility)

Constraints:
  Power balance (equality, per scenario weighted mean)
  SOC bounds (hard, cumulative)
  Mutex: u_c + u_d ≤ 1
  bc = u_c * P_B,  bd = u_d * P_B

MILP gives globally optimal integer solution -> absolute cheapest feasible schedule.
QAOA, SQA and D-Wave results are compared against this baseline.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Optional
from microgrid.config import DT, MicrogridParams


def milp_schedule_window(
    load_vec: np.ndarray,
    scenarios: List[Tuple[float, np.ndarray, np.ndarray]],
    mg: MicrogridParams,
    soc0: float,
    price_vec: np.ndarray,
    min_soc_end: Optional[float] = None,
    soc_ref_end: Optional[float] = None,
    lam_track:   float = 0.5,
    soc_ref:     Optional[np.ndarray] = None,
) -> Optional[Dict]:
    """Solve one scheduling window with MILP (binary charge/discharge, continuous grid).

    Returns dict with keys: g, bc, bd, u_c, u_d, s, soc, soc_end,
                            deploy_cost, window_cost, ok
    Returns None only if MILP is infeasible (should never happen with load shed slack).

    SOC-reference tracking. ``soc_ref`` applies the same per-step tracking the QUBO uses
    (``H_track`` in optimization_qubo), so both formulations are anchored to the
    day-ahead trajectory over the same horizon and with the same weight. ``soc_ref_end``
    is the earlier terminal-only variant, retained for backward compatibility; passing
    ``soc_ref`` supersedes it.

    The QUBO penalises the deviation quadratically. A MILP cannot express a quadratic
    objective, so the linear (L1) counterpart is used here. Scope and weight therefore
    match; the functional form necessarily does not, and that residual difference is the
    one asymmetry that cannot be removed without turning the baseline into a MIQP.
    """
    import pulp

    T = len(load_vec)

    # Scenario-weighted mean renewable
    mean_re = np.zeros(T)
    for w, sv, wv in scenarios:
        mean_re += float(w) * (np.asarray(sv[:T], float) + np.asarray(wv[:T], float))

    net_load = np.asarray(load_vec, float) - mean_re  # positive = deficit

    prob = pulp.LpProblem("milp_schedule", pulp.LpMinimize)

    # Decision variables
    g   = [pulp.LpVariable(f"g_{t}",   0.0, mg.P_G)          for t in range(T)]
    u_c = [pulp.LpVariable(f"uc_{t}",  cat="Binary")          for t in range(T)]
    u_d = [pulp.LpVariable(f"ud_{t}",  cat="Binary")          for t in range(T)]
    s   = [pulp.LpVariable(f"s_{t}",   0.0,
                           mg.shed_frac * float(load_vec[t])) for t in range(T)]

    # SOC auxiliary variable
    soc_var = [pulp.LpVariable(f"soc_{t}", mg.soc_min, mg.soc_max) for t in range(T)]

    chg_f = (mg.P_B / mg.SOC_CAP) * DT * mg.eta_c
    dis_f = (mg.P_B / mg.SOC_CAP) * DT / mg.eta_d

    # Objective: minimize total operating cost over window
    cost_terms = []
    for t in range(T):
        pr = float(price_vec[t])
        cost_terms.append(DT * pr * mg.C_grid  * g[t])
        cost_terms.append(DT * pr * mg.C_cycle * mg.P_B * u_c[t])
        cost_terms.append(DT * pr * mg.C_cycle * mg.P_B * u_d[t])
        cost_terms.append(DT * pr * mg.C_shed  * s[t])
    prob += pulp.lpSum(cost_terms)

    for t in range(T):
        pr = float(price_vec[t])
        net_t = float(net_load[t])

        # Power balance: g + P_B*u_d - P_B*u_c + s = net_load
        prob += (g[t] + mg.P_B * u_d[t] - mg.P_B * u_c[t] + s[t] == net_t,
                 f"balance_{t}")

        # Mutex: charge and discharge cannot both be active
        prob += (u_c[t] + u_d[t] <= 1, f"mutex_{t}")

        # SOC dynamics
        if t == 0:
            prob += (soc_var[t] == soc0 + chg_f * u_c[t] - dis_f * u_d[t],
                     f"soc_dyn_{t}")
        else:
            prob += (soc_var[t] == soc_var[t - 1] + chg_f * u_c[t] - dis_f * u_d[t],
                     f"soc_dyn_{t}")

    # Optional terminal SOC floor: ensures the window ends above a target reserve.
    if min_soc_end is not None:
        eff_floor = max(mg.soc_min, float(min_soc_end))
        eff_floor = min(eff_floor, mg.soc_max)
        prob += (soc_var[T - 1] >= eff_floor, "soc_end_floor")

    # Soft SOC tracking to the daily reference trajectory.
    if soc_ref is not None:
        # Per-step tracking, matching the QUBO's H_track scope and weight.
        ref_arr = np.clip(np.asarray(soc_ref[:T], float), mg.soc_min, mg.soc_max)
        for t in range(T):
            dpos = pulp.LpVariable(f"soc_dev_pos_{t}", 0.0, None)
            dneg = pulp.LpVariable(f"soc_dev_neg_{t}", 0.0, None)
            prob += (soc_var[t] - float(ref_arr[t]) == dpos - dneg, f"soc_track_eq_{t}")
            cost_terms.append(lam_track * (dpos + dneg))
        prob.objective = pulp.lpSum(cost_terms)
    elif soc_ref_end is not None:
        # Terminal-only tracking (legacy behaviour).
        ref = float(np.clip(soc_ref_end, mg.soc_min, mg.soc_max))
        dpos = pulp.LpVariable("soc_dev_pos", 0.0, None)
        dneg = pulp.LpVariable("soc_dev_neg", 0.0, None)
        prob += (soc_var[T - 1] - ref == dpos - dneg, "soc_track_eq")
        cost_terms.append(lam_track * (dpos + dneg))
        prob.objective = pulp.lpSum(cost_terms)

    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=30, gapRel=1e-6))

    status = pulp.LpStatus[prob.status]
    if status not in ("Optimal", "Feasible"):
        return None

    g_sol   = np.array([pulp.value(g[t])   for t in range(T)], dtype=float)
    u_c_sol = np.array([pulp.value(u_c[t]) for t in range(T)], dtype=float)
    u_d_sol = np.array([pulp.value(u_d[t]) for t in range(T)], dtype=float)
    s_sol   = np.array([pulp.value(s[t])   for t in range(T)], dtype=float)
    soc_sol = np.array([pulp.value(soc_var[t]) for t in range(T)], dtype=float)

    # Round binaries (CBC can return 0.9999...)
    u_c_sol = np.round(u_c_sol).clip(0, 1)
    u_d_sol = np.round(u_d_sol).clip(0, 1)
    g_sol   = np.clip(g_sol,   0.0, mg.P_G)
    s_sol   = np.clip(s_sol,   0.0, None)
    soc_sol = np.clip(soc_sol, mg.soc_min, mg.soc_max)

    bc_sol = u_c_sol * mg.P_B
    bd_sol = u_d_sol * mg.P_B

    pr0         = float(price_vec[0])
    deploy_cost = DT * pr0 * (
        mg.C_grid  * g_sol[0]
        + mg.C_cycle * mg.P_B * (u_c_sol[0] + u_d_sol[0])
        + mg.C_shed  * s_sol[0]
    )

    return {
        "g":   g_sol,
        "bc":  bc_sol,
        "bd":  bd_sol,
        "u_c": u_c_sol,
        "u_d": u_d_sol,
        "s":   s_sol,
        "soc": soc_sol,
        "soc_end":    float(soc_sol[-1]),
        "deploy_cost": float(deploy_cost),
        "window_cost": float(pulp.value(prob.objective)),
        "ok": True,
        "status": status,
    }


def milp_daily(
    load_vec: np.ndarray,
    solar_vec: np.ndarray,
    wind_vec: np.ndarray,
    mg: MicrogridParams,
    price_vec: np.ndarray,
    soc0: float,
) -> np.ndarray:
    """Solve full 24h MILP (binary c/d) for SOC reference trajectory.

    Returns SOC array of shape (T,). Used instead of LP reference when MILP baseline active.
    Falls back to LP if MILP fails.
    """
    import pulp

    T = len(load_vec)
    renew = np.asarray(solar_vec[:T], float) + np.asarray(wind_vec[:T], float)
    load  = np.asarray(load_vec[:T], float)
    price = np.asarray(price_vec[:T], float)
    net_load = load - renew

    chg_f = (mg.P_B / mg.SOC_CAP) * DT * mg.eta_c
    dis_f = (mg.P_B / mg.SOC_CAP) * DT / mg.eta_d

    prob = pulp.LpProblem("milp_daily", pulp.LpMinimize)

    g   = [pulp.LpVariable(f"g_{t}",  0.0, mg.P_G)                      for t in range(T)]
    u_c = [pulp.LpVariable(f"uc_{t}", cat="Binary")                      for t in range(T)]
    u_d = [pulp.LpVariable(f"ud_{t}", cat="Binary")                      for t in range(T)]
    s   = [pulp.LpVariable(f"s_{t}",  0.0, mg.shed_frac * float(load[t])) for t in range(T)]
    soc_v = [pulp.LpVariable(f"soc_{t}", mg.soc_min, mg.soc_max)         for t in range(T)]

    obj = []
    for t in range(T):
        pr = float(price[t])
        obj.append(DT * pr * mg.C_grid  * g[t])
        obj.append(DT * pr * mg.C_cycle * mg.P_B * u_c[t])
        obj.append(DT * pr * mg.C_cycle * mg.P_B * u_d[t])
        obj.append(DT * pr * mg.C_shed  * s[t])
    prob += pulp.lpSum(obj)

    for t in range(T):
        prob += (g[t] + mg.P_B * u_d[t] - mg.P_B * u_c[t] + s[t] == float(net_load[t]),
                 f"bal_{t}")
        prob += (u_c[t] + u_d[t] <= 1, f"mutex_{t}")
        if t == 0:
            prob += (soc_v[t] == soc0 + chg_f * u_c[t] - dis_f * u_d[t], f"soc_{t}")
        else:
            prob += (soc_v[t] == soc_v[t-1] + chg_f * u_c[t] - dis_f * u_d[t], f"soc_{t}")

    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=60, gapRel=1e-4))

    status = pulp.LpStatus[prob.status]
    if status not in ("Optimal", "Feasible"):
        # Fall back to LP SOC reference
        from microgrid.daily_plan import solve_daily_lp
        return solve_daily_lp(load_vec, solar_vec, wind_vec, mg,
                              price_vec=price_vec, soc0=soc0)

    soc_arr = np.array([float(pulp.value(soc_v[t])) for t in range(T)])
    return np.clip(soc_arr, mg.soc_min, mg.soc_max)
