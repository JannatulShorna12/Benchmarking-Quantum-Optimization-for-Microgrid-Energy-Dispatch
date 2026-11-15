from __future__ import annotations
import numpy as np
from scipy.optimize import linprog
from microgrid.config import (
    DT, PRICE_VEC, SOC_INIT, MicrogridParams,
)


class _SOCRef(np.ndarray):
    """np.ndarray subclass that allows extra attribute assignment for LP diagnostics."""
    def __new__(cls, arr):
        return np.asarray(arr).view(cls)

    def __array_finalize__(self, obj):
        pass


def solve_daily_lp(
    load_vec: np.ndarray,
    solar_vec: np.ndarray,
    wind_vec: np.ndarray,
    mg: MicrogridParams,
    price_vec: np.ndarray = PRICE_VEC,
    soc0: float = SOC_INIT,
) -> np.ndarray:
    """Solve 24h relaxed LP with forecast. Returns reference SOC trajectory[T].

    Decision vars (continuous, all in [0,1]):
      g[t], c[t], d[t], s[t]  for t in 0..T-1
    Constraints:
      Power balance (equality per step)
      SOC bounds (inequality, cumulative)
    Objective:
      Sum(DT * price[t] * (C_grid*P_G*g + C_cycle*P_B*(c+d) + C_shed*shed_frac*load*s))

    Mutex (c+d≤1) relaxed - LP finds continuous SOC target, scheduling respects mutex.

    Note: LP includes load shedding (s[t]) for balance feasibility. QUBO has no shedding.
    When LP relies on shedding (lp_shed_used=True in diagnostics), the SOC reference
    may not be achievable by QUBO solvers - treat soc_ref as less reliable for those hours.
    """
    T     = len(load_vec)
    renew = np.asarray(solar_vec[:T], float) + np.asarray(wind_vec[:T], float)
    load  = np.asarray(load_vec[:T], float)
    price = np.asarray(price_vec[:T], float)

    n_v = 4 * T
    i_g, i_c, i_d, i_s = 0, T, 2*T, 3*T

    # Objective coefficients
    c_obj = np.zeros(n_v)
    for t in range(T):
        c_obj[i_g + t] = DT * price[t] * mg.C_grid  * mg.P_G
        c_obj[i_c + t] = DT * price[t] * mg.C_cycle * mg.P_B
        c_obj[i_d + t] = DT * price[t] * mg.C_cycle * mg.P_B
        c_obj[i_s + t] = DT * price[t] * mg.C_shed  * mg.shed_frac * load[t]

    # Power balance equality: P_G*g + P_B*d - P_B*c + shed_frac*load*s + renew = load
    A_eq = np.zeros((T, n_v))
    b_eq = np.zeros(T)
    for t in range(T):
        A_eq[t, i_g + t] =  mg.P_G
        A_eq[t, i_d + t] =  mg.P_B
        A_eq[t, i_c + t] = -mg.P_B
        A_eq[t, i_s + t] =  mg.shed_frac * load[t]
        b_eq[t]          =  load[t] - renew[t]

    # SOC bounds: soc[t] = soc0 + sum_{k<=t}(chg_f*c[k] - dis_f*d[k])
    chg_f = (mg.P_B / mg.SOC_CAP) * DT * mg.eta_c
    dis_f = (mg.P_B / mg.SOC_CAP) * DT / mg.eta_d
    A_ub = np.zeros((2*T, n_v))
    b_ub = np.zeros(2*T)
    for t in range(T):
        # soc[t] <= soc_max  ->  sum(chg_f*c - dis_f*d) <= soc_max - soc0
        for k in range(t + 1):
            A_ub[t, i_c + k] =  chg_f
            A_ub[t, i_d + k] = -dis_f
        b_ub[t] = mg.soc_max - soc0
        # soc[t] <= -soc_min  ->  sum(-chg_f*c + dis_f*d) <= soc0 - soc_min
        for k in range(t + 1):
            A_ub[T + t, i_c + k] = -chg_f
            A_ub[T + t, i_d + k] =  dis_f
        b_ub[T + t] = soc0 - mg.soc_min

    bounds = [(0.0, 1.0)] * n_v

    res = linprog(
        c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
        bounds=bounds, method="highs",
    )
    if not res.success:
        raise RuntimeError(f"Daily LP failed: {res.message}")

    x = res.x
    soc = np.empty(T, dtype=float)
    cur = soc0
    for t in range(T):
        cur = cur + chg_f * x[i_c + t] - dis_f * x[i_d + t]
        soc[t] = cur
    soc_ref = _SOCRef(np.clip(soc, mg.soc_min, mg.soc_max))

    # Shed diagnostics: how much LP relies on load shedding (QUBO has no shed)
    shed_amounts = x[i_s: i_s + T] * mg.shed_frac * load  # shape (T,)
    lp_shed_total = float(np.sum(shed_amounts))
    soc_ref.lp_shed_used   = lp_shed_total > 1e-6
    soc_ref.lp_shed_total  = lp_shed_total
    soc_ref.lp_shed_per_t  = shed_amounts

    return soc_ref
