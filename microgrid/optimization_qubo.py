from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Optional
from microgrid.config import (
    GRID_COEFFS, DT,
    LAM_BALANCE, LAM_MUTEX, LAM_SOC, LAM_RESERVE, LAM_TRACK, SOC_RESERVE,
    LAM_CHARGE_RECOVERY,
    MicrogridParams,
)
from microgrid.utils import idx_g0, idx_g1, idx_g2, idx_c, idx_d, n_qubits


def _add(Q: Dict, i: int, j: int, v: float) -> None:
    if abs(v) < 1e-14:
        return
    if i > j:
        i, j = j, i
    Q[(i, j)] = Q.get((i, j), 0.0) + v


def build_qubo(
    load_vec: np.ndarray,
    scenarios: List[Tuple[float, np.ndarray, np.ndarray]],
    mg: MicrogridParams,
    soc0: float,
    price_vec: np.ndarray,
    lam_balance: float = None,
    lam_soc:     float = None,
    lam_reserve: float = None,
    lam_mutex:   float = None,
    soc_ref:     Optional[np.ndarray] = None,
    lam_track:   float = None,
    soc_reserve_target: float = None,
    battery_mode: Optional[str] = None,
) -> Tuple[Dict, int, float]:
    """Build QUBO. lam_* override config defaults (used by QAOA adaptive penalties).
    soc_ref: per-step reference SOC target (length T). lam_track: weight for tracking penalty.
    """
    T = len(load_vec)
    n = n_qubits(T)
    Q: Dict[Tuple[int, int], float] = {}
    offset = 0.0

    _lam_balance = lam_balance if lam_balance is not None else LAM_BALANCE
    _lam_soc     = lam_soc     if lam_soc     is not None else LAM_SOC
    _lam_reserve = lam_reserve if lam_reserve is not None else LAM_RESERVE
    _lam_mutex   = lam_mutex   if lam_mutex   is not None else LAM_MUTEX
    _lam_track   = lam_track   if lam_track   is not None else LAM_TRACK

    # 1. Objective: operating cost (load shedding removed)
    for t in range(T):
        pr = float(price_vec[t])
        _add(Q, idx_g0(t), idx_g0(t), DT * pr * mg.C_grid  * mg.P_G * GRID_COEFFS[0])
        _add(Q, idx_g1(t), idx_g1(t), DT * pr * mg.C_grid  * mg.P_G * GRID_COEFFS[1])
        _add(Q, idx_g2(t), idx_g2(t), DT * pr * mg.C_grid  * mg.P_G * GRID_COEFFS[2])
        _add(Q, idx_c(t),  idx_c(t),  DT * pr * mg.C_cycle * mg.P_B)
        _add(Q, idx_d(t),  idx_d(t),  DT * pr * mg.C_cycle * mg.P_B)

        # Charge recovery reward: applied only during forecast low-net-load "charge" mode
        # AND only when starting SOC is below target. Linear bonus on c[t] (negative diag).
        if (battery_mode == "charge"
                and soc_reserve_target is not None
                and soc0 < soc_reserve_target):
            charge_reward = LAM_CHARGE_RECOVERY * (float(soc_reserve_target) - float(soc0))
            _add(Q, idx_c(t), idx_c(t), -charge_reward)

    # 2. Power-balance penalty (scenario-weighted)
    lam_b = _lam_balance * mg.C_grid * mg.P_G
    for w, sv, wv in scenarios:
        for t in range(T):
            b = float(sv[t]) + float(wv[t]) - float(load_vec[t])
            terms = [
                (idx_g0(t), mg.P_G * GRID_COEFFS[0]),
                (idx_g1(t), mg.P_G * GRID_COEFFS[1]),
                (idx_g2(t), mg.P_G * GRID_COEFFS[2]),
                (idx_c(t),  -mg.P_B),
                (idx_d(t),   mg.P_B),
            ]
            pen = float(w) * lam_b
            for k, (ii, ai) in enumerate(terms):
                for jj, aj in terms[k + 1:]:
                    _add(Q, ii, jj, pen * 2.0 * ai * aj)
                _add(Q, ii, ii, pen * (ai ** 2 + 2.0 * ai * b))
            offset += pen * b ** 2

    # 3. Charge/discharge mutual exclusion
    lam_m = _lam_mutex * mg.C_grid * mg.P_G
    for t in range(T):
        _add(Q, idx_c(t), idx_d(t), lam_m)

    # 4. Cumulative SOC reserve penalty - quadratic on full path SOC(t) toward target.
    # Target defaults to SOC_RESERVE (config) but can be overridden per-hour via soc_reserve_target.
    # Penalizes deviation from reserve at each step over cumulative SOC trajectory.
    # SOC(t) = soc0 + sum_{k<=t}(chg_f*c[k] - dis_f*d[k])
    # (SOC(t) - target)² expands to standard QUBO quadratic form.
    _soc_reserve = soc_reserve_target if soc_reserve_target is not None else SOC_RESERVE
    dis_f = (mg.P_B / mg.SOC_CAP) * DT / mg.eta_d
    chg_f = (mg.P_B / mg.SOC_CAP) * DT * mg.eta_c
    lam_s = _lam_soc * mg.C_grid * mg.P_G
    if lam_s > 0.0:
        for t in range(T):
            delta = soc0 - _soc_reserve
            terms = []
            for k in range(t + 1):
                terms.append((idx_c(k),  chg_f))
                terms.append((idx_d(k), -dis_f))
            for i, (ii, ai) in enumerate(terms):
                for jj, aj in terms[i + 1:]:
                    _add(Q, ii, jj, lam_s * 2.0 * ai * aj)
                _add(Q, ii, ii, lam_s * (ai ** 2 + 2.0 * ai * delta))
            offset += lam_s * delta ** 2

    # 5. Terminal SOC value function - linear opportunity cost of depleted battery
    lam_v = _lam_reserve * mg.C_grid * mg.P_G
    for t in range(T):
        _add(Q, idx_d(t), idx_d(t),  lam_v * dis_f)
        _add(Q, idx_c(t), idx_c(t), -lam_v * chg_f)

    # 6. SOC-reference tracking (daily-LP plan)
    # Penalty: sum_t lam_track * (soc(t) - soc_ref(t))²
    # soc(t) = soc0 + sum_{k<=t}(chg_f*c[k] - dis_f*d[k])
    # Let delta_t = soc0 - soc_ref[t]; sum_t = sum_{k<=t}(chg_f*c[k] - dis_f*d[k])
    # (delta_t + sum_t)² = delta_t² + 2*delta_t*sum_t + sum_t²
    if soc_ref is not None and _lam_track > 0.0:
        lam_k = _lam_track * mg.C_grid * mg.P_G
        for t in range(T):
            target = float(soc_ref[t])
            delta  = soc0 - target
            terms = []
            for k in range(t + 1):
                terms.append((idx_c(k),  chg_f))
                terms.append((idx_d(k), -dis_f))
            for i, (ii, ai) in enumerate(terms):
                for jj, aj in terms[i + 1:]:
                    _add(Q, ii, jj, lam_k * 2.0 * ai * aj)
                _add(Q, ii, ii, lam_k * (ai ** 2 + 2.0 * ai * delta))
            offset += lam_k * delta ** 2

    return Q, n, offset

