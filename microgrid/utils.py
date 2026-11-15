from __future__ import annotations
import numpy as np
from typing import Dict, Tuple
from microgrid.config import BITS_PER_STEP, GRID_COEFFS, DT, MicrogridParams


def idx_g0(t: int) -> int: return BITS_PER_STEP * t
def idx_g1(t: int) -> int: return BITS_PER_STEP * t + 1
def idx_g2(t: int) -> int: return BITS_PER_STEP * t + 2
def idx_c(t: int)  -> int: return BITS_PER_STEP * t + 3
def idx_d(t: int)  -> int: return BITS_PER_STEP * t + 4

def n_qubits(T: int) -> int:
    return BITS_PER_STEP * T


def decode(x: np.ndarray, T: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x  = np.asarray(x, dtype=np.float64)
    B  = BITS_PER_STEP
    g0 = x[0::B][:T]
    g1 = x[1::B][:T]
    g2 = x[2::B][:T]
    c  = x[3::B][:T]
    d  = x[4::B][:T]
    g  = g0 * GRID_COEFFS[0] + g1 * GRID_COEFFS[1] + g2 * GRID_COEFFS[2]
    return g.copy(), c.copy(), d.copy()


def qubo_to_matrix(Q: Dict[Tuple[int, int], float], n: int) -> np.ndarray:
    M = np.zeros((n, n))
    for (i, j), v in Q.items():
        M[i, j] += v
    return M


def qubo_to_ising(
    Q: Dict[Tuple[int, int], float],
    n: int,
    offset: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    h = np.zeros(n)
    J = np.zeros((n, n))
    C = offset
    for (i, j), v in Q.items():
        if i == j:
            h[i] -= v / 2
            C    += v / 4
        else:
            J[i, j] += v / 4
            h[i]    -= v / 4
            h[j]    -= v / 4
            C       += v / 4
    return h, J, C


def op_cost(
    x: np.ndarray,
    apply_t: int,
    load_vec: np.ndarray,
    mg: MicrogridParams,
    price_vec: np.ndarray | None = None,
    scenarios = None,
) -> float:
    """Operating cost: grid + cycle + shed (post-hoc residual-as-shed when scenarios given)."""
    T = len(x) // BITS_PER_STEP
    if price_vec is None:
        price_vec = np.ones(apply_t)
    price_vec = np.asarray(price_vec, dtype=np.float64)
    g, c, d = decode(x, T)

    # Avg renewables across scenarios (post-hoc shed needs supply estimate)
    renew = np.zeros(apply_t)
    if scenarios is not None:
        for w, sv, wv in scenarios:
            renew += float(w) * (
                np.asarray(sv[:apply_t], dtype=np.float64)
              + np.asarray(wv[:apply_t], dtype=np.float64)
            )

    # Unserved load is priced post hoc. The power balance is enforced only to within
    # FEASIBILITY_TOL, which is approximately one grid quantum (P_G/7), so a schedule
    # can leave load unserved and still count as feasible; charging that shortfall at
    # C_shed is what keeps the comparison against the exactly-balanced MILP honest.
    cost = 0.0
    for t in range(apply_t):
        pr     = float(price_vec[t])
        supply = mg.P_G * float(g[t]) + mg.P_B * (float(d[t]) - float(c[t])) + float(renew[t])
        shed_t = max(0.0, float(load_vec[t]) - supply) if scenarios is not None else 0.0
        cost += DT * pr * (
            mg.C_grid  * mg.P_G * float(g[t])
          + mg.C_cycle * mg.P_B * (float(c[t]) + float(d[t]))
          + mg.C_shed  * shed_t
        )
    return cost


def soc_trajectory(
    c_arr: np.ndarray,
    d_arr: np.ndarray,
    mg: MicrogridParams,
    soc0: float,
) -> np.ndarray:
    soc   = []
    cur   = soc0
    step_c = (mg.P_B / mg.SOC_CAP) * DT * mg.eta_c
    step_d = (mg.P_B / mg.SOC_CAP) * DT / mg.eta_d
    for c, d in zip(c_arr, d_arr):
        cur = cur + step_c * float(c) - step_d * float(d)
        soc.append(cur)
    return np.array(soc)
