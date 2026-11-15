"""Exact solution of a dispatch-window QUBO by exhaustive enumeration.

At the instance size used throughout this study the QUBO has
``n = BITS_PER_STEP * T = 20`` binary variables, so the entire feasible space of
$2^{20}$ assignments can be enumerated. That makes two quantities available that no
heuristic solver can supply:

  1. the true QUBO ground state, which bounds how much of any solver's cost gap is
     attributable to the solver rather than to the QUBO encoding itself; and
  2. the cheapest feasible dispatch reachable under the encoding, which bounds the
     encoding penalty relative to the continuous MILP.

Together these separate three distinct sources of the observed MILP-to-quantum gap:
encoding (MILP optimum vs. exact QUBO optimum), solver (exact optimum vs. solver raw
output), and repair (solver raw output vs. committed schedule).

Energies are evaluated in chunks via BLAS. Feasibility, SOC trajectories and operating
costs are evaluated vectorized across all candidate assignments, mirroring the scalar
definitions in ``microgrid.feasibility`` and ``microgrid.utils``.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple

from microgrid.config import (
    APPLY_STEPS, BITS_PER_STEP, DT, FEASIBILITY_TOL, GRID_COEFFS, SOC_TOL,
    MicrogridParams,
)

# 2**24 assignments x 24 float64 columns is ~3 GB of working set; refuse beyond that
# rather than exhaust memory mid-run.
MAX_EXACT_QUBITS = 24

# Assignments evaluated per BLAS call. Keeps the working set near 40 MB at n=20.
CHUNK = 1 << 17


def _bit_matrix(n: int, lo: int, hi: int) -> np.ndarray:
    """Rows lo..hi of the 2**n x n binary counting matrix."""
    idx = np.arange(lo, hi, dtype=np.int64)[:, None]
    shifts = np.arange(n, dtype=np.int64)[None, :]
    return ((idx >> shifts) & 1).astype(np.float64)


def _qubo_matrix(Q: Dict[Tuple[int, int], float], n: int) -> np.ndarray:
    M = np.zeros((n, n), dtype=np.float64)
    for (i, j), v in Q.items():
        M[i, j] += v
    return M


def _decode(X: np.ndarray, T: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized counterpart of utils.decode. X is (N, n) -> three (N, T) arrays."""
    B = BITS_PER_STEP
    g = (X[:, 0::B][:, :T] * GRID_COEFFS[0]
         + X[:, 1::B][:, :T] * GRID_COEFFS[1]
         + X[:, 2::B][:, :T] * GRID_COEFFS[2])
    return g, X[:, 3::B][:, :T], X[:, 4::B][:, :T]


def _feasible_mask(
    g: np.ndarray, c: np.ndarray, d: np.ndarray,
    load_vec: np.ndarray,
    scenarios: List[Tuple[float, np.ndarray, np.ndarray]],
    mg: MicrogridParams, soc0: float, eval_t: int, tol: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Expected-value and per-scenario feasibility over the first ``eval_t`` steps.

    Returns (expected_ok, scenario_ok), each a boolean mask over candidate assignments.
    Matches feasibility.check_feasibility: balance within ``tol``, no simultaneous
    charge/discharge, and SOC inside bounds with slack ``SOC_TOL``.
    """
    lv = np.asarray(load_vec[:eval_t], dtype=np.float64)
    gt, ct, dt_ = g[:, :eval_t], c[:, :eval_t], d[:, :eval_t]

    common = mg.P_G * gt + mg.P_B * (dt_ - ct)

    avg_renew = np.zeros(eval_t, dtype=np.float64)
    for w, sv, wv in scenarios:
        avg_renew += float(w) * (np.asarray(sv[:eval_t], dtype=np.float64)
                                      + np.asarray(wv[:eval_t], dtype=np.float64))

    balance_ok = np.all(np.abs(common + avg_renew - lv) <= tol, axis=1)
    mutex_ok = ~np.any((ct > 0.5) & (dt_ > 0.5), axis=1)

    step = (mg.P_B / mg.SOC_CAP) * DT
    soc = soc0 + np.cumsum(step * (mg.eta_c * ct - dt_ / mg.eta_d), axis=1)
    soc_ok = np.all((soc >= mg.soc_min - SOC_TOL) & (soc <= mg.soc_max + SOC_TOL), axis=1)

    expected_ok = balance_ok & mutex_ok & soc_ok

    scenario_ok = expected_ok.copy()
    for _, sv, wv in scenarios:
        renew = (np.asarray(sv[:eval_t], dtype=np.float64)
                 + np.asarray(wv[:eval_t], dtype=np.float64))
        scenario_ok &= np.all(np.abs(common + renew - lv) <= tol, axis=1)

    return expected_ok, scenario_ok


def _op_cost(
    g: np.ndarray, c: np.ndarray, d: np.ndarray,
    load_vec: np.ndarray, mg: MicrogridParams,
    price_vec: np.ndarray, avg_renew: np.ndarray, apply_t: int,
) -> np.ndarray:
    """Vectorized counterpart of utils.op_cost, including post-hoc shortfall as shed."""
    gt, ct, dt_ = g[:, :apply_t], c[:, :apply_t], d[:, :apply_t]
    pr = np.asarray(price_vec[:apply_t], dtype=np.float64)
    lv = np.asarray(load_vec[:apply_t], dtype=np.float64)

    supply = mg.P_G * gt + mg.P_B * (dt_ - ct) + avg_renew
    shed = np.maximum(0.0, lv - supply)

    return (DT * pr * (mg.C_grid * mg.P_G * gt
                       + mg.C_cycle * mg.P_B * (ct + dt_)
                       + mg.C_shed * shed)).sum(axis=1)


def run_exact_window(
    Q: Dict,
    n: int,
    offset: float,
    T: int,
    load_vec: np.ndarray,
    scenarios: List[Tuple[float, np.ndarray, np.ndarray]],
    mg: MicrogridParams,
    soc0: float,
    price_vec: np.ndarray,
    min_soc_end: Optional[float] = None,
    tol: float = FEASIBILITY_TOL,
) -> Tuple[Optional[np.ndarray], float, Dict]:
    """Enumerate every assignment and return the cheapest feasible dispatch.

    Signature and return contract match ``optimization_sqa.run_sqa_window`` so this
    drops into the same scheduler path: ``(best_x, best_op_cost, diagnostics)``.

    Selection mirrors the heuristic solvers: among assignments feasible in expectation
    over the commit horizon and the full window, take the minimum operating cost,
    preferring candidates that also meet the terminal SOC floor. The diagnostics carry
    the unconditioned QUBO ground state for encoding-versus-solver attribution.
    """
    if n > MAX_EXACT_QUBITS:
        raise ValueError(
            f"exhaustive enumeration needs n <= {MAX_EXACT_QUBITS}, got n={n} "
            f"({2**n:.3g} assignments)"
        )

    M = _qubo_matrix(Q, n)
    total = 1 << n

    avg_renew = np.zeros(APPLY_STEPS, dtype=np.float64)
    for w, sv, wv in scenarios:
        avg_renew += float(w) * (np.asarray(sv[:APPLY_STEPS], dtype=np.float64)
                                      + np.asarray(wv[:APPLY_STEPS], dtype=np.float64))

    chg_f = (mg.P_B / mg.SOC_CAP) * DT * mg.eta_c
    dis_f = (mg.P_B / mg.SOC_CAP) * DT / mg.eta_d

    ground_energy, ground_state = np.inf, None
    best = {"cost": np.inf, "x": None, "energy": np.inf}
    best_soc = {"cost": np.inf, "x": None, "energy": np.inf}
    n_expected = n_scenario = 0

    for lo in range(0, total, CHUNK):
        hi = min(lo + CHUNK, total)
        X = _bit_matrix(n, lo, hi)

        energy = np.einsum("ij,ij->i", X @ M, X) + offset

        k = int(np.argmin(energy))
        if energy[k] < ground_energy:
            ground_energy, ground_state = float(energy[k]), X[k].copy()

        g, c, d = _decode(X, T)

        commit_ok, _ = _feasible_mask(g, c, d, load_vec, scenarios, mg,
                                      soc0, APPLY_STEPS, tol)
        window_ok, scen_ok = _feasible_mask(g, c, d, load_vec, scenarios, mg,
                                            soc0, T, tol)
        ok = commit_ok & window_ok
        n_expected += int(ok.sum())
        n_scenario += int((ok & scen_ok).sum())
        if not ok.any():
            continue

        idx = np.flatnonzero(ok)
        cost = _op_cost(g[idx], c[idx], d[idx], load_vec, mg,
                        price_vec, avg_renew, APPLY_STEPS)

        j = int(np.argmin(cost))
        if cost[j] < best["cost"]:
            best = {"cost": float(cost[j]), "x": X[idx[j]].copy(),
                    "energy": float(energy[idx[j]])}

        if min_soc_end is not None:
            soc_end = np.clip(
                soc0 + (chg_f * c[idx, :APPLY_STEPS] - dis_f * d[idx, :APPLY_STEPS]).sum(axis=1),
                mg.soc_min, mg.soc_max,
            )
            keep = np.flatnonzero(soc_end >= float(min_soc_end) - 1e-9)
            if keep.size:
                m = keep[int(np.argmin(cost[keep]))]
                if cost[m] < best_soc["cost"]:
                    best_soc = {"cost": float(cost[m]), "x": X[idx[m]].copy(),
                                "energy": float(energy[idx[m]])}

    chosen = best_soc if best_soc["x"] is not None else best

    g_gs, c_gs, d_gs = _decode(ground_state[None, :], T)
    gs_commit, _ = _feasible_mask(g_gs, c_gs, d_gs, load_vec, scenarios, mg,
                                  soc0, APPLY_STEPS, tol)

    diag = {
        "n_assignments":          total,
        "n_expected_feasible":    n_expected,
        "n_scenario_feasible":    n_scenario,
        "n_infeasible":           total - n_expected,
        "ground_energy":          ground_energy,
        "ground_state_feasible":  bool(gs_commit[0]),
        "best_energy":            chosen["energy"],
        # Excess QUBO energy of the committed schedule over the unconstrained ground
        # state: how much the feasibility and SOC-floor requirements cost in the
        # QUBO's own objective.
        "energy_above_ground":    (chosen["energy"] - ground_energy
                                   if chosen["x"] is not None else float("nan")),
        "soc_floor_binding":      best_soc["x"] is not None
                                  and best["cost"] < best_soc["cost"] - 1e-12,
        "used_viol_fallback":     False,
    }

    if chosen["x"] is None:
        return None, np.inf, diag

    return chosen["x"].astype(np.float64), chosen["cost"], diag
