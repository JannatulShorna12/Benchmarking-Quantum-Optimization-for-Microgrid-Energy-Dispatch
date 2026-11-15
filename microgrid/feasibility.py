from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple

from microgrid.config import (BITS_PER_STEP, FEASIBILITY_TOL, GRID_COEFFS, SOC_TOL, DT,
                              MicrogridParams)
from microgrid.utils import decode


FeasibilityReport = Dict


def batch_decode(X: np.ndarray, T: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized decode. X is (N, n) -> (g, c, d) each (N, T)."""
    B = BITS_PER_STEP
    g = (X[:, 0::B][:, :T] * GRID_COEFFS[0]
         + X[:, 1::B][:, :T] * GRID_COEFFS[1]
         + X[:, 2::B][:, :T] * GRID_COEFFS[2])
    return g, X[:, 3::B][:, :T], X[:, 4::B][:, :T]


def batch_expected_feasible(
    X: np.ndarray,
    T: int,
    load_vec: np.ndarray,
    scenarios: List[Tuple[float, np.ndarray, np.ndarray]],
    mg: MicrogridParams,
    soc0: float,
    eval_t: int,
    tol: float = FEASIBILITY_TOL,
) -> np.ndarray:
    """Batch equivalent of ``check_feasibility(...)["expected_feasible"]``.

    Returns a boolean mask over the N candidate assignments in ``X``. The predicate is
    the same as the scalar version: expected-value power balance within ``tol`` at every
    step of the first ``eval_t``, no simultaneous charge and discharge, and the SOC path
    inside ``[soc_min - SOC_TOL, soc_max + SOC_TOL]``.

    Exists because selecting from a sample pool calls the scalar checker once per
    sample per horizon, which dominates runtime at thousands of reads per window.
    Agreement with the scalar implementation is asserted in
    ``scripts/verify_batch_feasibility.py``.
    """
    g, c, d = batch_decode(np.asarray(X, dtype=np.float64), T)
    gt, ct, dt_ = g[:, :eval_t], c[:, :eval_t], d[:, :eval_t]

    lv = np.asarray(load_vec[:eval_t], dtype=np.float64)
    avg_renew = np.zeros(eval_t, dtype=np.float64)
    for w, sv, wv in scenarios:
        avg_renew += float(w) * (np.asarray(sv[:eval_t], dtype=np.float64)
                                 + np.asarray(wv[:eval_t], dtype=np.float64))

    supply = mg.P_G * gt + mg.P_B * (dt_ - ct) + avg_renew
    balance_ok = np.all(np.abs(supply - lv) <= tol, axis=1)
    mutex_ok = ~np.any((ct > 0.5) & (dt_ > 0.5), axis=1)

    step = (mg.P_B / mg.SOC_CAP) * DT
    soc = soc0 + np.cumsum(step * (mg.eta_c * ct - dt_ / mg.eta_d), axis=1)
    soc_ok = np.all((soc >= mg.soc_min - SOC_TOL) & (soc <= mg.soc_max + SOC_TOL), axis=1)

    return balance_ok & mutex_ok & soc_ok


def batch_strictly_feasible(
    X: np.ndarray,
    T: int,
    load_vec: np.ndarray,
    scenarios: List[Tuple[float, np.ndarray, np.ndarray]],
    mg: MicrogridParams,
    soc0: float,
    eval_t: int,
    tol: float = FEASIBILITY_TOL,
) -> np.ndarray:
    """Batch equivalent of ``check_feasibility(...)["is_feasible"]``.

    Strict feasibility requires every individual scenario to balance, not only the
    weighted mean, in addition to the expected-value conditions. Strict feasibility
    implies expected feasibility, so this is the stronger predicate.
    """
    expected = batch_expected_feasible(X, T, load_vec, scenarios, mg, soc0, eval_t, tol)
    if not expected.any():
        return expected

    g, c, d = batch_decode(np.asarray(X, dtype=np.float64), T)
    gt, ct, dt_ = g[:, :eval_t], c[:, :eval_t], d[:, :eval_t]
    lv = np.asarray(load_vec[:eval_t], dtype=np.float64)
    common = mg.P_G * gt + mg.P_B * (dt_ - ct)

    step = (mg.P_B / mg.SOC_CAP) * DT
    soc = soc0 + np.cumsum(step * (mg.eta_c * ct - dt_ / mg.eta_d), axis=1)
    soc_ok = np.all((soc >= mg.soc_min - SOC_TOL) & (soc <= mg.soc_max + SOC_TOL), axis=1)

    strict = expected & soc_ok
    for _, sv, wv in scenarios:
        renew = (np.asarray(sv[:eval_t], dtype=np.float64)
                 + np.asarray(wv[:eval_t], dtype=np.float64))
        strict &= np.all(np.abs(common + renew - lv) <= tol, axis=1)

    return strict


def fast_violations(
    x: np.ndarray,
    T: int,
    load_vec: np.ndarray,
    scenarios: List[Tuple[float, np.ndarray, np.ndarray]],
    mg: MicrogridParams,
    soc0: float,
    tol: float = FEASIBILITY_TOL,
) -> int:
    g, c, d = decode(x, T)
    lv = np.asarray(load_vec[:T], dtype=np.float64)

    avg_renew = np.zeros(T)
    for w, sv, wv in scenarios:
        avg_renew += float(w) * (np.asarray(sv[:T], dtype=np.float64) +
                                 np.asarray(wv[:T], dtype=np.float64))

    supply_common = mg.P_G * g + mg.P_B * (d - c)
    residual_exp  = supply_common + avg_renew - lv
    n_bal = int(np.sum(np.abs(residual_exp) > tol))
    n_mut = int(np.sum((c > 0.5) & (d > 0.5)))

    step = (mg.P_B / mg.SOC_CAP) * DT
    delta_soc = step * (mg.eta_c * c - d / mg.eta_d)
    soc_path = soc0 + np.cumsum(delta_soc)
    n_soc = int(np.sum(soc_path < mg.soc_min - SOC_TOL)) + \
            int(np.sum(soc_path > mg.soc_max + SOC_TOL))

    n_scen = 0
    for w, sv, wv in scenarios:
        renew_sc = np.asarray(sv[:T], dtype=np.float64) + np.asarray(wv[:T], dtype=np.float64)
        res_sc = supply_common + renew_sc - lv
        n_scen += int(np.sum(np.abs(res_sc) > tol))
        n_scen += int(np.sum(soc_path < mg.soc_min - SOC_TOL))
        n_scen += int(np.sum(soc_path > mg.soc_max + SOC_TOL))

    return n_bal + n_mut + n_soc + n_scen


def check_feasibility(
    x: np.ndarray,
    T: int,
    load_vec: np.ndarray,
    scenarios: List[Tuple[float, np.ndarray, np.ndarray]],
    mg: MicrogridParams,
    soc0: float = 0.5,
    tol: float = FEASIBILITY_TOL,
    eval_t: int | None = None,
) -> FeasibilityReport:
    if eval_t is None:
        eval_t = T

    g, c, d = decode(x, T)

    avg_renew = [
        sum(float(w) * (float(sv[t]) + float(wv[t])) for w, sv, wv in scenarios)
        for t in range(eval_t)
    ]

    soc = soc0
    n_bal = n_mut = n_soc = 0
    residuals: List[float] = []
    soc_path: List[float] = []

    for t in range(eval_t):
        supply   = mg.P_G * float(g[t]) + mg.P_B * float(d[t]) - mg.P_B * float(c[t]) + avg_renew[t]
        residual = supply - float(load_vec[t])
        residuals.append(residual)
        if abs(residual) > tol:
            n_bal += 1

        if float(c[t]) > 0.5 and float(d[t]) > 0.5:
            n_mut += 1

        soc = soc + (mg.P_B / mg.SOC_CAP) * DT * (mg.eta_c * float(c[t]) - float(d[t]) / mg.eta_d)
        soc_path.append(soc)

        if soc < mg.soc_min - SOC_TOL:
            n_soc += 1
        if soc > mg.soc_max + SOC_TOL:
            n_soc += 1

    expected_feasible = (n_bal + n_mut + n_soc) == 0

    scenario_bal_violations: List[Dict] = []
    scenario_soc_violations: List[Dict] = []

    for w, sv, wv in scenarios:
        soc_sc = soc0
        for t in range(eval_t):
            renew_t = float(sv[t]) + float(wv[t])
            supply  = mg.P_G * float(g[t]) + mg.P_B * float(d[t]) - mg.P_B * float(c[t]) + renew_t
            res = abs(supply - float(load_vec[t]))
            if res > tol:
                scenario_bal_violations.append({"scenario": w, "timestep": t, "residual": res})

            soc_sc = soc_sc + (mg.P_B / mg.SOC_CAP) * DT * (mg.eta_c * float(c[t]) - float(d[t]) / mg.eta_d)
            if soc_sc < mg.soc_min - SOC_TOL or soc_sc > mg.soc_max + SOC_TOL:
                scenario_soc_violations.append({"scenario": w, "timestep": t, "soc": soc_sc})

    scenario_violations = scenario_bal_violations + scenario_soc_violations
    n_scen = len(scenario_violations)
    strict_total = (n_bal + n_mut + n_soc) + n_scen

    return dict(
        is_feasible=(strict_total == 0),
        expected_feasible=expected_feasible,
        scenario_feasible=(n_scen == 0),
        total_violations=strict_total,
        balance_violations=n_bal,
        mutex_violations=n_mut,
        soc_violations=n_soc,
        final_soc=soc,
        max_abs_balance_residual=float(np.max(np.abs(residuals))) if residuals else 0.0,
        soc_path=soc_path,
        scenario_bal_violations=scenario_bal_violations,
        scenario_soc_violations=scenario_soc_violations,
        scenario_violations=scenario_violations,
        eval_t=eval_t,
    )


def print_failure_report(
    hr: int,
    solver: str,
    rep: FeasibilityReport,
    x: np.ndarray,
    T: int,
    load_vec: np.ndarray,
    mg: MicrogridParams,
    soc0: float,
) -> None:
    g, c, d = decode(x, T)
    eval_t = rep.get("eval_t", T)
    scope  = "COMMIT" if eval_t == 1 else f"WINDOW(t=0..{eval_t-1})"
    t0_viols = [v for v in rep.get("scenario_violations", []) if v.get("timestep", 0) == 0]
    label    = "COMMIT-FAIL" if (t0_viols or rep["balance_violations"] > 0) else "TAIL-FAIL"

    print(
        f"  [{label}] HR{hr:02d} {solver} [{scope}]: "
        f"bal={rep['balance_violations']} mut={rep['mutex_violations']} "
        f"soc={rep['soc_violations']} max_res={rep['max_abs_balance_residual']:.4f}"
    )
    soc = soc0
    for t in range(eval_t):
        soc_before = soc
        soc = soc + (mg.P_B / mg.SOC_CAP) * DT * (mg.eta_c * float(c[t]) - float(d[t]) / mg.eta_d)
        print(f"    t={t}: g={float(g[t]):.2f} c={int(c[t])} d={int(d[t])}  SOC: {soc_before:.3f}->{soc:.3f}  load={float(load_vec[t]):.4f}")
