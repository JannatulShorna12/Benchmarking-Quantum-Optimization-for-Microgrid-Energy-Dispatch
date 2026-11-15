from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Optional

from microgrid.config import (APPLY_STEPS, DT, MicrogridParams,
                              FEASIBILITY_TOL, LAM_CHARGE_RECOVERY, MILP_LAM_TRACK)
from microgrid.utils import op_cost, idx_c, idx_d
from microgrid.feasibility import check_feasibility, fast_violations
from microgrid.optimization_dwave import _load_config

STRIDE_SOLVER_DEFAULT = "hybrid_nonlinear_program_version1p"

_SAMPLER_CACHE: dict = {}


def _stride_solver_id(cfg: dict) -> str:
    return cfg.get("solver", {}).get("stride", STRIDE_SOLVER_DEFAULT)


def _get_stride_sampler(cfg: dict):
    solver_id = _stride_solver_id(cfg)
    key = ("stride", solver_id)
    if key in _SAMPLER_CACHE:
        return _SAMPLER_CACHE[key]
    from dwave.system import LeapHybridNLSampler
    sampler = LeapHybridNLSampler(
        token=cfg["token"],
        endpoint=cfg.get("endpoint"),
        solver=solver_id,
    )
    _SAMPLER_CACHE[key] = sampler
    return sampler


def _build_nl_model(Q: Dict, n: int):
    """QUBO dict (diag (i,i)=linear, off-diag=quadratic) -> dwave.optimization model.

    quadratic_model(x, Q) builds objective sum_ij Q[i,j] x_i x_j exactly.
    Returns (model, x) with model.minimize already set.
    """
    from dwave.optimization import Model
    model = Model()
    x = model.binary(n)
    model.minimize(model.quadratic_model(x, Q))
    return model, x


def _qubo_energy(x: np.ndarray, Q: Dict, offset: float) -> float:
    e = float(offset)
    for (i, j), v in Q.items():
        e += float(v) * float(x[i]) * float(x[j])
    return e


def run_stride_window(
    Q: Dict,
    n: int,
    offset: float,
    T: int,
    load_vec: np.ndarray,
    scenarios: List[Tuple[float, np.ndarray, np.ndarray]],
    mg: MicrogridParams,
    soc0: float,
    price_vec: np.ndarray,
    time_limit_s: Optional[float] = None,
    min_soc_end: Optional[float] = None,
    **_ignored,
) -> Tuple[Optional[np.ndarray], float, dict]:
    """Run one scheduling window on D-Wave Stride (LeapHybridNLSampler).

    Returns (best_x, best_op_cost, diag_info), same contract as run_sqa_window.
    time_limit_s: None -> SDK-estimated minimum; below the floor it is clamped.
    """
    cfg = _load_config()
    defaults = cfg.get("defaults", {})
    if time_limit_s is None:
        time_limit_s = defaults.get("stride_time_limit_s")  # may stay None -> SDK default

    model, x = _build_nl_model(Q, n)
    sampler = _get_stride_sampler(cfg)

    min_t = None
    try:
        min_t = float(sampler.estimated_min_time_limit(model))
    except Exception:
        pass

    used_t = time_limit_s
    if used_t is not None and min_t is not None and used_t < min_t:
        used_t = min_t  # solver won't honor a shorter runtime

    sample_kwargs = {"label": f"Microgrid-Stride n={n}"}
    if used_t is not None:
        sample_kwargs["time_limit"] = used_t

    future = sampler.sample(model, **sample_kwargs)
    result = future.result()  # block until solved; populates model states

    best_x = None
    best_energy = np.inf
    best_op_cost = np.inf
    best_x_socok = None
    best_energy_socok = np.inf
    best_op_cost_socok = np.inf
    n_strict = 0
    n_expected = 0
    viol_list: list = []

    _chg_f = (mg.P_B / mg.SOC_CAP) * DT * mg.eta_c
    _dis_f = (mg.P_B / mg.SOC_CAP) * DT / mg.eta_d

    def _predict_soc_end(xb):
        soc = float(soc0)
        for tt in range(APPLY_STEPS):
            soc = soc + _chg_f * float(xb[idx_c(tt)]) - _dis_f * float(xb[idx_d(tt)])
        return float(np.clip(soc, mg.soc_min, mg.soc_max))

    n_states = int(model.states.size())
    for k in range(n_states):
        raw = np.asarray(x.state(k), dtype=np.float64).reshape(-1)
        xb = np.rint(raw).astype(np.float64)  # binary 0/1
        energy = _qubo_energy(xb, Q, offset)

        rep_c = check_feasibility(xb, T, load_vec, scenarios, mg, soc0=soc0, eval_t=APPLY_STEPS)
        rep_w = check_feasibility(xb, T, load_vec, scenarios, mg, soc0=soc0)

        if rep_c["is_feasible"] and rep_w["is_feasible"]:
            n_strict += 1

        if rep_c["expected_feasible"] and rep_w["expected_feasible"]:
            n_expected += 1
            cur_cost = float(op_cost(xb, APPLY_STEPS, load_vec, mg, price_vec=price_vec, scenarios=scenarios))
            cost_better = cur_cost < best_op_cost * (1.0 - 1e-3)
            cost_tied = abs(cur_cost - best_op_cost) <= abs(best_op_cost) * 1e-3
            if cost_better or (cost_tied and energy < best_energy):
                best_energy = float(energy)
                best_op_cost = cur_cost
                best_x = xb.copy()
            if min_soc_end is not None:
                _se = _predict_soc_end(xb)
                if _se >= float(min_soc_end) - 1e-9:
                    sb_better = cur_cost < best_op_cost_socok * (1.0 - 1e-3)
                    sb_tied = abs(cur_cost - best_op_cost_socok) <= abs(best_op_cost_socok) * 1e-3
                    if sb_better or (sb_tied and energy < best_energy_socok):
                        best_energy_socok = float(energy)
                        best_op_cost_socok = cur_cost
                        best_x_socok = xb.copy()
        else:
            viol = fast_violations(xb, T, load_vec, scenarios, mg, soc0)
            viol_list.append(viol)

    if best_x_socok is not None:
        best_x = best_x_socok
        best_energy = best_energy_socok
        best_op_cost = best_op_cost_socok

    timing = {}
    try:
        info = getattr(result, "info", None) or {}
        timing = info.get("timing", info) or {}
    except Exception:
        pass

    diag_info = {
        "dwave_backend": "stride",
        "dwave_solver": _stride_solver_id(cfg),
        "time_limit_s": used_t,
        "min_time_limit_s": min_t,
        "n_states": n_states,
        "n_reads": n_states,
        "n_strictly_feasible": n_strict,
        "n_expected_feasible": n_expected,
        "n_infeasible": n_states - n_expected,
        "viol_min": int(min(viol_list)) if viol_list else 0,
        "viol_mean": float(np.mean(viol_list)) if viol_list else 0.0,
        "best_energy": float(best_energy),
        "timing": timing,
        "used_viol_fallback": False,
    }

    if best_x is None:
        return None, np.inf, diag_info

    return best_x, float(best_op_cost), diag_info


def run_stride_native_window(
    load_vec: np.ndarray,
    scenarios: List[Tuple[float, np.ndarray, np.ndarray]],
    mg: MicrogridParams,
    soc0: float,
    price_vec: np.ndarray,
    T: int,
    grid_levels: int = 100,
    time_limit_s: Optional[float] = None,
    min_soc_end: Optional[float] = None,
    soc_ref_end: Optional[float] = None,
    battery_mode: Optional[str] = None,
    soc_reserve_target: Optional[float] = None,
) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray, float, dict]:
    """Best-case Stride: NATIVE dwave.optimization model (not a QUBO).

    grid = integer 0..grid_levels (g_pu = g/grid_levels, ~0.01 resolution -> kills
    the 8-level discretization gap), battery = binary on/off, with NATIVE balance /
    SOC / mutex constraints + shed slack. Models the full IBM/QAOA/MILP stack:
      (1) soft soc_ref daily-plan tracking (MILP_LAM_TRACK),
      (2) per-scenario (stochastic) balance robustness across all scenarios,
      (3) charge-recovery reward in charge mode (LAM_CHARGE_RECOVERY).
    Returns (g_pu[T], c[T], d[T], window_cost, diag).
    """
    import dwave.optimization as dno
    from dwave.optimization import Model
    cfg = _load_config()
    defaults = cfg.get("defaults", {})
    if time_limit_s is None:
        time_limit_s = defaults.get("stride_time_limit_s")

    GL = int(grid_levels)
    mean_re = np.zeros(T)
    for w, sv, wv in scenarios:
        mean_re += float(w) * (np.asarray(sv[:T], float) + np.asarray(wv[:T], float))

    chg_f = (mg.P_B / mg.SOC_CAP) * DT * mg.eta_c
    dis_f = (mg.P_B / mg.SOC_CAP) * DT / mg.eta_d
    price = np.asarray(price_vec[:T], float)

    SL = 1000  # shed resolution
    load_w = np.asarray(load_vec[:T], float)
    shed_cap = mg.shed_frac * load_w           # max sheddable load per step
    m = Model()
    g = m.integer(T, lower_bound=0, upper_bound=GL)   # grid level
    c = m.binary(T)                                    # charge on/off
    d = m.binary(T)                                    # discharge on/off
    s = m.integer(T, lower_bound=0, upper_bound=SL)    # shed slack (like MILP) -> always feasible
    g_pu = g * m.constant(1.0 / GL)
    s_pu = s * m.constant(shed_cap / SL)               # 0..shed_cap
    supply = g_pu * m.constant(np.full(T, mg.P_G)) + (d - c) * m.constant(np.full(T, mg.P_B))

    delta = c * m.constant(np.full(T, chg_f)) - d * m.constant(np.full(T, dis_f))
    soc_exprs = [delta[:tt + 1].sum() + m.constant(float(soc0)) for tt in range(T)]
    soc_end_expr = soc_exprs[-1]

    cost = (m.constant(DT * price * mg.C_grid * mg.P_G) * g_pu
            + m.constant(DT * price * mg.C_cycle * mg.P_B) * (c + d)
            + m.constant(DT * price * mg.C_shed) * s_pu).sum()
    if battery_mode == "charge" and soc_reserve_target is not None and soc0 < float(soc_reserve_target):
        reward = LAM_CHARGE_RECOVERY * (float(soc_reserve_target) - float(soc0))
        cost = cost - (m.constant(DT * price * reward) * c).sum()
    if soc_ref_end is not None:
        cost = cost + m.constant(float(MILP_LAM_TRACK)) * dno.absolute(
            soc_end_expr - m.constant(float(soc_ref_end)))
    m.minimize(cost)

    m.add_constraint(((c + d) <= m.constant(np.ones(T))).all())
    for w, sv, wv in scenarios:
        renew_sc = np.asarray(sv[:T], float) + np.asarray(wv[:T], float)
        m.add_constraint((supply + m.constant(renew_sc) + s_pu >= m.constant(load_w)).all())
    for tt in range(T):
        m.add_constraint(soc_exprs[tt] <= m.constant(float(mg.soc_max)))
        m.add_constraint(soc_exprs[tt] >= m.constant(float(mg.soc_min)))
    if min_soc_end is not None:
        m.add_constraint(soc_end_expr >= m.constant(float(min_soc_end)))

    sampler = _get_stride_sampler(cfg)
    try:
        min_t = float(sampler.estimated_min_time_limit(m))
    except Exception:
        min_t = None
    used_t = time_limit_s
    if used_t is not None and min_t is not None and used_t < min_t:
        used_t = min_t
    kw = {"label": f"Microgrid-Stride-Native T={T}"}
    if used_t is not None:
        kw["time_limit"] = used_t
    sampler.sample(m, **kw).result()

    best = None
    best_cost = np.inf
    for k in range(int(m.states.size())):
        gv = np.asarray(g.state(k), float) / GL
        cv = np.rint(np.asarray(c.state(k), float))
        dv = np.rint(np.asarray(d.state(k), float))
        supply_k = mg.P_G * gv + mg.P_B * (dv - cv) + mean_re
        shed_k = np.maximum(0.0, load_w - supply_k)
        cst = float(np.sum(DT * price * (mg.C_grid * mg.P_G * gv
                                         + mg.C_cycle * mg.P_B * (cv + dv)
                                         + mg.C_shed * shed_k)))
        if cst < best_cost:
            best_cost, best = cst, (gv, cv, dv)

    diag = {"backend": "stride_native", "grid_levels": GL, "time_limit_s": used_t,
            "min_time_limit_s": min_t, "n_states": int(m.states.size())}
    if best is None:
        return None, np.zeros(T), np.zeros(T), np.inf, diag
    gv, cv, dv = best
    return gv, cv, dv, float(best_cost), diag


def run_stride_sweep(
    Q: Dict,
    n: int,
    offset: float,
    T: int,
    load_vec: np.ndarray,
    scenarios: List[Tuple[float, np.ndarray, np.ndarray]],
    mg: MicrogridParams,
    soc0: float,
    price_vec: np.ndarray,
    time_limits_s: List[float],
    min_soc_end: Optional[float] = None,
) -> Tuple[dict, List[dict]]:
    """Run one window at several time_limits, pick best.

    "Best" = lowest feasible op-cost; ties (<=0.1%) broken by shorter time.
    Returns (best_record, all_records). Each record:
        {time_limit_s, op_cost, best_energy, n_states, n_expected_feasible, feasible}
    """
    records: List[dict] = []
    for t in sorted(set(float(v) for v in time_limits_s)):
        bx, cost, diag = run_stride_window(
            Q=Q, n=n, offset=offset, T=T, load_vec=load_vec, scenarios=scenarios,
            mg=mg, soc0=soc0, price_vec=price_vec, time_limit_s=t, min_soc_end=min_soc_end,
        )
        records.append({
            "time_limit_s": t,
            "used_time_limit_s": diag.get("time_limit_s"),
            "op_cost": cost,
            "best_energy": diag.get("best_energy"),
            "n_states": diag.get("n_states"),
            "n_expected_feasible": diag.get("n_expected_feasible"),
            "feasible": bx is not None,
        })

    feas = [r for r in records if r["feasible"]]
    pool = feas if feas else records
    best = min(pool, key=lambda r: (r["op_cost"], r["time_limit_s"]))
    return best, records
