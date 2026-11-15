from __future__ import annotations
import json
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from microgrid.config import (
    APPLY_STEPS, DT, MicrogridParams, REPO_ROOT,
)
from microgrid.utils import op_cost, idx_c, idx_d
from microgrid.feasibility import check_feasibility, fast_violations

CONFIG_PATH = REPO_ROOT / "dwave_config.json"

_CLIENT_CACHE: dict = {}


def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"D-Wave config missing: {CONFIG_PATH}. "
            "Create it with your Leap API token."
        )
    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f)
    tok = cfg.get("token", "")
    if (not tok) or tok.startswith("PASTE_"):
        env_tok = os.environ.get("DWAVE_API_TOKEN", "")
        if env_tok:
            cfg["token"] = env_tok
        else:
            raise ValueError(
                "D-Wave token not set. Edit dwave_config.json or export DWAVE_API_TOKEN."
            )
    return cfg


def _build_bqm(Q: Dict, offset: float):
    import dimod
    bqm = dimod.BinaryQuadraticModel("BINARY")
    for (i, j), v in Q.items():
        if i == j:
            bqm.add_variable(i, v)
        else:
            bqm.add_interaction(i, j, v)
    bqm.offset = offset

    all_vals = list(bqm.linear.values()) + list(bqm.quadratic.values())
    max_abs = max(abs(v) for v in all_vals) if all_vals else 1.0
    if max_abs < 1e-12:
        max_abs = 1.0

    bqm_n = dimod.BinaryQuadraticModel("BINARY")
    for v, bias in bqm.linear.items():
        bqm_n.add_variable(v, bias / max_abs)
    for (u, v), bias in bqm.quadratic.items():
        bqm_n.add_interaction(u, v, bias / max_abs)
    bqm_n.offset = bqm.offset / max_abs
    return bqm_n, max_abs


def _get_qpu_sampler(cfg: dict):
    key = ("qpu", cfg["solver"]["qpu"])
    if key in _CLIENT_CACHE:
        return _CLIENT_CACHE[key]
    from dwave.system import DWaveSampler, EmbeddingComposite
    qpu = DWaveSampler(
        token=cfg["token"],
        endpoint=cfg.get("endpoint"),
        solver=cfg["solver"]["qpu"],
    )
    sampler = EmbeddingComposite(qpu)
    _CLIENT_CACHE[key] = sampler
    return sampler


def _get_hybrid_sampler(cfg: dict):
    key = ("hybrid", cfg["solver"]["hybrid"])
    if key in _CLIENT_CACHE:
        return _CLIENT_CACHE[key]
    from dwave.system import LeapHybridSampler
    sampler = LeapHybridSampler(
        token=cfg["token"],
        endpoint=cfg.get("endpoint"),
        solver=cfg["solver"]["hybrid"],
    )
    _CLIENT_CACHE[key] = sampler
    return sampler


def run_dwave_window(
    Q: Dict,
    n: int,
    offset: float,
    T: int,
    load_vec: np.ndarray,
    scenarios: List[Tuple[float, np.ndarray, np.ndarray]],
    mg: MicrogridParams,
    soc0: float,
    price_vec: np.ndarray,
    backend: str = "qpu",
    num_reads: Optional[int] = None,
    annealing_time_us: Optional[int] = None,
    hybrid_time_limit_s: Optional[float] = None,
    min_soc_end: Optional[float] = None,
) -> Tuple[Optional[np.ndarray], float, dict]:
    """Run BQM on D-Wave Leap cloud (QPU annealer or hybrid).

    backend: "qpu" -> DWaveSampler + EmbeddingComposite (real QA)
             "hybrid" -> LeapHybridSampler (hybrid classical+quantum)

    Mirrors run_sqa_window signature so scheduler can swap.
    """
    cfg = _load_config()
    defaults = cfg.get("defaults", {})
    num_reads = num_reads or defaults.get("num_reads", 1000)
    annealing_time_us = annealing_time_us or defaults.get("annealing_time_us", 20)
    hybrid_time_limit_s = hybrid_time_limit_s or defaults.get("hybrid_time_limit_s", 5.0)
    chain_scale = defaults.get("chain_strength_scale", 1.0)

    bqm_n, max_abs = _build_bqm(Q, offset)

    if backend == "qpu":
        from dwave.embedding.chain_strength import uniform_torque_compensation
        sampler = _get_qpu_sampler(cfg)
        chain_strength = chain_scale * uniform_torque_compensation(bqm_n)
        response = sampler.sample(
            bqm_n,
            num_reads=num_reads,
            annealing_time=annealing_time_us,
            chain_strength=chain_strength,
            label=f"Microgrid-Schedule n={n}",
        )
    elif backend == "hybrid":
        sampler = _get_hybrid_sampler(cfg)
        response = sampler.sample(
            bqm_n,
            time_limit=hybrid_time_limit_s,
            label=f"Microgrid-Schedule-Hybrid n={n}",
        )
    else:
        raise ValueError(f"backend must be 'qpu' or 'hybrid', got {backend!r}")

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

    total_samples = 0
    for rec in response.data(["sample", "energy", "num_occurrences"]):
        occ = int(getattr(rec, "num_occurrences", 1))
        total_samples += occ
        x = np.array([rec.sample.get(i, 0) for i in range(n)], dtype=np.float64)
        energy = rec.energy * max_abs

        rep_c = check_feasibility(x, T, load_vec, scenarios, mg, soc0=soc0, eval_t=APPLY_STEPS)
        rep_w = check_feasibility(x, T, load_vec, scenarios, mg, soc0=soc0)

        if rep_c["is_feasible"] and rep_w["is_feasible"]:
            n_strict += occ

        if rep_c["expected_feasible"] and rep_w["expected_feasible"]:
            n_expected += occ
            cur_cost = float(op_cost(x, APPLY_STEPS, load_vec, mg, price_vec=price_vec, scenarios=scenarios))
            cost_better = cur_cost < best_op_cost * (1.0 - 1e-3)
            cost_tied = abs(cur_cost - best_op_cost) <= abs(best_op_cost) * 1e-3
            if cost_better or (cost_tied and energy < best_energy):
                best_energy = float(energy)
                best_op_cost = cur_cost
                best_x = x.copy()
            if min_soc_end is not None:
                _se = _predict_soc_end(x)
                if _se >= float(min_soc_end) - 1e-9:
                    sb_better = cur_cost < best_op_cost_socok * (1.0 - 1e-3)
                    sb_tied = abs(cur_cost - best_op_cost_socok) <= abs(best_op_cost_socok) * 1e-3
                    if sb_better or (sb_tied and energy < best_energy_socok):
                        best_energy_socok = float(energy)
                        best_op_cost_socok = cur_cost
                        best_x_socok = x.copy()
        else:
            viol = fast_violations(x, T, load_vec, scenarios, mg, soc0)
            viol_list.append(viol)

    if best_x_socok is not None:
        best_x = best_x_socok
        best_energy = best_energy_socok
        best_op_cost = best_op_cost_socok

    timing = {}
    try:
        info = response.info or {}
        timing = info.get("timing", {}) or {}
        timing["problem_id"] = info.get("problem_id")
    except Exception:
        pass

    diag_info = {
        "dwave_backend": backend,
        "dwave_solver": cfg["solver"][backend],
        "n_reads": num_reads if backend == "qpu" else total_samples,
        "n_strictly_feasible": n_strict,
        "n_expected_feasible": n_expected,
        "n_infeasible": total_samples - n_expected,
        "viol_min": int(min(viol_list)) if viol_list else 0,
        "viol_mean": float(np.mean(viol_list)) if viol_list else 0.0,
        "best_energy": float(best_energy),
        "timing": timing,
        "used_viol_fallback": False,
    }

    if best_x is None:
        return None, np.inf, diag_info

    return best_x, float(best_op_cost), diag_info
