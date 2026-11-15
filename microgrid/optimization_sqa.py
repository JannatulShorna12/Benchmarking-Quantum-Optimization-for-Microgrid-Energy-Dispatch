"""Simulated quantum annealing (path-integral) for a dispatch-window QUBO.

SQA is the classical stand-in for the D-Wave annealer: it samples the same QUBO under a
transverse-field schedule, so it isolates the effect of the annealing heuristic from the
effect of running on hardware.

The QUBO construction, normalisation and sample selection are shared with the other
classical baselines in ``optimization_bqm``. Only the sampler and its transverse-field
schedule are specific to this module, which is what makes the SQA-versus-baseline
comparison attributable to the sampler.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple

from microgrid.config import (
    SQA_READS, SQA_SWEEPS, SQA_HD_START, SQA_HD_END,
    SQA_BETA_START, SQA_BETA_END, SAMPLER_SEED, MicrogridParams,
)
from microgrid.optimization_bqm import build_normalised_bqm, select_best

_SAMPLER = None


def _get_sampler():
    global _SAMPLER
    if _SAMPLER is None:
        from dwave.samplers import PathIntegralAnnealingSampler
        _SAMPLER = PathIntegralAnnealingSampler()
    return _SAMPLER


def run_sqa_window(
    Q: Dict,
    n: int,
    offset: float,
    T: int,
    load_vec: np.ndarray,
    scenarios: List[Tuple[float, np.ndarray, np.ndarray]],
    mg: MicrogridParams,
    soc0: float,
    price_vec: np.ndarray,
    num_reads:  int = SQA_READS,
    num_sweeps: int = SQA_SWEEPS,
    min_soc_end: Optional[float] = None,
    seed: Optional[int] = SAMPLER_SEED,
) -> Tuple[Optional[np.ndarray], float, Dict]:
    """Sample one dispatch window with path-integral annealing.

    Returns ``(best_x, best_op_cost, diagnostics)``, the contract shared with
    ``optimization_bqm.run_bqm_window`` and ``optimization_exact.run_exact_window``.

    The longitudinal field follows a geometric ramp and the transverse field a linear
    ramp to zero, both over ``num_sweeps``.
    """
    bqm, scale = build_normalised_bqm(Q, offset)

    Hp = list(np.geomspace(SQA_BETA_START, SQA_BETA_END, num_sweeps))
    Hd = list(np.linspace(SQA_HD_START, SQA_HD_END, num_sweeps))

    response = _get_sampler().sample(
        bqm, num_reads=num_reads, num_sweeps=num_sweeps,
        beta_schedule_type="custom", Hp_field=Hp, Hd_field=Hd,
        seed=seed,
    )

    x, cost, diag = select_best(response, n, T, load_vec, scenarios, mg,
                                soc0, price_vec, scale, min_soc_end=min_soc_end)
    diag["sampler"] = "sqa"
    diag["bqm_scale"] = scale
    return x, cost, diag
