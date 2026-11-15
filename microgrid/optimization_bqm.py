"""Classical BQM samplers run through the same pipeline as the quantum solvers.

Each baseline uses the same QUBO, the same max-abs normalisation, the same
feasible-then-minimum-cost selection over the sample pool, and the same staged repair
as the quantum samplers, so only the sampler differs.

Samplers from the ``dwave-samplers`` family:

    sa      SimulatedAnnealingSampler   classical annealing
    tabu    TabuSampler                 tabu search
    greedy  SteepestDescentSampler      deterministic local search
    random  RandomSampler               shot-matched control

``random`` is the control that isolates the repair stage: if random sampling followed by
repair matches a hardware sampler followed by repair, the repair is doing the work.

``optimization_sqa.run_sqa_window`` remains the reference implementation for the
selection logic; ``select_best`` here is factored out of it so both share one definition.
"""
from __future__ import annotations

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple

from microgrid.config import (APPLY_STEPS, DT, MicrogridParams, SAMPLER_SEED,
                              SQA_READS)
from microgrid.utils import idx_c, idx_d, op_cost
from microgrid.feasibility import (batch_expected_feasible,
                                   batch_strictly_feasible, fast_violations)

# Sampler registry. Each entry builds a sampler and the kwargs for one sample() call.
# num_reads is passed through so every baseline draws the same number of samples as the
# quantum runs it is compared against.
SAMPLERS: Dict[str, Callable[[], object]] = {}


def _register(name: str, factory: Callable[[], object]) -> None:
    SAMPLERS[name] = factory


def _sa():
    from dwave.samplers import SimulatedAnnealingSampler
    return SimulatedAnnealingSampler()


def _tabu():
    from dwave.samplers import TabuSampler
    return TabuSampler()


def _greedy():
    from dwave.samplers import SteepestDescentSampler
    return SteepestDescentSampler()


def _random():
    from dwave.samplers import RandomSampler
    return RandomSampler()


_register("sa", _sa)
_register("tabu", _tabu)
_register("greedy", _greedy)
_register("random", _random)

# RandomSampler takes num_reads only; the others accept num_sweeps as well.
_ACCEPTS_SWEEPS = {"sa", "tabu", "greedy"}

_CACHE: Dict[str, object] = {}


def get_sampler(name: str):
    if name not in SAMPLERS:
        raise ValueError(f"unknown sampler {name!r}; available: {sorted(SAMPLERS)}")
    if name not in _CACHE:
        _CACHE[name] = SAMPLERS[name]()
    return _CACHE[name]


def build_normalised_bqm(Q: Dict, offset: float):
    """BQM scaled by the largest absolute bias. Returns (bqm, scale).

    Identical to the normalisation in optimization_sqa.run_sqa_window: annealing and
    tabu are sensitive to bias magnitude, and the penalty weights here reach O(200)
    against unit cost terms. Energies are rescaled by ``scale`` on the way out.
    """
    import dimod

    bqm = dimod.BinaryQuadraticModel("BINARY")
    for (i, j), v in Q.items():
        if i == j:
            bqm.add_variable(i, v)
        else:
            bqm.add_interaction(i, j, v)
    bqm.offset = offset

    biases = list(bqm.linear.values()) + list(bqm.quadratic.values())
    scale = max((abs(v) for v in biases), default=1.0)
    if scale < 1e-12:
        scale = 1.0

    scaled = dimod.BinaryQuadraticModel("BINARY")
    for v, bias in bqm.linear.items():
        scaled.add_variable(v, bias / scale)
    for (u, v), bias in bqm.quadratic.items():
        scaled.add_interaction(u, v, bias / scale)
    scaled.offset = bqm.offset / scale

    return scaled, scale


def select_best(
    samples,
    n: int,
    T: int,
    load_vec: np.ndarray,
    scenarios: List[Tuple[float, np.ndarray, np.ndarray]],
    mg: MicrogridParams,
    soc0: float,
    price_vec: np.ndarray,
    scale: float,
    min_soc_end: Optional[float] = None,
) -> Tuple[Optional[np.ndarray], float, Dict]:
    """Pick the committed dispatch from a sample pool.

    Mirrors optimization_sqa.run_sqa_window exactly: keep samples feasible in
    expectation over both the commit horizon and the full window, take the minimum
    operating cost, break ties within 0.1% by QUBO energy, and prefer candidates that
    also clear the terminal SOC floor.

    Note that selection is by operating cost, not by energy. Because the penalty terms
    dominate the QUBO objective, the minimum-energy sample is generally not the
    cheapest feasible dispatch, so this step performs cost optimisation that the
    sampler itself does not.
    """
    chg_f = (mg.P_B / mg.SOC_CAP) * DT * mg.eta_c
    dis_f = (mg.P_B / mg.SOC_CAP) * DT / mg.eta_d

    records = list(samples.data(["sample", "energy"]))
    n_reads = len(records)
    if n_reads == 0:
        return None, np.inf, {"n_reads": 0, "n_unique_samples": 0,
                              "n_strictly_feasible": 0, "n_expected_feasible": 0,
                              "n_infeasible": 0, "viol_min": 0, "viol_mean": 0.0,
                              "best_energy": float("inf"), "used_viol_fallback": False}

    X = np.array([[rec.sample.get(i, 0) for i in range(n)] for rec in records],
                 dtype=np.float64)
    energies = np.array([rec.energy for rec in records], dtype=np.float64) * scale

    # Expected-value feasibility over both horizons, evaluated for the whole pool at
    # once. The per-sample scalar checker dominates runtime at thousands of reads.
    ok = (batch_expected_feasible(X, T, load_vec, scenarios, mg, soc0, APPLY_STEPS)
          & batch_expected_feasible(X, T, load_vec, scenarios, mg, soc0, T))
    feasible_idx = np.flatnonzero(ok)
    n_expected = int(feasible_idx.size)

    best_x = best_x_soc = None
    best_cost = best_cost_soc = np.inf
    best_energy = best_energy_soc = np.inf

    # Strict feasibility (every individual scenario balances, not only the weighted
    # mean) is a reported diagnostic. Computed for the pool at once: on high-renewable
    # days almost every sample is feasible, so a per-sample scalar check here dominated
    # the runtime of the whole window.
    n_strict = int((batch_strictly_feasible(X, T, load_vec, scenarios, mg,
                                            soc0, APPLY_STEPS)
                    & batch_strictly_feasible(X, T, load_vec, scenarios, mg,
                                              soc0, T)).sum())

    # Iterate the feasible subset in sample order so the tie-breaking is unchanged:
    # minimum operating cost, ties within 0.1% broken by lower QUBO energy.
    for i in feasible_idx:
        x = X[i]
        energy = float(energies[i])

        cost = float(op_cost(x, APPLY_STEPS, load_vec, mg,
                             price_vec=price_vec, scenarios=scenarios))

        better = cost < best_cost * (1.0 - 1e-3)
        tied = abs(cost - best_cost) <= abs(best_cost) * 1e-3
        if better or (tied and energy < best_energy):
            best_cost, best_energy, best_x = cost, energy, x.copy()

        if min_soc_end is not None:
            soc_end = float(np.clip(
                soc0 + sum(chg_f * x[idx_c(t)] - dis_f * x[idx_d(t)]
                           for t in range(APPLY_STEPS)),
                mg.soc_min, mg.soc_max))
            if soc_end >= float(min_soc_end) - 1e-9:
                better_s = cost < best_cost_soc * (1.0 - 1e-3)
                tied_s = abs(cost - best_cost_soc) <= abs(best_cost_soc) * 1e-3
                if better_s or (tied_s and energy < best_energy_soc):
                    best_cost_soc, best_energy_soc = cost, energy
                    best_x_soc = x.copy()

    seen = {X[i].tobytes() for i in range(n_reads)}
    viols = [fast_violations(X[i], T, load_vec, scenarios, mg, soc0)
             for i in np.flatnonzero(~ok)]

    if best_x_soc is not None:
        best_x, best_cost, best_energy = best_x_soc, best_cost_soc, best_energy_soc

    diag = {
        "n_reads":             n_reads,
        "n_unique_samples":    len(seen),
        "n_strictly_feasible": n_strict,
        "n_expected_feasible": n_expected,
        "n_infeasible":        n_reads - n_expected,
        "viol_min":            int(min(viols)) if viols else 0,
        "viol_mean":           float(np.mean(viols)) if viols else 0.0,
        "best_energy":         float(best_energy),
        "used_viol_fallback":  False,
    }

    if best_x is None:
        return None, np.inf, diag
    return best_x, float(best_cost), diag


def run_bqm_window(
    sampler_name: str,
    Q: Dict,
    n: int,
    offset: float,
    T: int,
    load_vec: np.ndarray,
    scenarios: List[Tuple[float, np.ndarray, np.ndarray]],
    mg: MicrogridParams,
    soc0: float,
    price_vec: np.ndarray,
    num_reads: int = SQA_READS,
    num_sweeps: int = 1000,
    min_soc_end: Optional[float] = None,
    seed: Optional[int] = SAMPLER_SEED,
) -> Tuple[Optional[np.ndarray], float, Dict]:
    """Sample one dispatch window with a named classical sampler.

    Return contract matches optimization_sqa.run_sqa_window and
    optimization_exact.run_exact_window: ``(best_x, best_op_cost, diagnostics)``.
    """
    bqm, scale = build_normalised_bqm(Q, offset)

    kwargs: Dict[str, object] = {"num_reads": num_reads}
    if sampler_name in _ACCEPTS_SWEEPS and sampler_name != "greedy":
        kwargs["num_sweeps"] = num_sweeps
    if seed is not None:
        kwargs["seed"] = seed

    response = get_sampler(sampler_name).sample(bqm, **kwargs)

    x, cost, diag = select_best(response, n, T, load_vec, scenarios, mg,
                                soc0, price_vec, scale, min_soc_end=min_soc_end)
    diag["sampler"] = sampler_name
    diag["bqm_scale"] = scale
    return x, cost, diag
