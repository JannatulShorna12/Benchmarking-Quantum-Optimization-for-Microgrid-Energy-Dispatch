"""Model, QUBO and solver settings. Symbols and section numbers refer to the NAPS 2026 paper.

All power and energy values are per unit of P_scale = 73,262 MW (Section IV), which is
computed from the data in microgrid/preprocessing.py.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from dataclasses import dataclass
import os as _os

REPO_ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR      = REPO_ROOT / "data"
DATA_PROC     = REPO_ROOT / "data" / "processed"
DATA_FORECAST = REPO_ROOT / "data" / "forecast"
RESULTS_DIR   = REPO_ROOT / "results"

DATASET_DIR  = Path(_os.environ.get("DATASET_DIR", str(DATA_DIR)))
DATA_FORCAST = DATA_FORECAST

# Six operating days spanning the seasons; 24 rolling windows each = 144 windows per solver
STUDY_DATES = [
    "2018-03-03", "2018-04-13", "2018-04-14",
    "2018-06-27", "2018-07-23", "2018-10-26",
]

TARGET_DT_MIN = 15                 # 15-minute resolution
DT            = 0.25               # Delta t (h)
SCHEDULING_HORIZON   = 4           # substeps t = 0..3 per dispatch window
APPLY_STEPS   = 4                  # substeps committed per window (one hour)
N_HOURS       = 24
N_STEPS       = N_HOURS * (60 // TARGET_DT_MIN)
N_SCENARIOS   = 3                  # k-means medoids of the 51-member ECMWF ensemble, w_k = n_k/51

# Microgrid (Section II and IV)
P_GRID        = 1.00               # P_G, grid-import limit (pu)
P_BATT        = 0.15               # P_B, battery power rating (pu)
SOC_CAP_HOURS = 4.0
SOC_CAP       = P_BATT * SOC_CAP_HOURS   # SOC_cap = 0.60 pu h
ETA_C         = 0.95               # eta_c
ETA_D         = 0.95               # eta_d
SHED_FRAC     = 0.00   # no shed decision in any formulation; unserved
                       # load is penalised post hoc at C_SHED instead
SOC_MIN       = 0.10               # SOC_min
SOC_MAX       = 0.90               # SOC_max
SOC_INIT      = 0.50               # SOC_0

P_GRID_FRAC = P_GRID
P_BATT_FRAC = P_BATT

# Cost coefficients (Section IV)
C_GRID  = 2.0                      # C_grid
C_SHED  = 8.0                      # C_shed
C_CYCLE = 0.018                    # C_cycle, battery degradation per unit throughput

SOC_RESERVE = 0.50                 # SOC_reserve in H_reserve
LAM_RESERVE = 0.50                 # lambda_v, reserve term H_reserve

# Time-of-use price pi_t: off-peak 0.7, morning peak 1.4, evening peak 1.6, standard 1.0
_p_h = np.ones(24)
_p_h[0:4]   = 0.7
_p_h[7:10]  = 1.4
_p_h[17:20] = 1.6
PRICE_VEC = np.repeat(_p_h, 4)

# QUBO encoding (Section III-A): g_t = P_G (1/7 b0 + 2/7 b1 + 4/7 b2), 8 levels.
# Five binaries per substep (three grid bits, c_t, d_t) -> n = 20 qubits per window.
GRID_COEFFS   = [1/7, 2/7, 4/7]
BITS_PER_STEP = 5

# QUBO penalty weights (Section III-B): H = H_cost + H_bal + H_mutex + H_SOC + H_reserve + H_track
LAM_BALANCE = 150.0                # lambda_b, scenario-weighted power balance
LAM_MUTEX   = 120.0                # lambda_m, charge/discharge mutual exclusion
LAM_SOC     = 250.0                # lambda_s, cumulative SOC tracking
LAM_TRACK   = 6.0                  # lambda_k, day-ahead terminal SOC tracking

# Feasibility repair (Section III-D): S1 fix battery and scan the 8 grid levels,
# S2 battery idle and re-scan, S3 full greedy search, with a Hamming anchor lambda_H.
# The IBM Kingston runs used LAM_REPAIR_HAMMING = 0.10 and LAM_REPAIR_BALANCE = 180.0;
# the D-Wave and classical-sampler runs used the defaults below. Set either value
# through the environment variable of the same name.
LAM_REPAIR_BALANCE = float(_os.environ.get("LAM_REPAIR_BALANCE", 10.0))
LAM_REPAIR_HAMMING = float(_os.environ.get("LAM_REPAIR_HAMMING", 2.0))   # lambda_H
ERR_MAX_REPAIR     = 0.15

LAM_CHARGE_RECOVERY = 0.08

LAM_DEFICIT_DISCHARGE = 10.0
LAM_SURPLUS_CHARGE    = 10.0
LAM_UNNEEDED_BATT     = 5.0

FEASIBILITY_TOL = 0.15
SOC_TOL         = 0.05

# QAOA (Section III-C): COBYLA on a noiseless simulator with INTERP warm start,
# then 16,384 shots per window on IBM Kingston at p = 2 and p = 3.
QAOA_P_SWEEP  = [2]
QAOA_RESTARTS = 8                  # COBYLA restarts
QAOA_SHOTS    = 16384              # shots per dispatch window
QAOA_MAX_ITER = 60                 # COBYLA maxiter

# Hardware used in the paper (Table I). IBM calibration values are the backend properties
# at run time. D-Wave cost results (Table II, Figs. 2-3) come from Advantage2_system1; the
# runtime study (Section IV-C) used Advantage_system6.4.
IBM_HARDWARE = {
    "backend": "ibm_kingston", "processor": "Heron r2", "topology": "heavy-hex",
    "qubits": 156, "median_t1_us": 173, "cz_error": 2.1e-3, "readout_error": 1.5e-2,
    "shots_per_window": QAOA_SHOTS, "transpile_optimization_level": 3,
}
DWAVE_HARDWARE = {
    "cost_runs": {"solver": "Advantage2_system1", "topology": "Zephyr",
                  "num_reads": 1000, "annealing_time_us": 20},
    "runtime_study": {"solver": "Advantage_system6.4", "topology": "Pegasus", "qubits": 5627,
                      "num_reads": 4000, "annealing_time_us": 100},
    "embedding": "minorminer", "chain_strength": "uniform_torque_compensation",
    "chain_break": "majority vote",
}
# Repair weights of the IBM Kingston runs (Section III-D); see LAM_REPAIR_* above
IBM_RUN_REPAIR = {"LAM_REPAIR_BALANCE": 180.0, "LAM_REPAIR_HAMMING": 0.10}

SQA_READS      = 3000
SQA_SWEEPS     = 5000
SQA_HD_START   = 2.0
SQA_HD_END     = 0.0
SQA_BETA_START = 0.1
SQA_BETA_END   = 10.0

# Reads per window for the classical BQM baselines (sa / tabu / greedy / random).
# Matched to the D-Wave QPU read count so the sample pools being selected from are
# comparable across solvers.
BQM_BASELINE_READS = 1000

# Seed passed to every stochastic sampler (SQA and the classical baselines). Fixing it
# makes a run reproducible: without it the sample pool depends on how much of the
# global RNG stream earlier solvers happened to consume, so results were not
# reproducible across different solver combinations.
SAMPLER_SEED = 42

# MILP baseline (Section III-E): CBC branch-and-bound
MILP_TIME_LIMIT = 30       # CBC solver time limit (seconds) per scheduling window
MILP_GAP_REL    = 1e-6     # CBC relative MIP gap tolerance
MILP_LAM_TRACK  = 2.0      # L1 penalty weight: MILP window SOC tracking vs daily ref

SOLVER_COLORS = {
    "milp":         "#0173B2",
    "qaoa_p2":      "#DE8F05",
    "qaoa_p3":      "#F77F00",
    "qaoa_p5":      "#D62828",
    "ibm_p2":       "#F77F00",
    "ibm_p3":       "#D62828",
    "ibm_p5":       "#9B2226",
    "sqa":          "#CC78BC",
    "dwave_qa":     "#029E73",
    "dwave_hybrid": "#56B4E9",
}
SOLVER_LABELS = {
    "milp":         "MILP (optimal baseline)",
    "qaoa_p2":      "QAOA p=2 (Simulator)",
    "qaoa_p3":      "QAOA p=3",
    "qaoa_p5":      "QAOA p=5",
    "ibm_p2":       "IBM Kingston p=2,3",
    "ibm_p3":       "IBM Kingston p=3",
    "ibm_p5":       "IBM Kingston p=5",
    "sqa":          "SQA",
    "dwave_qa":     "D-Wave QPU",
    # Stride/Leap Hybrid NLP is a classical-quantum hybrid on a non-QUBO nonlinear
    # model - a formulation reference, not a QPU result. Label must not imply otherwise.
    "dwave_hybrid": "D-Wave Stride (hybrid NLP ref.)",
}
FIGURE_DPI  = 150
FIGURE_SIZE = (10, 4)


@dataclass
class MicrogridParams:
    P_G:       float = P_GRID
    P_B:       float = P_BATT
    SOC_CAP:   float = SOC_CAP
    eta_c:     float = ETA_C
    eta_d:     float = ETA_D
    shed_frac: float = SHED_FRAC
    soc_min:   float = SOC_MIN
    soc_max:   float = SOC_MAX
    C_grid:    float = C_GRID
    C_shed:    float = C_SHED
    C_cycle:   float = C_CYCLE


def make_mg() -> MicrogridParams:
    return MicrogridParams()


def results_dir_for_date(date: str) -> Path:
    d = RESULTS_DIR / "by_date" / date
    for sub in ("tables", "figures", "logs"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d


def processed_dir_for_date(date: str) -> Path:
    d = DATA_PROC / date
    d.mkdir(parents=True, exist_ok=True)
    return d
