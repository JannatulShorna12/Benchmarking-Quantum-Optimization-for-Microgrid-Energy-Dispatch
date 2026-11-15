"""Run D-Wave QA (Leap QPU) across study dates.

Reuses scheduling pipeline + repair stack from run_single_date, but swaps the SQA
sampler call for D-Wave QPU (real quantum annealing). Outputs land in a
separate folder tree so SQA / QAOA / MILP baselines stay untouched.

Token: dwave_config.json (or env DWAVE_API_TOKEN).
"""

from __future__ import annotations

import sys
import shutil
import logging
import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DWAVE_RESULTS_DIR = REPO_ROOT / "results" / "dwave_qa"
DWAVE_FIG_DIR     = REPO_ROOT / "results" / "figures" / "dwave_qa"

import microgrid.config as _config
import microgrid.scheduler as _scheduler
import microgrid.optimization_sqa as _opt_sqa

_config.RESULTS_DIR = DWAVE_RESULTS_DIR
_scheduler.RESULTS_DIR = DWAVE_RESULTS_DIR

from microgrid.optimization_dwave import run_dwave_window

_QPU_NUM_READS:   int  = 1000
_QPU_ANNEAL_US:   int  = 20
_HYBRID_TIME_S:   float = 3.0
_BACKEND:         str  = "qpu"


def _qpu_sqa_shim(Q, n, offset, T, load_vec, scenarios, mg, soc0,
                  price_vec, num_reads=None, num_sweeps=None, min_soc_end=None):
    """Drop-in replacement for run_sqa_window - routes to D-Wave QPU or hybrid."""
    return run_dwave_window(
        Q=Q, n=n, offset=offset, T=T,
        load_vec=load_vec, scenarios=scenarios,
        mg=mg, soc0=soc0, price_vec=price_vec,
        backend=_BACKEND,
        num_reads=num_reads or _QPU_NUM_READS,
        annealing_time_us=_QPU_ANNEAL_US,
        hybrid_time_limit_s=_HYBRID_TIME_S,
        min_soc_end=min_soc_end,
    )


_opt_sqa.run_sqa_window = _qpu_sqa_shim
_scheduler.run_sqa_window = _qpu_sqa_shim

from microgrid.config            import STUDY_DATES
from microgrid.pipeline import run_single_date
from microgrid.plotting          import generate_date_figures
from microgrid.config            import make_mg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dwave_qa")


def _rename_sqa_to_dwave(date: str) -> None:
    """Rewrite df_* so solver label reflects active D-Wave backend."""
    label = "dwave_hybrid" if _BACKEND == "hybrid" else "dwave_qa"
    rd = DWAVE_RESULTS_DIR / "by_date" / date / "tables"
    if not rd.exists():
        return
    import pandas as pd
    for fname in ("df_compare.csv", "df_schedule.csv", "df_diagnostics.csv"):
        fpath = rd / fname
        if not fpath.exists():
            continue
        df = pd.read_csv(fpath)
        if "solver" in df.columns:
            df["solver"] = df["solver"].replace({"sqa": label})
            df.to_csv(fpath, index=False)
    meta_path = rd / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            m = json.load(f)
        m["dwave_backend"] = _BACKEND
        from microgrid.optimization_dwave import _load_config as _lc
        _qpu_id = _lc().get("solver", {}).get("qpu", "Advantage2_system1")
        m["dwave_solver"]  = "hybrid_binary_quadratic_model_version2p" if _BACKEND == "hybrid" else _qpu_id
        if _BACKEND == "qpu":
            m["num_reads"]     = _QPU_NUM_READS
            m["annealing_time_us"] = _QPU_ANNEAL_US
        else:
            m["hybrid_time_limit_s"] = _HYBRID_TIME_S
        with open(meta_path, "w") as f:
            json.dump(m, f, indent=2)


def _mirror_figures(date: str) -> None:
    src = DWAVE_RESULTS_DIR / "by_date" / date / "figures"
    if not src.exists():
        return
    dst = DWAVE_FIG_DIR / date
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        shutil.copy2(f, dst / f.name)


def run(date: str, force: bool = False) -> None:
    logger.info(f"D-Wave QA run - date={date}")
    run_single_date(
        date=date,
        force_preprocess=force,
        enable_qaoa=False,
        enable_sqa=True,
        ibm_backend=None,
    )
    _rename_sqa_to_dwave(date)
    _mirror_figures(date)
    logger.info(f"  tables  -> {DWAVE_RESULTS_DIR / 'by_date' / date}")
    logger.info(f"  figures -> {DWAVE_FIG_DIR / date}")


def _parse_args():
    p = argparse.ArgumentParser(description="D-Wave QA (Leap QPU) scheduler")
    p.add_argument("--dates", nargs="+", default=None,
                   help="YYYY-MM-DD list (default: all STUDY_DATES)")
    p.add_argument("--force", action="store_true",
                   help="Force re-preprocessing")
    p.add_argument("--num-reads", type=int, default=1000,
                   help="QPU shots per window (default 1000)")
    p.add_argument("--annealing-time-us", type=int, default=20,
                   help="QPU annealing time microseconds (default 20)")
    p.add_argument("--backend", choices=["qpu", "hybrid"], default="qpu",
                   help="D-Wave backend: qpu (Advantage) or hybrid (LeapHybridSampler)")
    p.add_argument("--hybrid-time-s", type=float, default=3.0,
                   help="Hybrid solver time_limit per window (s, default 3)")
    p.add_argument("--results-dir", default=None,
                   help="Folder under results/ for the output (default dwave_qa, or "
                        "dwave_hybrid with --backend hybrid)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    _QPU_NUM_READS = args.num_reads
    _QPU_ANNEAL_US = args.annealing_time_us
    _HYBRID_TIME_S = args.hybrid_time_s
    _BACKEND       = args.backend

    _out = args.results_dir or ("dwave_hybrid" if _BACKEND == "hybrid" else "dwave_qa")
    DWAVE_RESULTS_DIR = REPO_ROOT / "results" / _out
    DWAVE_FIG_DIR     = REPO_ROOT / "results" / "figures" / _out
    _config.RESULTS_DIR = DWAVE_RESULTS_DIR
    _scheduler.RESULTS_DIR = DWAVE_RESULTS_DIR
    import microgrid.pipeline as _rsd
    _rsd.RESULTS_DIR = DWAVE_RESULTS_DIR

    DWAVE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DWAVE_FIG_DIR.mkdir(parents=True, exist_ok=True)

    dates = args.dates if args.dates else STUDY_DATES
    completed, failed = [], []
    for i, d in enumerate(dates):
        logger.info(f"\n[{i+1}/{len(dates)}] --- {d} ---")
        try:
            run(d, force=args.force)
            completed.append(d)
        except Exception as e:
            logger.error(f"  FAILED {d}: {e}", exc_info=True)
            failed.append(d)

    logger.info(f"\nDone. ok={len(completed)} fail={len(failed)}")
    if failed:
        logger.warning(f"Failed: {failed}")
