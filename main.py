from __future__ import annotations

import argparse
import logging

from microgrid.config import QAOA_SHOTS, STUDY_DATES
from microgrid.pipeline import run_single_date, run_multiple_dates

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quantum microgrid rolling-horizon dispatch pipeline")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--date", help="Run one date (YYYY-MM-DD)")
    mode.add_argument("--all", action="store_true", help="Run all STUDY_DATES")
    parser.add_argument("--dates", nargs="+", help="Run specific dates (YYYY-MM-DD ...)")
    parser.add_argument("--force", action="store_true", help="Force re-preprocessing and re-run")
    parser.add_argument("--no-qaoa", action="store_true", help="Skip the QAOA solver")
    parser.add_argument("--no-sqa", action="store_true", help="Skip the SQA solver")
    parser.add_argument("--exact", action="store_true",
                        help="Run the exhaustive-enumeration QUBO baseline (n<=24). "
                             "Returns the cheapest dispatch the encoding admits, so it "
                             "separates encoding error from solver and repair error.")
    parser.add_argument("--baselines", nargs="*", default=None,
                        metavar="NAME",
                        choices=["sa", "tabu", "greedy", "random"],
                        help="Classical BQM baselines to run through the same QUBO, "
                             "repair and selection path as SQA. 'random' is the control "
                             "that isolates the contribution of the repair stage.")
    parser.add_argument("--ibm-backend", default=None,
                        help="IBM backend name for QAOA hardware execution (e.g. ibm_kingston). "
                             "If omitted, uses the local AerSimulator.")
    parser.add_argument("--ibm-shots", type=int, default=QAOA_SHOTS,
                        help="Shots per QAOA window on an IBM backend (default 16384)")
    parser.add_argument("--qaoa-p-layers", type=int, nargs="+", default=None,
                        help="QAOA circuit depth(s) to run, e.g. --qaoa-p-layers 3 or 2 3 5 "
                             "(single-date mode only). Defaults to config.QAOA_P_SWEEP.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    backend_kwargs = dict(
        ibm_backend=args.ibm_backend, ibm_shots=args.ibm_shots,
    )

    if args.date:
        run_single_date(date=args.date, force_preprocess=args.force,
                        enable_qaoa=not args.no_qaoa, enable_sqa=not args.no_sqa,
                        enable_exact=args.exact, bqm_baselines=args.baselines,
                        qaoa_p_layers=args.qaoa_p_layers, **backend_kwargs)
        return

    if args.all or args.dates:
        dates = args.dates if args.dates else STUDY_DATES
        run_multiple_dates(dates=dates, force_preprocess=args.force,
                           enable_qaoa=not args.no_qaoa, enable_sqa=not args.no_sqa,
                           enable_exact=args.exact, bqm_baselines=args.baselines,
                           skip_existing=not args.force, **backend_kwargs)
        return

    build_parser().print_help()


if __name__ == "__main__":
    main()
