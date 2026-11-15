from __future__ import annotations
import sys, argparse, logging
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

import json
from microgrid.config import (N_STEPS, APPLY_STEPS, SCHEDULING_HORIZON, SOC_INIT,
                              DT, PRICE_VEC, FEASIBILITY_TOL, SOC_TOL, MILP_LAM_TRACK, make_mg)
from microgrid.scheduler import _physics_guard
from microgrid.preprocessing import preprocess_date
from microgrid.scenario_generation import generate_scenarios, scenarios_to_tuples
from microgrid.optimization_milp import milp_schedule_window, milp_daily
from microgrid.optimization_stride import run_stride_native_window
from microgrid.utils import soc_trajectory

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("stride_native")


def score_gcd(g_pu, c, d, load, price, scen, mg, apply_t):
    """op_cost over the deployed steps (grid + cycle + post-hoc shed), scenario-mean renew.
    Returns (cost, shed_energy) where shed_energy = sum of unserved load (a.u.)."""
    g = np.asarray(g_pu, float)[:apply_t]; c = np.asarray(c, float)[:apply_t]; d = np.asarray(d, float)[:apply_t]
    load = np.asarray(load, float)[:apply_t]; price = np.asarray(price, float)[:apply_t]
    renew = np.zeros(apply_t)
    for w, sv, wv in scen:
        renew += float(w) * (np.asarray(sv[:apply_t], float) + np.asarray(wv[:apply_t], float))
    supply = mg.P_G * g + mg.P_B * (d - c) + renew
    shed = np.maximum(0.0, load - supply)
    cost = float(np.sum(DT * price * (mg.C_grid * mg.P_G * g + mg.C_cycle * mg.P_B * (c + d) + mg.C_shed * shed)))
    return cost, float(np.sum(shed))


def is_feasible(g_pu, c, d, load, scen, soc0, mg, apply_t):
    """Per-scenario balance (|res|<=tol) + SOC bounds + mutex, matching feasibility.py."""
    g = np.asarray(g_pu, float)[:apply_t]; c = np.asarray(c, float)[:apply_t]; d = np.asarray(d, float)[:apply_t]
    load = np.asarray(load, float)[:apply_t]
    if np.any((c > 0.5) & (d > 0.5)):
        return False, "mutex"
    soc = soc_trajectory(c, d, mg, soc0)
    if np.any(soc < mg.soc_min - SOC_TOL) or np.any(soc > mg.soc_max + SOC_TOL):
        return False, "soc"
    supply_common = mg.P_G * g + mg.P_B * (d - c)
    for w, sv, wv in scen:
        renew_sc = np.asarray(sv[:apply_t], float) + np.asarray(wv[:apply_t], float)
        if np.any(np.abs(supply_common + renew_sc - load) > FEASIBILITY_TOL):
            return False, "balance"
    return True, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2018-03-03")
    ap.add_argument("--grid-levels", type=int, default=100)
    ap.add_argument("--time-limit-s", type=float, default=5.0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    mg = make_mg()
    pday = preprocess_date(args.date, force=args.force)
    sc = scenarios_to_tuples(generate_scenarios(pday))

    # daily SOC reference + battery policy (mirrors scheduler)
    load_fc = np.asarray(pday.load_fc_n, float)[:N_STEPS]
    solar_fc = np.asarray(pday.solar_ec_n, float).mean(axis=0)[:N_STEPS]
    wind_fc = np.asarray(pday.wind_ec_n, float).mean(axis=0)[:N_STEPS]
    try:
        soc_ref_full = milp_daily(load_fc, solar_fc, wind_fc, mg, price_vec=PRICE_VEC, soc0=SOC_INIT)
    except Exception:
        soc_ref_full = None
    net_fc = load_fc - solar_fc - wind_fc
    net_h = net_fc.reshape(24, APPLY_STEPS).mean(axis=1)
    lo, hi = float(np.percentile(net_h, 30)), float(np.percentile(net_h, 75))
    peak = int(np.argmax(net_h))
    stress = float(np.max(net_h)) / (float(np.mean(net_h)) + 1e-9)
    try:
        month = pd.to_datetime(pday.date).month
    except Exception:
        month = 6
    high_stress = (stress > 1.25) or (month in (6, 7, 8, 9))
    CT, PT = (0.80, 0.70) if high_stress else (0.75, 0.65)

    def policy(hr):  # hr 1-indexed -> (min_soc_end, reserve_target, mode)
        h = hr - 1
        if h < peak and net_h[h] <= lo:   return 0.55, CT, "charge"
        elif h < peak:                    return 0.45, PT, "preserve"
        elif net_h[h] >= hi:              return 0.15, 0.20, "discharge"
        else:                             return 0.25, 0.40, "recover"

    max_gain = mg.P_B * mg.eta_c * APPLY_STEPS * DT / mg.SOC_CAP

    scale = float(getattr(pday, "scale", 1.0) or 1.0)
    soc_st = {"milp": SOC_INIT, "stride": SOC_INIT}
    tot = {"milp": 0.0, "stride": 0.0}
    shed_tot = {"milp": 0.0, "stride": 0.0}
    nfeas = {"milp": 0, "stride": 0}
    load_tot = 0.0
    nwin = 0
    rows_sched, rows_cmp = [], []

    def emit_sched(hr, t0, g, c, d, soc0_w, wl, wsol, wwin, solver):
        soc = soc_trajectory(np.asarray(c)[:APPLY_STEPS], np.asarray(d)[:APPLY_STEPS], mg, soc0_w)
        for t in range(APPLY_STEPS):
            supply = mg.P_G * float(g[t]) + mg.P_B * (float(d[t]) - float(c[t]))
            shed_t = max(0.0, float(wl[t]) - supply - (float(wsol[t]) + float(wwin[t])))
            rows_sched.append(dict(
                hour=hr, substep=t, global_step=t0 + t, solver=solver,
                load_mw=float(wl[t]) * scale, solar_mw=float(wsol[t]) * scale, wind_mw=float(wwin[t]) * scale,
                grid=round(float(g[t]), 4), charge=int(round(float(c[t]))), discharge=int(round(float(d[t]))),
                grid_mw=float(g[t]) * mg.P_G * scale, batt_mw=(float(d[t]) - float(c[t])) * mg.P_B * scale,
                soc=float(soc[t]), shed=round(shed_t, 4)))
        return float(np.clip(soc[-1], mg.soc_min, mg.soc_max))

    for step in range(0, N_STEPS, APPLY_STEPS):
        t0, t1 = step, min(step + SCHEDULING_HORIZON, N_STEPS)
        T = t1 - t0
        hr = step // APPLY_STEPS + 1
        wl = pday.load_n[t0:t1]
        wsol = np.asarray(pday.solar_n[t0:t1], float)
        wwin = np.asarray(pday.wind_n[t0:t1], float)
        wsc = [(float(w), sv[t0:t1], wv[t0:t1]) for (w, sv, wv) in sc]
        pw = PRICE_VEC[t0:t1]
        mse0, reserve_t, mode = policy(hr)
        soc_ref_end = float(soc_ref_full[t1 - 1]) if soc_ref_full is not None and t1 - 1 < len(soc_ref_full) else None
        if soc_ref_end is not None and mode != "discharge":
            mse0 = max(mse0, soc_ref_end - 0.15)
        if mode == "discharge":
            mse0 = max(mg.soc_min - 0.05, mse0 * 0.5)

        def clamp(soc):
            return min(mse0, min(mg.soc_max, float(soc) + max_gain) - 1e-6)

        # MILP baseline (same as the production scheduler): daily-plan soc_ref tracking
        # + physics guard. Reproduces the 16.34 total.
        m_soc = soc_st["milp"]
        ml = milp_schedule_window(wl, wsc, mg, m_soc, pw, min_soc_end=clamp(m_soc),
                                  soc_ref_end=soc_ref_end, lam_track=MILP_LAM_TRACK)
        if ml is not None:
            g_ml = np.asarray(ml["g"], float) / mg.P_G          # normalize to [0,1]
            c_ml, d_ml = np.asarray(ml["u_c"], float), np.asarray(ml["u_d"], float)
            g_ml, c_ml, d_ml = _physics_guard(g_ml, c_ml, d_ml, m_soc, mg, APPLY_STEPS)
            cM, shM = score_gcd(g_ml, c_ml, d_ml, wl, pw, wsc, mg, APPLY_STEPS)
            okM, _ = is_feasible(g_ml, c_ml, d_ml, wl, wsc, m_soc, mg, APPLY_STEPS)
            tot["milp"] += cM; shed_tot["milp"] += shM; nfeas["milp"] += int(okM)
            soc_st["milp"] = emit_sched(hr, t0, g_ml, c_ml, d_ml, m_soc, wl, wsol, wwin, "milp")
            rows_cmp.append(dict(solver="milp", hour=hr, cost=cM, is_feasible=bool(okM), shed=shM, mode=mode))
        load_tot += float(np.sum(np.asarray(wl, float)[:APPLY_STEPS]))

        # native Stride
        s_soc = soc_st["stride"]
        gv, cv, dv, _c, diag = run_stride_native_window(wl, wsc, mg, s_soc, pw, T,
                                                        grid_levels=args.grid_levels,
                                                        time_limit_s=args.time_limit_s,
                                                        min_soc_end=clamp(s_soc),
                                                        soc_ref_end=soc_ref_end,
                                                        battery_mode=mode,
                                                        soc_reserve_target=reserve_t)
        if gv is not None:
            cS, shS = score_gcd(gv, cv, dv, wl, pw, wsc, mg, APPLY_STEPS)
            okS, why = is_feasible(gv, cv, dv, wl, wsc, s_soc, mg, APPLY_STEPS)
            tot["stride"] += cS; shed_tot["stride"] += shS; nfeas["stride"] += int(okS)
            soc_st["stride"] = emit_sched(hr, t0, gv, cv, dv, s_soc, wl, wsol, wwin, "stride_native")
            rows_cmp.append(dict(solver="stride_native", hour=hr, cost=cS, is_feasible=bool(okS), shed=shS, mode=mode))
            nc, nd = int(np.sum(np.rint(cv[:APPLY_STEPS]))), int(np.sum(np.rint(dv[:APPLY_STEPS])))
            log.info(f"Hr{hr:02d} [{mode:9s}] MILP={tot['milp']:.3f} Stride={tot['stride']:.3f} | "
                     f"soc={soc_st['stride']:.3f} chg={nc} dis={nd} feas={okS}({why}) shed={shS:.3f}")
        nwin += 1

    # ---- save in standard format ----
    out = REPO_ROOT / "results" / "stride_native" / "by_date" / args.date / "tables"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_sched).to_csv(out / "df_schedule.csv", index=False)
    pd.DataFrame(rows_cmp).to_csv(out / "df_compare.csv", index=False)
    meta = dict(date=args.date, solver="stride_native", grid_levels=args.grid_levels,
                time_limit_s=args.time_limit_s, windows=nwin, scale=scale,
                milp_total=tot["milp"], stride_total=tot["stride"],
                milp_shed=shed_tot["milp"], stride_shed=shed_tot["stride"],
                milp_feasible=nfeas["milp"], stride_feasible=nfeas["stride"],
                deployed_load=load_tot)
    with open(out / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    log.info("=" * 60)
    log.info(f"date={args.date}  windows={nwin}  grid_levels={args.grid_levels}")
    log.info(f"  deployed load = {load_tot:.3f} a.u.")
    log.info(f"  MILP   total={tot['milp']:.4f}  feasible={nfeas['milp']}/{nwin}  shed={shed_tot['milp']:.4f} ({100*shed_tot['milp']/load_tot:.3f}% of load)")
    log.info(f"  Stride total={tot['stride']:.4f}  feasible={nfeas['stride']}/{nwin}  shed={shed_tot['stride']:.4f} ({100*shed_tot['stride']/load_tot:.3f}% of load)")
    log.info(f"  native vs MILP = {(tot['stride']-tot['milp'])/tot['milp']*100:+.1f}%   vs QUBO-Stride(20.1483) = {(tot['stride']-20.1483)/20.1483*100:+.1f}%")
    log.info(f"  saved -> {out}")


if __name__ == "__main__":
    main()
