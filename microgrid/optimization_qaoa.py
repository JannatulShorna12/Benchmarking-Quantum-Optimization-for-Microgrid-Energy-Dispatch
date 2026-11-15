from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Optional
from microgrid.config import (
    QAOA_SHOTS, QAOA_MAX_ITER, QAOA_RESTARTS, APPLY_STEPS,
    DT, GRID_COEFFS, SOC_TOL, LAM_RESERVE, LAM_TRACK,
    LAM_REPAIR_BALANCE, LAM_REPAIR_HAMMING,
    LAM_CHARGE_RECOVERY, MicrogridParams,
    LAM_DEFICIT_DISCHARGE, LAM_SURPLUS_CHARGE, LAM_UNNEEDED_BATT,
)
from microgrid.utils import qubo_to_ising, op_cost, idx_g0, idx_g1, idx_g2, idx_c, idx_d, decode
from microgrid.feasibility import check_feasibility, fast_violations


def _get_simulator():
    from qiskit_aer import AerSimulator
    return AerSimulator(method="statevector")


def _get_ibm_backend(backend_name: str):
    """Return IBM backend. Tries saved account first, then apikey.json in repo root."""
    from qiskit_ibm_runtime import QiskitRuntimeService
    import json, pathlib
    # Try apikey.json in repo root
    apikey_path = pathlib.Path(__file__).resolve().parent.parent / "apikey.json"
    if apikey_path.exists():
        token = json.loads(apikey_path.read_text()).get("apikey")
        service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
    else:
        service = QiskitRuntimeService(channel="ibm_quantum_platform")
    return service.backend(backend_name)


def _run_ibm_shots(qc_bound, backend, shots: int) -> np.ndarray:
    """Run bound circuit on IBM backend, return probability array over 2^n basis states."""
    from qiskit_ibm_runtime import SamplerV2 as Sampler
    try:
        from qiskit import transpile
    except ImportError:
        from qiskit.transpile import transpile
    qc_meas = qc_bound.copy()
    qc_meas.measure_all()
    qc_t = transpile(qc_meas, backend, optimization_level=3)
    sampler = Sampler(backend)
    job  = sampler.run([qc_t], shots=shots)
    # Hard timeout on the otherwise-unbounded result fetch: a dropped connection
    # whose fetch stalled for hours). SIGALRM raises so the caller's retry loop
    # re-submits instead of stalling. (Main-thread only; the solve runs there.)
    import signal as _signal
    def _ibm_timeout(signum, frame):
        raise TimeoutError("IBM job.result() exceeded 600s timeout")
    _old_handler = _signal.signal(_signal.SIGALRM, _ibm_timeout)
    _signal.alarm(600)
    try:
        result = job.result()
    finally:
        _signal.alarm(0)
        _signal.signal(_signal.SIGALRM, _old_handler)
    pub_result = result[0]
    counts = pub_result.data.meas.get_counts()
    n = qc_bound.num_qubits
    N = 1 << n
    prob = np.zeros(N, dtype=float)
    total = sum(counts.values())
    for bitstr, cnt in counts.items():
        # bitstring is big-endian in Qiskit: bit 0 is rightmost
        idx = int(bitstr[::-1], 2)
        prob[idx] += cnt / total
    return prob


def _build_circuit(h, J, n, p, save_sv: bool = True):
    """Build parameterized QAOA circuit. save_sv=False omits save_statevector for IBM hardware."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    qc     = QuantumCircuit(n)
    gammas = [Parameter(f"g{l}") for l in range(p)]
    betas  = [Parameter(f"b{l}") for l in range(p)]
    qc.h(range(n))
    for l in range(p):
        for i in range(n):
            if abs(h[i]) > 1e-12:
                qc.rz(2 * gammas[l] * h[i], i)
        for i in range(n):
            for j in range(i + 1, n):
                if abs(J[i, j]) > 1e-12:
                    qc.rzz(2 * gammas[l] * J[i, j], i, j)
        for i in range(n):
            qc.rx(2 * betas[l], i)
    if save_sv:
        qc.save_statevector()
    return qc, gammas, betas


def _ising_diag(h, J, n, C):
    N   = 2 ** n
    idx = np.arange(N)
    z   = np.array([1.0 - ((idx >> k) & 1) * 2.0 for k in range(n)])
    E   = np.zeros(N) + C
    for i in range(n):
        E += h[i] * z[i]
    for i in range(n):
        for j in range(i + 1, n):
            if abs(J[i, j]) > 1e-12:
                E += J[i, j] * z[i] * z[j]
    return E


# Repair helpers

def _avg_renew(scenarios: List[Tuple[float, np.ndarray, np.ndarray]], T: int) -> np.ndarray:
    ar = np.zeros(T)
    for w, sv, wv in scenarios:
        ar += float(w) * (np.asarray(sv[:T], float) + np.asarray(wv[:T], float))
    return ar


def _qubo_cost(
    x: np.ndarray, T: int, load_vec: np.ndarray, mg, price_vec: np.ndarray,
    soc0: float = None, soc_ref: np.ndarray = None,
) -> float:
    """QUBO surrogate objective used for candidate scoring.

    Combines the price-aware operating cost, a terminal SOC valuation term, and an
    optional L2 penalty tracking the daily SOC reference trajectory.
    """
    opc   = op_cost(x, T, load_vec, mg, price_vec=price_vec)
    dis_f = (mg.P_B / mg.SOC_CAP) * DT / mg.eta_d
    chg_f = (mg.P_B / mg.SOC_CAP) * DT * mg.eta_c
    lam_v = LAM_RESERVE * mg.C_grid * mg.P_G
    term  = 0.0
    for t in range(T):
        term += lam_v * dis_f * float(x[idx_d(t)]) - lam_v * chg_f * float(x[idx_c(t)])

    track = 0.0
    if soc_ref is not None and soc0 is not None and LAM_TRACK > 0.0:
        lam_k = LAM_TRACK * mg.C_grid * mg.P_G
        soc   = float(soc0)
        for t in range(T):
            soc = soc + chg_f * float(x[idx_c(t)]) - dis_f * float(x[idx_d(t)])
            track += lam_k * (soc - float(soc_ref[t])) ** 2
    return float(opc + term + track)


def _step_score(
    g0, g1, g2, cv, dv,
    load_t, pr_t, mg, lam_v, dis_f, chg_f,
    soc_after=None, soc_ref_t=None, lam_k=0.0,
    battery_mode=None, target_soc=None,
    net_wo_batt=None,
):
    """Per-step objective aligned with full QUBO: op_cost + terminal SOC value
    + LP-tracking penalty (when soc_ref provided). Charge recovery reward applied
    only during forecast 'charge' mode below target_soc."""
    grid_frac = g0 * GRID_COEFFS[0] + g1 * GRID_COEFFS[1] + g2 * GRID_COEFFS[2]
    s = (
        DT * pr_t * mg.C_grid  * mg.P_G * grid_frac
        + DT * pr_t * mg.C_cycle * mg.P_B * (cv + dv)
        + lam_v * dis_f * dv        # terminal SOC: discharge opportunity cost
        - lam_v * chg_f * cv        # terminal SOC: charging incentive
    )
    if lam_k > 0.0 and soc_after is not None and soc_ref_t is not None:
        s += lam_k * (float(soc_after) - float(soc_ref_t)) ** 2
    if (battery_mode == "charge"
            and target_soc is not None
            and soc_after is not None
            and cv == 1
            and float(soc_after) < float(target_soc)):
        s -= LAM_CHARGE_RECOVERY * (float(target_soc) - float(soc_after))

    # Substep deficit/surplus-aware battery logic.
    if net_wo_batt is not None:
        if net_wo_batt < -1e-6:
            # Deficit: prefer discharge, penalize charging.
            if dv == 1:
                s -= LAM_DEFICIT_DISCHARGE * abs(float(net_wo_batt))
            if cv == 1:
                s += LAM_UNNEEDED_BATT
        elif net_wo_batt > -0.10:
            # Low-deficit/headroom: good time to charge if SOC below target or max.
            if cv == 1 and soc_after is not None:
                if (target_soc is not None and float(soc_after) < float(target_soc)) or float(soc_after) < mg.soc_max:
                    s -= LAM_SURPLUS_CHARGE * (float(net_wo_batt) + 0.10)
            if dv == 1:
                s += LAM_UNNEEDED_BATT
        else:
            # Already balanced: avoid unnecessary cycling.
            if cv == 1 or dv == 1:
                s += LAM_UNNEEDED_BATT

    # Mild push toward target SOC when charging.
    if cv == 1 and target_soc is not None and soc_after is not None and float(soc_after) < float(target_soc):
        s -= 0.3 * (float(target_soc) - float(soc_after))

    return s


def _repair_pass(
    x_bits: np.ndarray,
    T: int,
    load_vec: np.ndarray,
    avg_renew: np.ndarray,
    mg: MicrogridParams,
    soc0: float,
    price_vec: Optional[np.ndarray],
    soc_ref: Optional[np.ndarray] = None,
    battery_mode: Optional[str] = None,
    target_soc: Optional[float] = None,
    min_soc_end: Optional[float] = None,
    preserve_battery: bool = False,
) -> np.ndarray:
    """Single forward greedy pass with combined objective:
        total = LAM_REPAIR_BALANCE * err² + score + LAM_REPAIR_HAMMING * hamming
    Hamming distance is from original (input) bits, anchoring repair to solver's choices.
    min_soc_end: hard SOC floor at final step - rejects combos that drain below LP plan.
    preserve_battery: lock c/d from solver bits, only search over grid (8 combos).
                      Used by SQA repair so SQA/QAOA produce different final solutions."""
    x  = x_bits.copy()
    x0 = x_bits.copy()  # original bits for Hamming anchor
    dis_f = (mg.P_B / mg.SOC_CAP) * DT / mg.eta_d
    chg_f = (mg.P_B / mg.SOC_CAP) * DT * mg.eta_c
    lam_v = LAM_RESERVE * mg.C_grid * mg.P_G
    lam_k = LAM_TRACK   * mg.C_grid * mg.P_G if soc_ref is not None else 0.0
    soc   = soc0

    for t in range(T):
        load_t  = float(load_vec[t])
        renew_t = float(avg_renew[t])
        pr_t    = float(price_vec[t]) if price_vec is not None else 1.0
        sref_t  = float(soc_ref[t]) if soc_ref is not None else None
        old_g0 = x0[idx_g0(t)]; old_g1 = x0[idx_g1(t)]; old_g2 = x0[idx_g2(t)]
        old_c  = x0[idx_c(t)];  old_d  = x0[idx_d(t)]
        is_last = (t == T - 1)
        best_total = np.inf; best_combo = None; best_soc = soc

        if preserve_battery:
            # Lock battery bits from solver; only search 8 grid combos
            cv = int(round(float(old_c))); dv = int(round(float(old_d)))
            if cv == 1 and dv == 1:
                cv, dv = 0, 0  # fix illegal state
            # Block discharge if current SOC already near floor
            if dv == 1 and soc <= mg.soc_min + 0.15:
                dv = 0  # force no discharge
            new_soc = soc + chg_f*cv - dis_f*dv
            new_soc = float(np.clip(new_soc, mg.soc_min, mg.soc_max))
            for gbits in range(8):
                g0=(gbits>>0)&1; g1=(gbits>>1)&1; g2=(gbits>>2)&1
                ns = soc + chg_f*cv - dis_f*dv
                if ns < mg.soc_min - SOC_TOL or ns > mg.soc_max + SOC_TOL: continue
                if is_last and min_soc_end is not None and ns < min_soc_end - SOC_TOL: continue
                grid_p = mg.P_G*(g0*GRID_COEFFS[0]+g1*GRID_COEFFS[1]+g2*GRID_COEFFS[2])
                err = abs(renew_t + grid_p + mg.P_B*(dv-cv) - load_t)
                net_wo_batt = renew_t + grid_p - load_t
                score = _step_score(g0, g1, g2, cv, dv, load_t, pr_t, mg, lam_v, dis_f, chg_f,
                                    soc_after=ns, soc_ref_t=sref_t, lam_k=lam_k,
                                    battery_mode=battery_mode, target_soc=target_soc,
                                    net_wo_batt=net_wo_batt)
                ham = ((g0 != old_g0) + (g1 != old_g1) + (g2 != old_g2))
                total = LAM_REPAIR_BALANCE * err * err + score + LAM_REPAIR_HAMMING * ham
                if total < best_total:
                    best_total = total; best_combo = (g0, g1, g2, cv, dv); best_soc = ns
        else:
            for bits in range(32):       # 2^5 = 32 combos (g0,g1,g2,c,d - no shed)
                g0=(bits>>0)&1; g1=(bits>>1)&1; g2=(bits>>2)&1
                cv=(bits>>3)&1; dv=(bits>>4)&1
                if cv==1 and dv==1: continue
                new_soc = soc + chg_f*cv - dis_f*dv
                if new_soc < mg.soc_min - SOC_TOL or new_soc > mg.soc_max + SOC_TOL: continue
                # Block discharge if current SOC already near floor (avoid deep drain without recovery)
                if dv==1 and soc <= mg.soc_min + 0.15:
                    continue
                if is_last and min_soc_end is not None and new_soc < min_soc_end - SOC_TOL:
                    continue  # hard SOC floor at window end
                grid_p = mg.P_G*(g0*GRID_COEFFS[0]+g1*GRID_COEFFS[1]+g2*GRID_COEFFS[2])
                err    = abs(renew_t + grid_p + mg.P_B*(dv-cv) - load_t)
                net_wo_batt = renew_t + grid_p - load_t
                score  = _step_score(g0, g1, g2, cv, dv, load_t, pr_t, mg, lam_v, dis_f, chg_f,
                                     soc_after=new_soc, soc_ref_t=sref_t, lam_k=lam_k,
                                     battery_mode=battery_mode, target_soc=target_soc,
                                     net_wo_batt=net_wo_batt)
                ham    = ((g0 != old_g0) + (g1 != old_g1) + (g2 != old_g2)
                          + (cv != old_c) + (dv != old_d))
                total  = LAM_REPAIR_BALANCE * err * err + score + LAM_REPAIR_HAMMING * ham
                if total < best_total:
                    best_total = total; best_combo = (g0, g1, g2, cv, dv); best_soc = new_soc

        if best_combo is not None:
            g0,g1,g2,cv,dv = best_combo
            x[idx_g0(t)]=float(g0); x[idx_g1(t)]=float(g1); x[idx_g2(t)]=float(g2)
            x[idx_c(t)]=float(cv);  x[idx_d(t)]=float(dv)
            soc = np.clip(best_soc, mg.soc_min, mg.soc_max)
    return x


def _repair_pairwise_pass(
    x_bits: np.ndarray,
    T: int,
    load_vec: np.ndarray,
    avg_renew: np.ndarray,
    mg: MicrogridParams,
    soc0: float,
    price_vec: Optional[np.ndarray],
    soc_ref: Optional[np.ndarray] = None,
    battery_mode: Optional[str] = None,
    target_soc: Optional[float] = None,
    min_soc_end: Optional[float] = None,
) -> np.ndarray:
    """Pairwise 32×32 repair with combined objective:
        total = LAM_REPAIR_BALANCE * (err0² + err1²) + score_pair + LAM_REPAIR_HAMMING * hamming
    Hamming over both timesteps (10 bits)."""
    x  = x_bits.copy()
    x0 = x_bits.copy()
    dis_f = (mg.P_B / mg.SOC_CAP) * DT / mg.eta_d
    chg_f = (mg.P_B / mg.SOC_CAP) * DT * mg.eta_c
    lam_v = LAM_RESERVE * mg.C_grid * mg.P_G
    lam_k = LAM_TRACK   * mg.C_grid * mg.P_G if soc_ref is not None else 0.0
    soc   = soc0
    t     = 0

    while t < T:
        if t + 1 < T:
            l0 = float(load_vec[t]);   r0 = float(avg_renew[t])
            l1 = float(load_vec[t+1]); r1 = float(avg_renew[t+1])
            p0 = float(price_vec[t])   if price_vec is not None else 1.0
            p1 = float(price_vec[t+1]) if price_vec is not None else 1.0
            sr0 = float(soc_ref[t])   if soc_ref is not None else None
            sr1 = float(soc_ref[t+1]) if soc_ref is not None else None
            old0_g0=x0[idx_g0(t)];   old0_g1=x0[idx_g1(t)];   old0_g2=x0[idx_g2(t)]
            old0_c =x0[idx_c(t)];    old0_d =x0[idx_d(t)]
            old1_g0=x0[idx_g0(t+1)]; old1_g1=x0[idx_g1(t+1)]; old1_g2=x0[idx_g2(t+1)]
            old1_c =x0[idx_c(t+1)];  old1_d =x0[idx_d(t+1)]
            best_total = np.inf; best_pair = None; best_soc2 = soc

            for b0 in range(32):
                g0_0=(b0>>0)&1; g1_0=(b0>>1)&1; g2_0=(b0>>2)&1
                cv0=(b0>>3)&1;  dv0=(b0>>4)&1
                if cv0==1 and dv0==1: continue
                s1 = soc + chg_f*cv0 - dis_f*dv0
                if s1 < mg.soc_min-SOC_TOL or s1 > mg.soc_max+SOC_TOL: continue
                grid0 = mg.P_G*(g0_0*GRID_COEFFS[0]+g1_0*GRID_COEFFS[1]+g2_0*GRID_COEFFS[2])
                e0 = abs(r0 + grid0 + mg.P_B*(dv0-cv0) - l0)
                net0 = r0 + grid0 - l0
                sc0 = _step_score(g0_0,g1_0,g2_0,cv0,dv0,l0,p0,mg,lam_v,dis_f,chg_f,
                                  soc_after=s1, soc_ref_t=sr0, lam_k=lam_k,
                                  battery_mode=battery_mode, target_soc=target_soc,
                                  net_wo_batt=net0)
                ham0 = ((g0_0!=old0_g0)+(g1_0!=old0_g1)+(g2_0!=old0_g2)
                        +(cv0!=old0_c)+(dv0!=old0_d))

                for b1 in range(32):
                    g0_1=(b1>>0)&1; g1_1=(b1>>1)&1; g2_1=(b1>>2)&1
                    cv1=(b1>>3)&1;  dv1=(b1>>4)&1
                    if cv1==1 and dv1==1: continue
                    s2 = s1 + chg_f*cv1 - dis_f*dv1
                    if s2 < mg.soc_min-SOC_TOL or s2 > mg.soc_max+SOC_TOL: continue
                    if (t + 1 == T - 1) and min_soc_end is not None and s2 < min_soc_end - SOC_TOL:
                        continue  # hard SOC floor at window end
                    grid1 = mg.P_G*(g0_1*GRID_COEFFS[0]+g1_1*GRID_COEFFS[1]+g2_1*GRID_COEFFS[2])
                    e1 = abs(r1 + grid1 + mg.P_B*(dv1-cv1) - l1)
                    net1 = r1 + grid1 - l1
                    sc1  = _step_score(g0_1,g1_1,g2_1,cv1,dv1,l1,p1,mg,lam_v,dis_f,chg_f,
                                       soc_after=s2, soc_ref_t=sr1, lam_k=lam_k,
                                       battery_mode=battery_mode, target_soc=target_soc,
                                       net_wo_batt=net1)
                    ham1 = ((g0_1!=old1_g0)+(g1_1!=old1_g1)+(g2_1!=old1_g2)
                            +(cv1!=old1_c)+(dv1!=old1_d))
                    total = (sc0 + sc1
                             + LAM_REPAIR_BALANCE * (e0*e0 + e1*e1)
                             + LAM_REPAIR_HAMMING * (ham0 + ham1))
                    if total < best_total:
                        best_total = total
                        best_pair=(g0_0,g1_0,g2_0,cv0,dv0, g0_1,g1_1,g2_1,cv1,dv1); best_soc2=s2

            if best_pair is not None:
                g0_0,g1_0,g2_0,cv0,dv0, g0_1,g1_1,g2_1,cv1,dv1 = best_pair
                x[idx_g0(t)]=float(g0_0); x[idx_g1(t)]=float(g1_0); x[idx_g2(t)]=float(g2_0)
                x[idx_c(t)]=float(cv0);   x[idx_d(t)]=float(dv0)
                x[idx_g0(t+1)]=float(g0_1); x[idx_g1(t+1)]=float(g1_1); x[idx_g2(t+1)]=float(g2_1)
                x[idx_c(t+1)]=float(cv1);   x[idx_d(t+1)]=float(dv1)
                soc = np.clip(best_soc2, mg.soc_min, mg.soc_max)
            t += 2
        else:
            # single trailing step
            load_t = float(load_vec[t]); renew_t = float(avg_renew[t])
            pr_t   = float(price_vec[t]) if price_vec is not None else 1.0
            sref_t = float(soc_ref[t]) if soc_ref is not None else None
            old_g0=x0[idx_g0(t)]; old_g1=x0[idx_g1(t)]; old_g2=x0[idx_g2(t)]
            old_c =x0[idx_c(t)];  old_d =x0[idx_d(t)]
            best_total=np.inf; best_combo=None; best_soc=soc
            for bits in range(32):
                g0=(bits>>0)&1; g1=(bits>>1)&1; g2=(bits>>2)&1
                cv=(bits>>3)&1; dv=(bits>>4)&1
                if cv==1 and dv==1: continue
                ns = soc+chg_f*cv-dis_f*dv
                if ns<mg.soc_min-SOC_TOL or ns>mg.soc_max+SOC_TOL: continue
                if min_soc_end is not None and ns < min_soc_end - SOC_TOL: continue
                grid_p = mg.P_G*(g0*GRID_COEFFS[0]+g1*GRID_COEFFS[1]+g2*GRID_COEFFS[2])
                err = abs(renew_t+grid_p+mg.P_B*(dv-cv)-load_t)
                net_wo_batt = renew_t + grid_p - load_t
                sc  = _step_score(g0,g1,g2,cv,dv,load_t,pr_t,mg,lam_v,dis_f,chg_f,
                                  soc_after=ns, soc_ref_t=sref_t, lam_k=lam_k,
                                  battery_mode=battery_mode, target_soc=target_soc,
                                  net_wo_batt=net_wo_batt)
                ham = ((g0!=old_g0)+(g1!=old_g1)+(g2!=old_g2)+(cv!=old_c)+(dv!=old_d))
                total = sc + LAM_REPAIR_BALANCE * err * err + LAM_REPAIR_HAMMING * ham
                if total < best_total:
                    best_total=total; best_combo=(g0,g1,g2,cv,dv); best_soc=ns
            if best_combo is not None:
                g0,g1,g2,cv,dv=best_combo
                x[idx_g0(t)]=float(g0); x[idx_g1(t)]=float(g1); x[idx_g2(t)]=float(g2)
                x[idx_c(t)]=float(cv);  x[idx_d(t)]=float(dv)
                soc = np.clip(best_soc, mg.soc_min, mg.soc_max)
            t += 1
    return x


def _repair(
    x_bits: np.ndarray,
    T: int,
    load_vec: np.ndarray,
    scenarios: List[Tuple[float, np.ndarray, np.ndarray]],
    mg: MicrogridParams,
    soc0: float,
    price_vec: Optional[np.ndarray] = None,
    n_passes: int = 3,
    soc_ref: Optional[np.ndarray] = None,
    battery_mode: Optional[str] = None,
    target_soc: Optional[float] = None,
    min_soc_end: Optional[float] = None,
    preserve_battery: bool = False,
) -> np.ndarray:
    """Multi-pass forward repair until convergence (≤ n_passes iterations).
    preserve_battery=True: lock SQA battery bits, only search grid combos."""
    ar = _avg_renew(scenarios, T)
    x  = x_bits.copy()
    for _ in range(n_passes):
        prev = x.copy()
        x = _repair_pass(x, T, load_vec, ar, mg, soc0, price_vec, soc_ref=soc_ref,
                         battery_mode=battery_mode, target_soc=target_soc,
                         min_soc_end=min_soc_end, preserve_battery=preserve_battery)
        if np.array_equal(x, prev):
            break
    return x


def _repair_pass_soft_preserve(
    x_bits: np.ndarray,
    T: int,
    load_vec: np.ndarray,
    avg_renew: np.ndarray,
    mg: MicrogridParams,
    soc0: float,
    price_vec: Optional[np.ndarray],
    soc_ref: Optional[np.ndarray] = None,
    battery_mode: Optional[str] = None,
    target_soc: Optional[float] = None,
    min_soc_end: Optional[float] = None,
) -> np.ndarray:
    """Soft-preserve repair for SQA: 16 combos per step (8 grid × {keep_batt, idle}).
    Prefers keeping SQA battery bits; allows flip to idle only when needed.
    Hamming anchored to SQA raw bits - stronger battery identity than full repair."""
    x  = x_bits.copy()
    x0 = x_bits.copy()
    dis_f = (mg.P_B / mg.SOC_CAP) * DT / mg.eta_d
    chg_f = (mg.P_B / mg.SOC_CAP) * DT * mg.eta_c
    lam_v = LAM_RESERVE * mg.C_grid * mg.P_G
    lam_k = LAM_TRACK   * mg.C_grid * mg.P_G if soc_ref is not None else 0.0
    soc   = soc0

    for t in range(T):
        load_t  = float(load_vec[t])
        renew_t = float(avg_renew[t])
        pr_t    = float(price_vec[t]) if price_vec is not None else 1.0
        sref_t  = float(soc_ref[t]) if soc_ref is not None else None
        old_g0 = x0[idx_g0(t)]; old_g1 = x0[idx_g1(t)]; old_g2 = x0[idx_g2(t)]
        old_c  = x0[idx_c(t)];  old_d  = x0[idx_d(t)]
        is_last = (t == T - 1)
        best_total = np.inf; best_combo = None; best_soc = soc

        raw_c = int(round(float(old_c))); raw_d = int(round(float(old_d)))
        if raw_c == 1 and raw_d == 1:
            raw_c, raw_d = 0, 0

        # Candidate battery states: keep raw, or flip to idle (c=0,d=0)
        batt_candidates = [(raw_c, raw_d)]
        if (raw_c, raw_d) != (0, 0):
            batt_candidates.append((0, 0))

        for gbits in range(8):
            g0=(gbits>>0)&1; g1=(gbits>>1)&1; g2=(gbits>>2)&1
            for cv, dv in batt_candidates:
                ns = soc + chg_f*cv - dis_f*dv
                if ns < mg.soc_min - SOC_TOL or ns > mg.soc_max + SOC_TOL:
                    continue
                if is_last and min_soc_end is not None and ns < min_soc_end - SOC_TOL:
                    continue
                grid_p = mg.P_G*(g0*GRID_COEFFS[0]+g1*GRID_COEFFS[1]+g2*GRID_COEFFS[2])
                err = abs(renew_t + grid_p + mg.P_B*(dv-cv) - load_t)
                net_wo_batt = renew_t + grid_p - load_t
                score = _step_score(g0, g1, g2, cv, dv, load_t, pr_t, mg, lam_v, dis_f, chg_f,
                                    soc_after=ns, soc_ref_t=sref_t, lam_k=lam_k,
                                    battery_mode=battery_mode, target_soc=target_soc,
                                    net_wo_batt=net_wo_batt)
                # Hamming anchored to SQA raw bits - battery flip penalised more than grid flip
                ham = ((g0 != old_g0) + (g1 != old_g1) + (g2 != old_g2)
                       + 2*(cv != old_c) + 2*(dv != old_d))
                total = LAM_REPAIR_BALANCE * err * err + score + LAM_REPAIR_HAMMING * ham
                if total < best_total:
                    best_total = total; best_combo = (g0, g1, g2, cv, dv); best_soc = ns

        if best_combo is not None:
            g0,g1,g2,cv,dv = best_combo
            x[idx_g0(t)]=float(g0); x[idx_g1(t)]=float(g1); x[idx_g2(t)]=float(g2)
            x[idx_c(t)]=float(cv);  x[idx_d(t)]=float(dv)
            soc = float(np.clip(best_soc, mg.soc_min, mg.soc_max))
    return x


def _repair_sqa_staged(
    x_bits: np.ndarray,
    T: int,
    load_vec: np.ndarray,
    scenarios: List[Tuple[float, np.ndarray, np.ndarray]],
    mg: MicrogridParams,
    soc0: float,
    price_vec: Optional[np.ndarray] = None,
    soc_ref: Optional[np.ndarray] = None,
    battery_mode: Optional[str] = None,
    target_soc: Optional[float] = None,
    min_soc_end: Optional[float] = None,
) -> Tuple[np.ndarray, str]:
    """3-stage SQA repair, returns (bits, stage_label).
    Stage 1 - preserve_battery: 8 grid combos, battery locked.
    Stage 2 - soft_preserve:    16 combos (8 grid × {keep_batt, idle}), battery-flip penalised.
    Stage 3 - full repair anchored to SQA raw bits (LAM_REPAIR_HAMMING applied to all 5 bits).
    Returns the label of the first stage that achieves feasibility."""
    from microgrid.feasibility import fast_violations
    ar = _avg_renew(scenarios, T)

    def _feasible(x: np.ndarray) -> bool:
        return fast_violations(x, T, load_vec, scenarios, mg, soc0) == 0

    def _soc_ok(x: np.ndarray) -> bool:
        if min_soc_end is None:
            return True
        from microgrid.utils import soc_trajectory
        _, c_arr, d_arr = decode(x, T)
        soc_end = float(np.clip(
            soc_trajectory(c_arr[:APPLY_STEPS], d_arr[:APPLY_STEPS], mg, soc0),
            mg.soc_min, mg.soc_max
        )[-1])
        return soc_end >= min_soc_end - SOC_TOL

    # Stage 1: preserve_battery
    x1 = x_bits.copy()
    for _ in range(3):
        prev = x1.copy()
        x1 = _repair_pass(x1, T, load_vec, ar, mg, soc0, price_vec, soc_ref=soc_ref,
                          battery_mode=battery_mode, target_soc=target_soc,
                          min_soc_end=min_soc_end, preserve_battery=True)
        if np.array_equal(x1, prev):
            break
    if _feasible(x1) and _soc_ok(x1):
        return x1, "sqa_preserve"

    # Stage 2: soft_preserve
    x2 = x_bits.copy()
    for _ in range(3):
        prev = x2.copy()
        x2 = _repair_pass_soft_preserve(x2, T, load_vec, ar, mg, soc0, price_vec,
                                        soc_ref=soc_ref, battery_mode=battery_mode,
                                        target_soc=target_soc, min_soc_end=min_soc_end)
        if np.array_equal(x2, prev):
            break
    if _feasible(x2) and _soc_ok(x2):
        return x2, "sqa_soft_preserve"

    # Stage 3: full repair anchored to SQA raw bits
    x3 = x_bits.copy()
    for _ in range(3):
        prev = x3.copy()
        x3 = _repair_pass(x3, T, load_vec, ar, mg, soc0, price_vec, soc_ref=soc_ref,
                          battery_mode=battery_mode, target_soc=target_soc,
                          min_soc_end=min_soc_end, preserve_battery=False)
        if np.array_equal(x3, prev):
            break
    if _feasible(x3) and _soc_ok(x3):
        return x3, "sqa_full_repair"

    # Return best of stage 3 even if not fully feasible (caller handles fallback)
    return x3, "sqa_full_repair"


def _repair_sqa(
    x_bits: np.ndarray,
    T: int,
    load_vec: np.ndarray,
    scenarios: List[Tuple[float, np.ndarray, np.ndarray]],
    mg: MicrogridParams,
    soc0: float,
    price_vec: Optional[np.ndarray] = None,
    soc_ref: Optional[np.ndarray] = None,
    battery_mode: Optional[str] = None,
    target_soc: Optional[float] = None,
    min_soc_end: Optional[float] = None,
) -> np.ndarray:
    """Legacy wrapper - returns bits only (no stage label). Used by scheduler fallback."""
    x, _ = _repair_sqa_staged(x_bits, T, load_vec, scenarios, mg, soc0, price_vec,
                               soc_ref=soc_ref, battery_mode=battery_mode,
                               target_soc=target_soc, min_soc_end=min_soc_end)
    return x


def _repair_pw(
    x_bits: np.ndarray,
    T: int,
    load_vec: np.ndarray,
    scenarios: List[Tuple[float, np.ndarray, np.ndarray]],
    mg: MicrogridParams,
    soc0: float,
    price_vec: Optional[np.ndarray] = None,
    soc_ref: Optional[np.ndarray] = None,
    battery_mode: Optional[str] = None,
    target_soc: Optional[float] = None,
    min_soc_end: Optional[float] = None,
) -> np.ndarray:
    """Pairwise repair followed by a single-pass polish."""
    ar = _avg_renew(scenarios, T)
    x  = _repair_pairwise_pass(x_bits, T, load_vec, ar, mg, soc0, price_vec, soc_ref=soc_ref,
                               battery_mode=battery_mode, target_soc=target_soc,
                               min_soc_end=min_soc_end)
    x  = _repair_pass(x, T, load_vec, ar, mg, soc0, price_vec, soc_ref=soc_ref,
                      battery_mode=battery_mode, target_soc=target_soc,
                      min_soc_end=min_soc_end)
    return x


# QAOA window solver

def _interp_warm_start(warm_params: np.ndarray, p_prev: int, p_new: int) -> np.ndarray:
    """INTERP strategy (Zhou et al. 2020): interpolate p_prev optimal params to p_new.
    For each layer l in 0..p_new-1, interpolate between adjacent p_prev params.
    Falls back to tiling when p_new <= p_prev."""
    if p_new <= p_prev or p_prev == 0:
        # tile existing params across new layers
        gammas_old = warm_params[:p_prev]
        betas_old  = warm_params[p_prev:]
        gammas_new = np.interp(np.linspace(0, p_prev - 1, p_new), np.arange(p_prev), gammas_old)
        betas_new  = np.interp(np.linspace(0, p_prev - 1, p_new), np.arange(p_prev), betas_old)
        return np.concatenate([gammas_new, betas_new])
    # Pad p_prev params with zeros at boundaries, then interpolate to p_new layers.
    # Formula: param_new[l] = ((p_new-l)*param_old[l-1] + l*param_old[l]) / p_new
    # where param_old is 0-indexed 0..p_prev-1 with sentinel 0 at -1 and p_prev.
    gammas_old = np.concatenate([[0.0], warm_params[:p_prev], [0.0]])  # size p_prev+2
    betas_old  = np.concatenate([[0.0], warm_params[p_prev:], [0.0]])
    # l runs 1..p_new; map to old indices via floor(l * p_prev / p_new)
    gammas_new = np.zeros(p_new)
    betas_new  = np.zeros(p_new)
    for l in range(1, p_new + 1):
        li = int(np.floor(l * p_prev / p_new))          # 0..p_prev
        gammas_new[l - 1] = ((p_new - l) * gammas_old[li] + l * gammas_old[min(li + 1, p_prev + 1)]) / p_new
        betas_new[l - 1]  = ((p_new - l) * betas_old[li]  + l * betas_old[min(li + 1, p_prev + 1)])  / p_new
    return np.concatenate([gammas_new, betas_new])


def run_qaoa_window(
    Q: Dict,
    n: int,
    offset: float,
    p: int,
    T: int,
    load_vec: np.ndarray,
    scenarios: List[Tuple[float, np.ndarray, np.ndarray]],
    mg: MicrogridParams,
    soc0: float,
    price_vec: np.ndarray,
    warm_params: Optional[np.ndarray] = None,
    shots: int = QAOA_SHOTS,
    max_iter: int = QAOA_MAX_ITER,
    n_restarts: int = QAOA_RESTARTS,
    baseline_cost: Optional[float] = None,
    soc_ref: Optional[np.ndarray] = None,
    min_soc_end: Optional[float] = None,
    ibm_backend: Optional[str] = None,
    ibm_shots: int = 4096,
) -> Tuple[Optional[np.ndarray], float, Optional[np.ndarray], List[float], Dict]:
    """Run QAOA for one scheduling window.

    Parameter optimization always uses a local AerSimulator (fast, free).
    When ``ibm_backend`` is provided the final circuit is shot on IBM hardware.
    ``baseline_cost`` is the MILP window cost used only for reporting.
    """
    import logging
    log = logging.getLogger(__name__)
    from scipy.optimize import minimize
    try:
        from qiskit import transpile
    except ImportError:
        from qiskit.transpile import transpile

    sim        = _get_simulator()
    h, J, C    = qubo_to_ising(Q, n, offset)
    diag       = _ising_diag(h, J, n, C)
    qc, gammas, betas = _build_circuit(h, J, n, p)
    qc_t       = transpile(qc, sim)

    use_ibm = (ibm_backend is not None)

    def objective(params):
        bind = {gammas[l]: params[l] for l in range(p)}
        bind.update({betas[l]: params[p + l] for l in range(p)})
        # Always use local simulator for optimization loop (fast)
        sv   = sim.run(qc_t.assign_parameters(bind)).result().get_statevector().data
        prob = np.abs(sv) ** 2
        prob = prob / prob.sum()
        return float(np.dot(prob, diag))

    # Build ordered list of initial parameter vectors for restarts.
    # Priority: (1) INTERP-extended warm_params, (2) random.
    init_list: List[np.ndarray] = []
    if warm_params is not None:
        p_prev = len(warm_params) // 2
        if p_prev == p:
            init_list.append(warm_params.copy())
        else:
            init_list.append(_interp_warm_start(warm_params, p_prev, p))
    best_val, best_params = np.inf, None
    trace: List[float] = []

    for r in range(n_restarts):
        if r < len(init_list):
            x0 = init_list[r]
        else:
            x0 = np.random.uniform(-np.pi, np.pi, 2 * p)
        _cur: List[float] = [np.inf]

        def obj_tracked(params):
            v = objective(params); _cur[0] = v; return v

        res = minimize(obj_tracked, x0, method="COBYLA",
                       options={"maxiter": max_iter, "rhobeg": 0.3},
                       callback=lambda _: trace.append(_cur[0]))
        if float(res.fun) < best_val:
            best_val, best_params = float(res.fun), res.x

    log.debug(f"p={p}, cobyla_best_val={best_val:.6f}")

    if best_params is None:
        return None, np.inf, warm_params, trace, {}

    # Final probability distribution
    bind = {gammas[l]: best_params[l] for l in range(p)}
    bind.update({betas[l]: best_params[p + l] for l in range(p)})

    if use_ibm:
        # Execute on IBM hardware with optimized parameters.
        # NO simulator fallback: retry transient (network/queue) failures with
        # backoff, re-resolving the backend each attempt; raise if all attempts
        # fail so a window is never silently filled with simulator data.
        import logging as _log, time as _time
        qc_ibm, gammas_ibm, betas_ibm = _build_circuit(h, J, n, p, save_sv=False)
        qc_bound = qc_ibm.assign_parameters(
            {gammas_ibm[l]: best_params[l] for l in range(p)} |
            {betas_ibm[l]: best_params[p + l] for l in range(p)}
        )
        _ibm_attempts = 6
        _last_err = None
        prob = None
        for _attempt in range(1, _ibm_attempts + 1):
            try:
                backend = _get_ibm_backend(ibm_backend)
                prob = _run_ibm_shots(qc_bound, backend, shots=ibm_shots)
                prob = prob / (prob.sum() + 1e-300)
                execution_backend = ibm_backend
                break
            except Exception as e:
                _last_err = e
                _wait = 30 * _attempt
                _log.getLogger(__name__).warning(
                    f"IBM '{ibm_backend}' attempt {_attempt}/{_ibm_attempts} failed "
                    f"({e}); retrying in {_wait}s (no simulator fallback)"
                )
                _time.sleep(_wait)
        if prob is None:
            raise RuntimeError(
                f"IBM backend '{ibm_backend}' failed after {_ibm_attempts} attempts; "
                f"refusing simulator fallback (last error: {_last_err})"
            )
    else:
        # Enumerate ALL 2^n states from final statevector (no sampling noise)
        sv   = sim.run(qc_t.assign_parameters(bind)).result().get_statevector().data
        prob = np.abs(sv) ** 2
        prob = prob / prob.sum()
        execution_backend = "simulator"

    prob_max     = float(np.max(prob))
    prob_min     = float(np.min(prob))
    prob_entropy = float(-np.sum(prob * np.log(prob + 1e-300)))
    log.debug(f"prob max={prob_max:.6f}, min={prob_min:.6f}")
    log.debug(f"prob entropy={prob_entropy:.4f}")

    # Rank all states by probability descending. The scan below stops early once
    # FEASIBLE_TARGET strict+repaired feasible candidates are found, since the long tail
    # past the top of the ranking rarely changes which feasible candidate wins (entropy
    # is typically near-max - distribution nearly flat at low p).
    FEASIBLE_TARGET = 500
    indices = np.argsort(prob)[::-1]
    x_raw_qaoa = np.array([(indices[0] >> i) & 1 for i in range(n)], dtype=np.float64)

    # Candidate tracking
    # Priority 1: strictly expected-feasible candidates (balance + SOC + mutex).
    # Selection metric is price-aware op_cost rather than QUBO energy to avoid bias
    # toward high-grid / no-discharge schedules at low-price hours.
    best_strict_x        = None
    best_strict_cost     = np.inf    # op_cost (selection)
    best_strict_op       = np.inf    # op_cost (reporting, same)
    best_strict_max_res  = float("nan")
    best_strict_n_scen   = 0

    # Priority 2: repaired expected-feasible
    best_repaired_x        = None
    best_repaired_cost     = np.inf  # op_cost (selection)
    best_repaired_op       = np.inf  # op_cost (reporting, same)
    best_repaired_max_res  = float("nan")
    best_repaired_n_scen   = 0

    # Priority 3: best low-violation repaired (even if not fully feasible)
    best_low_viol_x   = None
    best_low_viol_cnt = np.inf

    # Commit fallback (commit-step feasible only) - selected by op_cost
    fallback_x      = None
    fallback_cost   = np.inf

    n_strict   = 0
    n_repaired = 0
    n_commit   = 0
    n_repair_rejected = 0
    viol_list: List[int] = []

    raw_costs:      List[float] = []
    repaired_costs: List[float] = []
    shed_counts:    List[int]   = []
    grid_sums:      List[float] = []

    # Helper: predict soc_end for candidate bits
    _chg_f = (mg.P_B / mg.SOC_CAP) * DT * mg.eta_c
    _dis_f = (mg.P_B / mg.SOC_CAP) * DT / mg.eta_d
    def _predict_soc_end(x_b):
        soc = float(soc0)
        for tt in range(APPLY_STEPS):
            soc = soc + _chg_f * float(x_b[idx_c(tt)]) - _dis_f * float(x_b[idx_d(tt)])
        return float(np.clip(soc, mg.soc_min, mg.soc_max))

    # Track the best candidate that also satisfies the terminal SOC floor (kept separate
    # from the unconstrained best so we can fall back if no compliant candidate exists).
    best_strict_x_socok = None
    best_strict_cost_socok = np.inf
    best_strict_op_socok = np.inf
    best_strict_max_res_socok = float("nan")
    best_strict_n_scen_socok = 0
    best_repaired_x_socok = None
    best_repaired_cost_socok = np.inf
    best_repaired_op_socok = np.inf
    best_repaired_max_res_socok = float("nan")
    best_repaired_n_scen_socok = 0

    n_scanned = 0
    for idx in indices:
        n_scanned += 1
        x_bits = np.array([(idx >> i) & 1 for i in range(n)], dtype=np.float64)

        # Deployment feasibility: expected balance + SOC + mutex (no scenario required)
        rep_c = check_feasibility(x_bits, T, load_vec, scenarios, mg, soc0=soc0, eval_t=APPLY_STEPS)
        rep_w = check_feasibility(x_bits, T, load_vec, scenarios, mg, soc0=soc0)

        if rep_c["expected_feasible"] and rep_w["expected_feasible"]:
            n_strict += 1
            raw_cost = op_cost(x_bits, APPLY_STEPS, load_vec, mg, price_vec=price_vec, scenarios=scenarios)
            raw_costs.append(raw_cost)
            if raw_cost < best_strict_cost:
                best_strict_cost    = raw_cost
                best_strict_op      = raw_cost
                best_strict_x       = x_bits.copy()
                best_strict_max_res = rep_c["max_abs_balance_residual"]
                best_strict_n_scen  = len(rep_c["scenario_bal_violations"])
            # Track strictly-feasible candidate that also meets the terminal SOC floor.
            if min_soc_end is not None:
                _se = _predict_soc_end(x_bits)
                if _se >= float(min_soc_end) - 1e-9 and raw_cost < best_strict_cost_socok:
                    best_strict_cost_socok    = raw_cost
                    best_strict_op_socok      = raw_cost
                    best_strict_x_socok       = x_bits.copy()
                    best_strict_max_res_socok = rep_c["max_abs_balance_residual"]
                    best_strict_n_scen_socok  = len(rep_c["scenario_bal_violations"])
        else:
            viol = fast_violations(x_bits, T, load_vec, scenarios, mg, soc0)
            viol_list.append(viol)
            if viol <= 10:
                # Standard multi-pass repair
                x_rep  = _repair(x_bits, T, load_vec, scenarios, mg, soc0, price_vec, soc_ref=soc_ref, min_soc_end=min_soc_end)
                rep_cr = check_feasibility(x_rep, T, load_vec, scenarios, mg, soc0=soc0, eval_t=APPLY_STEPS)
                rep_wr = check_feasibility(x_rep, T, load_vec, scenarios, mg, soc0=soc0)

                if rep_cr["expected_feasible"] and rep_wr["expected_feasible"]:
                    n_repaired += 1
                    repair_cost = op_cost(x_rep, APPLY_STEPS, load_vec, mg, price_vec=price_vec, scenarios=scenarios)
                    if repair_cost < best_repaired_cost:
                        best_repaired_cost    = repair_cost
                        best_repaired_op      = repair_cost
                        best_repaired_x       = x_rep.copy()
                        best_repaired_max_res = rep_cr["max_abs_balance_residual"]
                        best_repaired_n_scen  = len(rep_cr["scenario_bal_violations"])
                        repaired_costs.append(repair_cost)
                    # Track repaired candidate that also meets the terminal SOC floor.
                    if min_soc_end is not None:
                        _se_r = _predict_soc_end(x_rep)
                        if _se_r >= float(min_soc_end) - 1e-9 and repair_cost < best_repaired_cost_socok:
                            best_repaired_cost_socok    = repair_cost
                            best_repaired_op_socok      = repair_cost
                            best_repaired_x_socok       = x_rep.copy()
                            best_repaired_max_res_socok = rep_cr["max_abs_balance_residual"]
                            best_repaired_n_scen_socok  = len(rep_cr["scenario_bal_violations"])
                        shed_counts.append(0)  # shedding removed
                        grid_sums.append(float(sum(
                            x_rep[idx_g0(tt)] * GRID_COEFFS[0]
                            + x_rep[idx_g1(tt)] * GRID_COEFFS[1]
                            + x_rep[idx_g2(tt)] * GRID_COEFFS[2]
                            for tt in range(T)
                        )))
                else:
                    # Try pairwise repair for stubborn failures
                    x_pw   = _repair_pw(x_bits, T, load_vec, scenarios, mg, soc0, price_vec, soc_ref=soc_ref, min_soc_end=min_soc_end)
                    rep_pw = check_feasibility(x_pw, T, load_vec, scenarios, mg, soc0=soc0, eval_t=APPLY_STEPS)
                    rep_pw2 = check_feasibility(x_pw, T, load_vec, scenarios, mg, soc0=soc0)
                    if rep_pw["expected_feasible"] and rep_pw2["expected_feasible"]:
                        n_repaired += 1
                        repair_cost = op_cost(x_pw, APPLY_STEPS, load_vec, mg, price_vec=price_vec, scenarios=scenarios)
                        if repair_cost < best_repaired_cost:
                            best_repaired_cost    = repair_cost
                            best_repaired_op      = repair_cost
                            best_repaired_x       = x_pw.copy()
                            best_repaired_max_res = rep_pw["max_abs_balance_residual"]
                            best_repaired_n_scen  = len(rep_pw["scenario_bal_violations"])
                            repaired_costs.append(repair_cost)
                        # Track pairwise-repaired candidate that also meets the terminal SOC floor.
                        if min_soc_end is not None:
                            _se_p = _predict_soc_end(x_pw)
                            if _se_p >= float(min_soc_end) - 1e-9 and repair_cost < best_repaired_cost_socok:
                                best_repaired_cost_socok    = repair_cost
                                best_repaired_op_socok      = repair_cost
                                best_repaired_x_socok       = x_pw.copy()
                                best_repaired_max_res_socok = rep_pw["max_abs_balance_residual"]
                                best_repaired_n_scen_socok  = len(rep_pw["scenario_bal_violations"])
                            shed_counts.append(0)  # shedding removed
                            grid_sums.append(float(sum(
                                x_pw[idx_g0(tt)] * GRID_COEFFS[0]
                                + x_pw[idx_g1(tt)] * GRID_COEFFS[1]
                                + x_pw[idx_g2(tt)] * GRID_COEFFS[2]
                                for tt in range(T)
                            )))
                    else:
                        # Track best low-violation candidate as last resort
                        viol_after = fast_violations(x_rep, T, load_vec, scenarios, mg, soc0)
                        viol_after_pw = fast_violations(x_pw, T, load_vec, scenarios, mg, soc0)
                        if viol_after <= viol_after_pw:
                            best_cand, best_vcnt = x_rep, viol_after
                        else:
                            best_cand, best_vcnt = x_pw, viol_after_pw
                        if best_vcnt < best_low_viol_cnt:
                            best_low_viol_cnt = best_vcnt
                            best_low_viol_x   = best_cand.copy()
            elif rep_c["expected_feasible"]:
                n_commit += 1
                fb_cost = float(op_cost(x_bits, APPLY_STEPS, load_vec, mg, price_vec=price_vec, scenarios=scenarios))
                if fb_cost < fallback_cost:
                    fallback_cost = fb_cost
                    fallback_x    = x_bits.copy()

        if (n_strict + n_repaired) >= FEASIBLE_TARGET:
            break

    # Priority selection: strict > repaired > low-viol > fallback
    # Selection priority prefers terminal-SOC-compliant variants when available.
    if best_strict_x_socok is not None:
        best_x        = best_strict_x_socok
        best_cost_out = best_strict_op_socok
        selected_type = "strict_socok"
        winner_max_res = best_strict_max_res_socok
        winner_n_scen  = best_strict_n_scen_socok
    elif best_strict_x is not None:
        best_x        = best_strict_x
        best_cost_out = best_strict_op           # op_cost for reporting
        selected_type = "strict"
        winner_max_res = best_strict_max_res
        winner_n_scen  = best_strict_n_scen
    elif best_repaired_x_socok is not None:
        best_x        = best_repaired_x_socok
        best_cost_out = best_repaired_op_socok
        selected_type = "repaired_socok"
        winner_max_res = best_repaired_max_res_socok
        winner_n_scen  = best_repaired_n_scen_socok
    elif best_repaired_x is not None:
        best_x        = best_repaired_x
        best_cost_out = best_repaired_op         # op_cost for reporting
        selected_type = "repaired"
        winner_max_res = best_repaired_max_res
        winner_n_scen  = best_repaired_n_scen
    elif best_low_viol_x is not None:
        best_x        = best_low_viol_x
        best_cost_out = op_cost(best_low_viol_x, APPLY_STEPS, load_vec, mg, price_vec=price_vec, scenarios=scenarios)
        selected_type = "low_viol"
        winner_max_res = float("nan")
        winner_n_scen  = 0
    elif fallback_x is not None:
        best_x        = fallback_x
        best_cost_out = op_cost(fallback_x, APPLY_STEPS, load_vec, mg, price_vec=price_vec, scenarios=scenarios)
        selected_type = "fallback"
        winner_max_res = float("nan")
        winner_n_scen  = 0
    else:
        best_x        = None
        best_cost_out = np.inf
        selected_type = "none"
        winner_max_res = float("nan")
        winner_n_scen  = 0

    repair_gap_pct = float("nan")
    if baseline_cost is not None and not np.isinf(baseline_cost) and abs(baseline_cost) > 1e-9:
        if selected_type in ("strict", "repaired"):
            repair_gap_pct = 100.0 * (best_cost_out - baseline_cost) / abs(baseline_cost)

    hamming = int(np.sum(best_x != x_raw_qaoa)) if best_x is not None else -1

    diag_info = {
        "n_unique_samples":    n_scanned,
        "n_strictly_feasible": n_strict,
        "n_repaired_feasible": n_repaired,
        "n_commit_feasible":   n_commit,
        "n_infeasible":        n_scanned - n_strict,
        "n_repair_rejected":   n_repair_rejected,
        "viol_min":            int(min(viol_list)) if viol_list else 0,
        "viol_mean":           float(np.mean(viol_list)) if viol_list else 0.0,
        "raw_cost":            float(np.min(raw_costs)) if raw_costs else float("nan"),
        "repaired_cost":       float(best_repaired_op) if not np.isinf(best_repaired_op) else float("nan"),
        "milp_cost":           float(baseline_cost) if baseline_cost is not None else float("nan"),
        "repair_gap_pct":      repair_gap_pct,
        "selected_type":       selected_type,
        "max_abs_residual":    winner_max_res,
        "n_scenario_bal":      winner_n_scen,
        "shed_count":          int(shed_counts[-1]) if shed_counts else 0,
        "grid_sum":            float(grid_sums[-1]) if grid_sums else float("nan"),
        "cobyla_best_val":     float(best_val),
        "execution_backend":   execution_backend,
        "prob_max":            prob_max,
        "prob_min":            prob_min,
        "prob_entropy":        prob_entropy,
        "hamming_raw_vs_final": hamming,
    }

    log.debug(f"selected_type={selected_type}, n_strict={n_strict}, n_repaired={n_repaired}, "
              f"hamming={hamming}, prob_entropy={prob_entropy:.4f}")

    if best_x is None:
        return None, np.inf, best_params, trace, diag_info

    final_cost = op_cost(best_x, APPLY_STEPS, load_vec, mg, price_vec=price_vec, scenarios=scenarios)
    return best_x, final_cost, best_params, trace, diag_info
