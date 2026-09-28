"""Synthetic round trip of the FINAL main estimator (combined, non-robust, block weights by variance components
iterated to convergence) - supplement to 44_synth_b_roundtrip.py, which was written for an earlier version of the
estimator (robust IRLS / 4 variance-component iterations).

Data generation is identical to 44_synth_b_roundtrip.make_data (same truths T1/T2, same noise models; scenario b =
detection noise + random 3D sticker/row geometry deviations sized to the real sticker RMS + edge bows and
direction deviations; c = diagnosed end-board offsets; a = detection noise only).

Usage: python3 45_synth_main.py [NREP_B=30] [NWORKERS=4]
Writes work/cache/synthetic_main.json (per-scenario bias, scatter, RMSE, mapping RMSE per region).
"""
import importlib
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

from common import CACHE, K_from, save_json

os.environ.setdefault("SYNTH_SETUP", f"{CACHE}/synthetic_setup_main.json")  # truths T1 / T2 of the main-estimator study
B = importlib.import_module("44_synth_b_roundtrip")
L = B.L
NREP_B = int(sys.argv[1]) if len(sys.argv) > 1 else 30
NW = int(sys.argv[2]) if len(sys.argv) > 2 else 4
NREP = {"T1_b": NREP_B, "T2_b": NREP_B, "T2_a": 8, "T2_c": 8}
NAMES = ["f", "cx", "cy", "k1", "k2"]


def run_task(task):
    scen, rep = task
    t0 = time.time()
    T, kind, pts, edges, (K0, d0, P0), g = B.make_data(scen, rep)
    cb, rc, Cc, Kc, dc = L.combined_estimate(pts, edges, K0, d0, P0, robust=False)
    retry = None
    if cb.block_rms(rc.x)["S"][0] > 1.0:  # failed start -> default start of 50_combined.run
        retry = float(cb.block_rms(rc.x)["S"][0])
        cb, rc, Cc, Kc, dc = L.combined_estimate(pts, edges, K_from(1420.0, 1420.0, *L.CEN), np.array([-0.33, 0.09, 0, 0, 0]), P0, robust=False)
    truth = np.array([T.params[n] for n in NAMES])
    disp = L.disp_field(T.K, T.d, Kc, dc)
    return scen, rep, dict(x=rc.x[: cb.ni].tolist(), truth=truth.tolist(), disp=disp.astype(np.float32), sig=dict(cb.sig),
                           vc_iters=getattr(cb, "vc_iters", None), retry=retry, sigma_cov=np.sqrt(np.clip(np.diag(Cc), 0, None)).tolist()), time.time() - t0


if __name__ == "__main__":
    tasks = [(s, r) for s, n in NREP.items() for r in range(n)]
    tasks.sort(key=lambda t: t[1])
    out = {s: {} for s in NREP}
    t0 = time.time()
    with Pool(NW) as pool:
        for i, (scen, rep, res, dt) in enumerate(pool.imap_unordered(run_task, tasks)):
            out[scen][rep] = res
            print(f"{i + 1}/{len(tasks)} {scen} rep {rep} {dt:.0f}s f={res['x'][0]:.1f} (truth {res['truth'][0]:.0f}) "
                  f"cy={res['x'][2]:.1f} it={res['vc_iters']}", flush=True)
    summary = {}
    for scen, reps in out.items():
        if not reps:
            continue
        X = np.array([r["x"] for r in reps.values()])
        Tt = np.array([r["truth"] for r in reps.values()])
        err = X - Tt
        n = len(X)
        D = np.array([np.sqrt(np.sum(r["disp"] ** 2, axis=1)) for r in reps.values()])  # per replicate per point
        rms_pt = np.sqrt(np.nanmean(D ** 2, axis=0))
        reg = L.region_summary(rms_pt)
        summary[scen] = dict(
            n=n, names=NAMES, truth=Tt[0].tolist(),
            bias=err.mean(0).tolist(), bias_se=(err.std(0, ddof=1) / np.sqrt(n)).tolist(),
            std=err.std(0, ddof=1).tolist(), rmse=np.sqrt(np.mean(err ** 2, 0)).tolist(),
            mapping_rmse={k: v for k, v in reg.items()},
            vc_iters=[r["vc_iters"] for r in reps.values()], retries=sum(r["retry"] is not None for r in reps.values()),
            median_block_sigmas={k: float(np.median([r["sig"][k] for r in reps.values()])) for k in ("M", "L", "V", "S")},
            claimed_cov_sigma_median=np.median([r["sigma_cov"] for r in reps.values()], 0).tolist(),
        )
        s = summary[scen]
        print(scen, "n", n, "bias", np.round(s["bias"], 4), "+-", np.round(s["bias_se"], 4), "rmse", np.round(s["rmse"], 4),
              "map", {k: round(v["median"], 2) for k, v in reg.items() if v["median"] is not None})
    save_json(dict(description=__doc__, nrep=NREP, summary=summary,
                   raw={s: {str(r): dict(x=v["x"], truth=v["truth"], vc_iters=v["vc_iters"]) for r, v in reps.items()} for s, reps in out.items()}),
              f"{CACHE}/synthetic_main.json")
    print("done", round(time.time() - t0), "s")
