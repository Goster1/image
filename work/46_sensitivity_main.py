"""Sensitivity of the FINAL main estimator (combined, non-robust, variance components iterated to convergence) to
small systematic detection errors on the REAL data. Perturbations as in 44_synth_c_sensitivity.py (same code),
plus the sticker sides: the traced sides are subject to the same edge bias towards black as the corners
(0.66 px, REPORT ch. 3), which was corrected only for the corners -> 'sides_bias' moves the 13 traced sides
outward by 0.66 px, 'all_bias_corrected' = corners bias-corrected + sides outward.

Usage: python3 46_sensitivity_main.py [NWORKERS=4] -> work/cache/sensitivity_main.json
"""
import importlib
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

from common import CACHE, save_json

os.environ.setdefault("SYNTH_SETUP", f"{CACHE}/synthetic_setup_main.json")  # truths T1 / T2 of the main-estimator study
C = importlib.import_module("44_synth_c_sensitivity")
L = C.L
NW = int(sys.argv[1]) if len(sys.argv) > 1 else 4
BIAS = 0.66


def estimate(ds):
    st = C.starts("real")
    cb, rc, _, K, d = L.combined_estimate(C.points_of(ds), ds["edges"] + ds["sides"], *st["cb"], robust=False)
    return {"main": dict(names=cb.inames, x=rc.x[: cb.ni], K=K, d=d, vc_iters=getattr(cb, "vc_iters", None))}


def sides_out(ds, val):
    d2 = dict(ds, sides=[dict(s) for s in ds["sides"]])
    for s in d2["sides"]:
        P = s["points"]
        n = L.curve_normals(P)
        sg = np.sign(((P - C.owner_centre(ds, s)) * n).sum(1))
        s["points"] = P + val * sg[:, None] * n
    return d2


PERTS = [("dilate", +0.3), ("dilate", -0.3), ("shift_x", 0.2), ("shift_y", 0.2), ("yscale_corners", 5e-4), ("yscale_all", 5e-4),
         ("aruco", None), ("bias_corrected", None), ("sides_bias", BIAS), ("all_bias_corrected", None),
         ("edge_dark", +0.3), ("edge_dark", -0.3), ("edge_outward", +0.3)] + [(f"frame{f}", None) for f in range(7)]


_BASE = {}


def run_one(task):
    name, val = task
    t0 = time.time()
    ds = C.dataset("real")
    if "base" not in _BASE:  # unperturbed fit once per worker
        _BASE["base"] = estimate(ds)
    base = _BASE["base"]
    if name == "sides_bias":
        pds = sides_out(ds, val)
    elif name == "all_bias_corrected":
        pds = sides_out(C.perturb(ds, "bias_corrected", None), BIAS)
    else:
        pds = C.perturb(ds, name, val)
    pert = estimate(pds)
    res = C.compare(base, pert, None)["main"]
    return f"{name}{'' if val is None else f'_{val:+g}'}", dict(res, base=base["main"]["x"].tolist(), vc_iters=pert["main"]["vc_iters"]), time.time() - t0


if __name__ == "__main__":
    t0 = time.time()
    out = {}
    with Pool(NW) as pool:
        for i, (key, res, dt) in enumerate(pool.imap_unordered(run_one, PERTS)):
            out[key] = res
            print(f"{i + 1}/{len(PERTS)} {key} {dt:.0f}s dparams {np.round(res['dparams'], 4)} map {({k: round(v, 2) for k, v in res['map'].items() if v is not None})}", flush=True)
    fr = [out[k] for k in out if k.startswith("frame")]
    out["_frames_summary"] = dict(df_range=[min(r["dparams"][0] for r in fr), max(r["dparams"][0] for r in fr)],
                                  map_max={k: max(r["map"][k] for r in fr) for k in ("centre", "cart_band", "corners")})
    save_json(dict(description=__doc__, results=out), f"{CACHE}/sensitivity_main.json")
    print("done", round(time.time() - t0), "s")
