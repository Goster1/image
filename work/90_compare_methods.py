"""Collect every work/cache/method_*.json, compare methods (table + plots), write results/lens_by_method.json.

Mapping comparison: for every method, the displacement of its mapping relative to the main result
(rotation-compensated) per region; this is the between-method part of the uncertainty.
"""
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import CACHE, RESULTS, load_json, save_json
from evaltools import grid, mapping_displacement, region_defs

MAIN = sys.argv[1] if len(sys.argv) > 1 else f"{RESULTS}/lens_result.json"


def KD(j):
    K = np.array(j["camera_matrix"], float)
    d = np.array(j["dist_coeffs"], float)
    return K, d


def collect():
    out = []
    for fn in sorted(glob.glob(f"{CACHE}/method_*.json")):
        j = load_json(fn)
        items = j if isinstance(j, list) else [j]
        for it in items:
            if "camera_matrix" not in it:
                continue
            it = dict(it)
            it.setdefault("method", os.path.basename(fn)[7:-5])
            it["_file"] = os.path.relpath(fn, os.path.dirname(RESULTS))
            out.append(it)
    return out


def main():
    ms = collect()
    main_j = load_json(MAIN) if os.path.exists(MAIN) else None
    uv, _ = grid(40)
    regs = region_defs()
    rows = []
    for m in ms:
        K, d = KD(m)
        row = dict(method=m["method"], fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], k1=d[0], k2=d[1], p1=d[2], p2=d[3], k3=d[4],
                   rms=m.get("rms_reprojection_error_px"), unc=m.get("uncertainty", {}), model=m.get("model"), file=m["_file"])
        if main_j is not None:
            K0, d0 = KD(main_j)
            try:
                disp = mapping_displacement(K0, d0, K, d, uv, compensate=True)
                r = np.hypot(disp[:, 0], disp[:, 1])
                row["vs_main_px"] = {n: float(np.median(r[f(uv)])) for n, f in regs.items()}
                row["vs_main_px_max"] = {n: float(np.max(r[f(uv)])) for n, f in regs.items()}
            except Exception as ex:  # noqa
                row["vs_main_px"] = None
        rows.append(row)
    for r in rows:
        u = r["unc"] or {}
        print(f"{r['method']:34s} f={r['fx']:7.1f}±{u.get('f_1sigma', u.get('fx_px_1sigma', float('nan'))):5.1f} "
              f"cx={r['cx']:6.1f} cy={r['cy']:6.1f} k1={r['k1']:+.4f} k2={r['k2']:+.4f} k3={r['k3']:+.4f} p=({r['p1']:+.4f},{r['p2']:+.4f}) "
              f"rms={r['rms']} vs_main={r.get('vs_main_px')}")
    save_json(rows, f"{CACHE}/methods_table.json")
    # lens_by_method.json: the method objects themselves (same shape as lens_result.json)
    clean = []
    for m in ms:
        c = {k: v for k, v in m.items() if not k.startswith("_")}
        clean.append(c)
    save_json(clean, f"{RESULTS}/lens_by_method.json")

    # plot
    fig, axs = plt.subplots(1, 4, figsize=(18, 0.45 * len(rows) + 2), dpi=110, sharey=True)
    y = np.arange(len(rows))
    for ax, key, sk in zip(axs, ["fx", "cx", "cy", "k1"], ["f_1sigma", "cx_1sigma", "cy_1sigma", "k1_1sigma"]):
        v = [r[key] for r in rows]
        e = []
        for r in rows:
            u = r["unc"] or {}
            e.append(u.get(sk, u.get({"fx": "fx_px_1sigma", "cx": "cx_px_1sigma", "cy": "cy_px_1sigma", "k1": "k1_1sigma"}[key], 0)) or 0)
        ax.errorbar(v, y, xerr=e, fmt="o", ms=4, capsize=3)
        ax.set_title(key)
        ax.grid(alpha=0.3)
    axs[0].set_yticks(y)
    axs[0].set_yticklabels([r["method"] for r in rows], fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{RESULTS}/methods_comparison.png")
    print("wrote", len(clean), "methods")


if __name__ == "__main__":
    main()
