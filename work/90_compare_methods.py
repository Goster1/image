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


def method_id_of(fn, it, seen):
    """Short unique id: first token of the method name (file name for the main method); duplicates get '@<file>'."""
    base = os.path.basename(fn)[7:-5]
    if base == "combined":
        mid = "main_combined"
    else:
        mid = it.get("method", base).split(" ")[0]
        if mid in seen:
            mid = f"{mid}@{base}"
    seen.add(mid)
    return mid


def main():
    from resultnorm import normalise
    ms = collect()
    main_j = load_json(MAIN) if os.path.exists(MAIN) else None
    uv, _ = grid(40)
    regs = region_defs()
    rows = []
    clean = []
    seen = set()
    for m in ms:
        fn = os.path.join(os.path.dirname(RESULTS), m["_file"])
        mid = method_id_of(fn, m, seen)
        if mid == "main_combined" and main_j is not None:  # the main method = the final result (total uncertainty)
            m = dict(main_j, _file=m["_file"])
            m["method"] = "main_combined: " + main_j.get("method", "combined")
        c = normalise({k: v for k, v in m.items() if not k.startswith("_")}, mid)
        c["source_file"] = m["_file"]
        K, d = KD(c)
        if main_j is not None:
            K0, d0 = KD(main_j)
            try:
                disp = mapping_displacement(K0, d0, K, d, uv, compensate=True)
                r = np.hypot(disp[:, 0], disp[:, 1])
                c["mapping_difference_vs_main_px"] = {n: float(np.nanmedian(r[f(uv)])) for n, f in regs.items()}
            except Exception as ex:  # noqa
                c["mapping_difference_vs_main_px"] = None
        clean.append(c)
        rows.append(dict(method_id=mid, method=c["method"], fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], k1=d[0], k2=d[1], p1=d[2], p2=d[3],
                         k3=d[4], rms=c["rms_reprojection_error_px"], unc=c["uncertainty"], model=c.get("model"), file=m["_file"],
                         vs_main_px=c.get("mapping_difference_vs_main_px")))
    for r in rows:
        u = r["unc"]
        print(f"{r['method_id']:40s} f={r['fx']:7.1f}±{(u.get('fx_px_1sigma') or float('nan')):5.1f} cx={r['cx']:6.1f} cy={r['cy']:6.1f} "
              f"k1={r['k1']:+.4f} k2={r['k2']:+.4f} rms={r['rms']:.2f} vs_main={ {k: round(v, 1) for k, v in (r['vs_main_px'] or {}).items()} }")
    save_json(rows, f"{CACHE}/methods_table.json")
    save_json(clean, f"{RESULTS}/lens_by_method.json")

    # plot (main highlighted, total 1-sigma; plumb-line has no f of its own)
    fig, axs = plt.subplots(1, 4, figsize=(18, 0.36 * len(rows) + 2), dpi=110, sharey=True)
    y = np.arange(len(rows))
    for ax, key, sk in zip(axs, ["fx", "cx", "cy", "k1"], ["fx_px_1sigma", "cx_px_1sigma", "cy_px_1sigma", "k1_1sigma"]):
        for yi, r in zip(y, rows):
            e = r["unc"].get(sk) or 0.0
            ismain = r["method_id"] == "main_combined"
            ax.errorbar(r[key], yi, xerr=e, fmt="s" if ismain else "o", ms=6 if ismain else 4, capsize=3,
                        color="C3" if ismain else "C0", lw=2 if ismain else 1)
        mr = [r for r in rows if r["method_id"] == "main_combined"]
        if mr:
            ax.axvline(mr[0][key], color="C3", lw=0.8, ls="--")
            e = mr[0]["unc"].get(sk) or 0.0
            ax.axvspan(mr[0][key] - e, mr[0][key] + e, color="C3", alpha=0.08)
        ax.set_title(key + " (red: main result, band = total 1-sigma)")
        ax.grid(alpha=0.3)
    axs[0].set_yticks(y)
    axs[0].set_yticklabels([r["method_id"] for r in rows], fontsize=8)
    axs[0].invert_yaxis()
    axs[0].set_xlim(1150, 1700)
    fig.tight_layout()
    fig.savefig(f"{RESULTS}/methods_comparison.png")
    print("wrote", len(clean), "methods")


if __name__ == "__main__":
    main()
