"""Final assembly: main lens, uncertainty budget, mapping uncertainty, plots, results/*.json.

Main estimate: work/cache/method_combined.json (robust combined: stickers with drawing geometry, cluster-robust
weights + verified edges). Uncertainty = statistical (cluster bootstrap of the main method) (+) systematic
(RMS deviation of the plausible alternative estimates listed in ALTS from the main one, evaluated on the
MAPPING, rotation-compensated, and on the parameters) (+) sensitivity terms from the synthetic/sensitivity study.
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np

from common import CACHE, H, RESULTS, W, K_from, fold_margin, lens_json, load_json, project, save_json, undistort_points
from evaltools import grid, mapping_displacement, region_defs

MAIN = f"{CACHE}/method_combined.json"
# plausible alternatives (not refuted by the data) that define the systematic part of the uncertainty
ALTS = [
    ("lines_joint_indep", f"{CACHE}/method_lines_joint_indep.json", None),  # edges only, independent implementation (shared vertical)
    ("lines_joint_lc", f"{CACHE}/method_lines_joint_lc.json", None),        # edges only, orchestrator implementation (separate cart frames)
    ("vanishing_indep", f"{CACHE}/method_vanishing_indep.json", None),  # plumb distortion + VP f/pp
    ("lines_joint_indep_mergedfloor", f"{CACHE}/method_lines_joint_indep_mergedfloor.json", None),  # one straight floor line
    ("alt_brown_k1k2_joint", f"{CACHE}/method_alt_brown_k1k2.json", None),      # model-choice study, joint data, non-robust
    ("alt_brown_k1k2k3_joint", f"{CACHE}/method_alt_brown_k1k2k3.json", None),
    ("alt_division_l1l2", f"{CACHE}/method_alt_division_l1l2.json", None),
    ("combined_k1k2k3", f"{CACHE}/method_combined_alternatives.json", "combined_k1k2k3"),
    ("combined_k1k2p1p2", f"{CACHE}/method_combined_alternatives.json", "combined_k1k2p1p2"),
    ("combined_k1k2_ppfixed", f"{CACHE}/method_combined_alternatives.json", "combined_k1k2_ppfixed"),
]
EXTRA = [a for a in sys.argv[1:]]  # extra method json files (e.g. independent implementations) to include


def KD(j):
    return np.array(j["camera_matrix"], float), np.array(j["dist_coeffs"], float)


def load_item(fn, method=None):
    if not os.path.exists(fn):
        return None
    j = load_json(fn)
    if isinstance(j, list):
        for it in j:
            if method is None or it.get("method") == method:
                return it
        return None
    return j


main = load_json(MAIN)
K0, d0 = KD(main)
print("main", main["method"], "f", K0[0, 0], "pp", K0[0, 2], K0[1, 2], "dist", np.round(d0, 4), "fold", fold_margin(K0, d0))
full = load_json(f"{CACHE}/combined_full.json")
names = full["summary"]["k1k2"]["names"]
boot = np.array(full["boot"])

uv, shp = grid(30)
regs = region_defs()


def lens_from_names(x):
    p = dict(zip(names, x))
    K = K_from(p.get("f", p.get("fx")), p.get("f", p.get("fy")), p["cx"], p["cy"])
    d = np.array([p.get("k1", 0), p.get("k2", 0), p.get("p1", 0), p.get("p2", 0), p.get("k3", 0)])
    return K, d


# ---- statistical part: bootstrap -> mapping displacement fields
Dstat = np.array([mapping_displacement(K0, d0, *lens_from_names(xb), uv) for xb in boot])
# robust per-point 1-sigma: 68th percentile of the displacement magnitude over replicates (degenerate replicates
# of the bootstrap - see REPORT - do not dominate)
sig_stat_map = np.nanpercentile(np.hypot(Dstat[..., 0], Dstat[..., 1]), 68.3, axis=0)
# ---- systematic part: alternatives
alts = []
for name, fn, meth in ALTS + [(os.path.basename(e)[:-5], e, None) for e in EXTRA]:
    it = load_item(fn, meth)
    if it is None:
        print("missing alternative", name)
        continue
    Ka, da = KD(it)
    fm = fold_margin(Ka, da)
    if fm <= 0:
        print("alternative", name, "is folded -> skipped")
        continue
    disp = mapping_displacement(K0, d0, Ka, da, uv)
    raw = mapping_displacement(K0, d0, Ka, da, uv, compensate=False)
    alts.append(dict(name=name, K=Ka, d=da, disp=disp, raw=raw))
    print(f"alt {name:28s} f={Ka[0,0]:7.1f} pp=({Ka[0,2]:6.1f},{Ka[1,2]:6.1f}) d={np.round(da,4)}  map diff median {np.median(np.hypot(*disp.T)):.2f} max {np.max(np.hypot(*disp.T)):.2f}")
Dsys = np.array([a["disp"] for a in alts]) if alts else np.zeros((0, len(uv), 2))
sig_sys_map = np.sqrt(np.nanmean(np.sum(Dsys ** 2, 2), 0)) if len(alts) else np.zeros(len(uv))
# ---- sensitivity part (from the synthetic/sensitivity study if available): px per region, added in quadrature
sens = {}
if os.path.exists(f"{CACHE}/synthetic_report.json"):
    try:
        sr = load_json(f"{CACHE}/synthetic_report.json")
        sens = sr.get("mapping_sensitivity_total_px", {}) or {}
    except Exception:  # noqa
        sens = {}
sig_tot_map = np.sqrt(sig_stat_map ** 2 + sig_sys_map ** 2)
reg_out = {}
for n, fn in regs.items():
    m = fn(uv)
    s_sens = float(sens.get(n, 0.0)) if isinstance(sens.get(n, 0.0), (int, float)) else 0.0
    reg_out[n] = dict(stat_median=float(np.nanmedian(sig_stat_map[m])), sys_median=float(np.nanmedian(sig_sys_map[m])),
                      sens=s_sens, total_median=float(np.sqrt(np.nanmedian(sig_tot_map[m]) ** 2 + s_sens ** 2)),
                      total_max=float(np.sqrt(np.nanmax(sig_tot_map[m]) ** 2 + s_sens ** 2)))
print("mapping 1-sigma per region:", {k: round(v["total_median"], 2) for k, v in reg_out.items()})

# ---- parameter uncertainties
pnames = ["f", "cx", "cy", "k1", "k2", "p1", "p2", "k3"]


def pvec(K, d):
    return np.array([K[0, 0], K[0, 2], K[1, 2], d[0], d[1], d[2], d[3], d[4]])


p0 = pvec(K0, d0)
pb = np.array([pvec(*lens_from_names(xb)) for xb in boot])
s_stat = 0.5 * (np.percentile(pb, 84, axis=0) - np.percentile(pb, 16, axis=0))
s_sys = np.sqrt(np.mean([(pvec(a["K"], a["d"]) - p0) ** 2 for a in alts], 0)) if alts else np.zeros(8)
s_tot = np.sqrt(s_stat ** 2 + s_sys ** 2)
for n, v, a, b, c in zip(pnames, p0, s_stat, s_sys, s_tot):
    print(f"  {n:3s} {v:10.5f}  stat {a:.5f}  sys {b:.5f}  total {c:.5f}")

unc = {"fx_px_1sigma": float(s_tot[0]), "fy_px_1sigma": float(s_tot[0]), "cx_px_1sigma": float(s_tot[1]),
       "cy_px_1sigma": float(s_tot[2]), "k1_1sigma": float(s_tot[3]), "k2_1sigma": float(s_tot[4]),
       "p1_1sigma": None, "p2_1sigma": None, "k3_1sigma": None,
       "note": "1-sigma = statistical (cluster bootstrap over stickers and edges) (+) systematic (RMS deviation of "
               "plausible alternative methods/models from the main estimate); p1=p2=k3=0 are fixed (not determined, "
               "see REPORT.md); parameters are strongly correlated - the mapping uncertainty is the relevant quantity",
       "components": {n: {"stat": float(a), "sys": float(b)} for n, a, b in zip(pnames, s_stat, s_sys)},
       "correlation_bootstrap": np.corrcoef(pb[:, :5].T).round(3).tolist()}
out = lens_json(K0, d0, model=main["model"], rms_reprojection_error_px=main["rms_reprojection_error_px"],
                uncertainty=unc,
                mapping_uncertainty_px={"centre": reg_out["centre"]["total_median"], "cart_band": reg_out["cart_band"]["total_median"],
                                        "corners": reg_out["corners"]["total_median"],
                                        "definition": "1-sigma displacement of the projection of a fixed viewing ray [px], "
                                                      "median over the region (centre: r<150 px; cart_band: both cart footprints; "
                                                      "corners: 200x150 px corner boxes), rotation-compensated (a pure camera "
                                                      "rotation is absorbed by the pose)",
                                        "max_in_region": {k: v["total_max"] for k, v in reg_out.items()},
                                        "components": reg_out},
                data_used=main.get("data_used"), geometry_assumptions=main.get("geometry_assumptions"))
out["method"] = main["method"]
save_json(out, f"{RESULTS}/lens_result.json")
save_json(dict(regions=reg_out, alternatives=[a["name"] for a in alts], params=dict(zip(pnames, p0.tolist())),
               sigma_total=dict(zip(pnames, s_tot.tolist()))), f"{CACHE}/final_budget.json")

# ---- plots: mapping uncertainty map
fig, axs = plt.subplots(1, 3, figsize=(20, 4.8), dpi=110)
img = cv2.cvtColor(cv2.imread(f"{CACHE}/mean_aligned_color.png"), cv2.COLOR_BGR2RGB)
for ax, s, t in zip(axs, [sig_stat_map, sig_sys_map, sig_tot_map], ["statistical (bootstrap)", "systematic (methods/models)", "total"]):
    ax.imshow(img, alpha=0.45)
    im = ax.imshow(s.reshape(shp), extent=(uv[:, 0].min() - 15, uv[:, 0].max() + 15, uv[:, 1].max() + 15, uv[:, 1].min() - 15),
                   cmap="magma", alpha=0.7, vmin=0, vmax=max(2.0, float(np.percentile(sig_tot_map, 98))))
    cs = ax.contour(uv[:, 0].reshape(shp), uv[:, 1].reshape(shp), s.reshape(shp), levels=[0.5, 1, 2, 4, 8], colors="w", linewidths=0.6)
    ax.clabel(cs, fmt="%.1f", fontsize=7)
    ax.set_title(f"mapping 1-sigma [px]: {t}")
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_axis_off()
fig.colorbar(im, ax=axs, shrink=0.8)
fig.savefig(f"{RESULTS}/mapping_uncertainty.png", bbox_inches="tight")
plt.close(fig)

# ---- undistorted mean image with the main model (visual straightness check)
newK = K0.copy()
m1, m2 = cv2.initUndistortRectifyMap(K0, d0, None, newK, (W, H), cv2.CV_32FC1)
und = cv2.remap(cv2.imread(f"{CACHE}/mean_aligned_color.png"), m1, m2, cv2.INTER_CUBIC)
cv2.imwrite(f"{RESULTS}/undistorted_mean.png", und)
print("done")
