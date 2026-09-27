"""Final assembly: main lens, uncertainty budget, mapping uncertainty, plots, results/*.json.

Main estimate: work/cache/method_combined.json (combined: ALL stickers with the drawing geometry at full weight +
verified edges, block weights by variance components). Uncertainty = statistical (cluster bootstrap of the main method) (+) systematic
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
    ("lines_joint_indep", f"{CACHE}/method_lines_joint.json", None),  # edges only, independent implementation (shared vertical)
    ("lines_joint_lc", f"{CACHE}/method_lines_joint_lc.json", None),        # edges only, orchestrator implementation (separate cart frames)
    ("vanishing_indep", f"{CACHE}/method_vanishing.json", None),  # plumb distortion + VP f/pp
    ("lines_joint_indep_mergedfloor", f"{CACHE}/method_lines_joint_indep_mergedfloor.json", None),  # one straight floor line
    ("alt_brown_k1k2_joint", f"{CACHE}/method_alt_brown_k1k2.json", None),      # model-choice study, joint data, non-robust
    ("alt_brown_k1k2k3_joint", f"{CACHE}/method_alt_brown_k1k2k3.json", None),
    ("alt_division_l1l2", f"{CACHE}/method_alt_division_l1l2.json", None),
    ("combined_k1k2_robust", f"{CACHE}/method_combined_alternatives.json", "combined_k1k2_robust"),  # top stickers ~0 weight
    ("combined_k1k2k3", f"{CACHE}/method_combined_alternatives.json", "combined_k1k2k3"),
    # combined_k1k2p1p2 NOT in the budget: p1,p2 are not determined by the edges (~0 there); with the drawing geometry
    # they only absorb the sticker-geometry mismatch (f 1389, pp_y 449) - reported in the comparison table instead
    ("combined_k1k2_ppfixed", f"{CACHE}/method_combined_alternatives.json", "combined_k1k2_ppfixed"),
    # same data / model, different defensible modelling choices (50_combined.py VARIANTS)
    ("combined_k1k2_common_vertical", f"{CACHE}/method_combined_alternatives.json", "combined_k1k2_common_vertical"),  # posts of both carts || scene vertical
    ("combined_k1k2_rowblocks", f"{CACHE}/method_combined_alternatives.json", "combined_k1k2_rowblocks"),  # separate weights top / shelf stickers
    ("combined_k1k2_no_80_3_7_sides", f"{CACHE}/method_combined_alternatives.json", "combined_k1k2_no_80_3_7_sides"),  # two most influential sides out
    ("combined_k1k2_no_bent_lip", f"{CACHE}/method_combined_alternatives.json", "combined_k1k2_no_bent_lip"),  # possibly bent lip edges of cart 80 out
    ("combined_k1k2_floor_straight_only", f"{CACHE}/method_combined_alternatives.json", "combined_k1k2_floor_straight_only"),  # floor lines not parallel
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


SYN_SCEN = "T2_b"  # realistic scenario, truth close to the estimate (T1_b reported in REPORT.md)


def synth_rt():
    """Synthetic round trip of the FINAL main estimator (45_synth_main.py, scenario T2_b) in the layout
    {names, rmse, mapping: {total: {region: {median, max}}}}."""
    s = load_json(f"{CACHE}/synthetic_main.json")["summary"][SYN_SCEN]
    print("synthetic round trip: 45_synth_main", SYN_SCEN, "n =", s["n"])
    return dict(names=s["names"], rmse=s["rmse"], n=s["n"], mapping=dict(total={k: v for k, v in s["mapping_rmse"].items()}))


sig_tot_map = np.sqrt(sig_stat_map ** 2 + sig_sys_map ** 2)
# same budget WITHOUT rotation compensation (a pure camera rotation counted as error; relevant only when rays are
# compared directly without re-estimating the pose)
Dstat_raw = np.array([mapping_displacement(K0, d0, *lens_from_names(xb), uv, compensate=False) for xb in boot])
sig_stat_raw = np.nanpercentile(np.hypot(Dstat_raw[..., 0], Dstat_raw[..., 1]), 68.3, axis=0)
sig_sys_raw = np.sqrt(np.nanmean(np.sum(np.array([a["raw"] for a in alts]) ** 2, 2), 0)) if alts else np.zeros(len(uv))
sig_tot_raw = np.sqrt(sig_stat_raw ** 2 + sig_sys_raw ** 2)
reg_raw = {n: dict(stat_median=float(np.nanmedian(sig_stat_raw[fn(uv)])), sys_median=float(np.nanmedian(sig_sys_raw[fn(uv)])),
                   total_median=float(np.nanmedian(sig_tot_raw[fn(uv)]))) for n, fn in regs.items()}
print("mapping WITHOUT rotation compensation:", {k: round(v["total_median"], 1) for k, v in reg_raw.items()})
reg_out = {}
for n, fn in regs.items():
    m = fn(uv)
    s_sens = float(sens.get(n, 0.0)) if isinstance(sens.get(n, 0.0), (int, float)) else 0.0
    reg_out[n] = dict(stat_median=float(np.nanmedian(sig_stat_map[m])), sys_median=float(np.nanmedian(sig_sys_map[m])),
                      sens=s_sens, total_median=float(np.sqrt(np.nanmedian(sig_tot_map[m]) ** 2 + s_sens ** 2)),
                      total_max=float(np.sqrt(np.nanmax(sig_tot_map[m]) ** 2 + s_sens ** 2)))
if os.path.exists(f"{CACHE}/synthetic_main.json"):
    try:
        rt = synth_rt()
        for n in reg_out:
            sm = rt["mapping"]["total"].get(n, {}).get("median")
            reg_out[n]["synthetic_rmse"] = sm
            reg_out[n]["budget_median"] = reg_out[n]["total_median"]
            if sm is not None:
                reg_out[n]["total_median"] = float(max(reg_out[n]["total_median"], sm))
                reg_out[n]["total_max"] = float(max(reg_out[n]["total_max"], rt["mapping"]["total"][n]["max"]))
    except Exception as ex:  # noqa
        print("synthetic mapping not usable:", ex)
print("mapping 1-sigma per region:", {k: round(v["total_median"], 2) for k, v in reg_out.items()})

# ---- parameter uncertainties
pnames = ["f", "cx", "cy", "k1", "k2", "p1", "p2", "k3"]


def pvec(K, d):
    return np.array([K[0, 0], K[0, 2], K[1, 2], d[0], d[1], d[2], d[3], d[4]])


p0 = pvec(K0, d0)
pb = np.array([pvec(*lens_from_names(xb)) for xb in boot])
s_stat = 0.5 * (np.percentile(pb, 84, axis=0) - np.percentile(pb, 16, axis=0))
s_sys = np.sqrt(np.mean([(pvec(a["K"], a["d"]) - p0) ** 2 for a in alts], 0)) if alts else np.zeros(8)
s_bud = np.sqrt(s_stat ** 2 + s_sys ** 2)
# synthetic round trip with a realistic geometry-noise model (scenario T2_b of 44_synth_*): RMSE of the same
# combined estimator; the final 1-sigma is the LARGER of (budget, synthetic RMSE) - conservative
syn = {}
syn_map = {}
if os.path.exists(f"{CACHE}/synthetic_main.json"):
    try:
        rt = synth_rt()
        syn = dict(zip(rt["names"], rt["rmse"]))
        syn_map = {k: v["median"] for k, v in rt["mapping"]["total"].items()}
    except Exception as ex:  # noqa
        print("synthetic report not usable:", ex)
s_syn = np.array([syn.get(n, 0.0) for n in pnames])
s_tot = np.maximum(s_bud, s_syn)
for n, v, a, b, c, e in zip(pnames, p0, s_stat, s_sys, s_syn, s_tot):
    print(f"  {n:3s} {v:10.5f}  stat {a:.5f}  sys {b:.5f}  synthetic {c:.5f}  final {e:.5f}")

unc = {"fx_px_1sigma": float(s_tot[0]), "fy_px_1sigma": float(s_tot[0]), "cx_px_1sigma": float(s_tot[1]),
       "cy_px_1sigma": float(s_tot[2]), "k1_1sigma": float(s_tot[3]), "k2_1sigma": float(s_tot[4]),
       "p1_1sigma": None, "p2_1sigma": None, "k3_1sigma": None,
       "note": "1-sigma = max(stat (+) sys, synthetic round-trip RMSE); p1=p2=k3=0 fixed (not determined); fx=fy fitted as "
               "one parameter; k1 and k2 are strongly correlated (-0.9) - use uncertainty_details.covariance or the "
               "mapping uncertainty, not the individual sigmas combined as independent"}
corr = np.corrcoef(pb[:, :5].T)
cov = (s_tot[:5, None] * corr * s_tot[None, :5])
unc_details = {
    "rule": "final 1-sigma = max(budget = statistical (+) systematic, synthetic round-trip RMSE of the same estimator, scenario "
            f"{SYN_SCEN}: realistic detection noise + random sticker/row geometry deviations sized to the real sticker RMS + "
            "edge bows/direction deviations)",
    "statistical": "cluster bootstrap over stickers and edges (edges resampled within their vanishing-point group), "
                   f"{len(boot)} replicates, robust 1-sigma = half of the 16-84 % range",
    "systematic": "RMS deviation of the plausible alternative methods/models/modelling choices from the main estimate: "
                  + ", ".join(a["name"] for a in alts),
    "components": {n: {"stat": float(a), "sys": float(b), "budget": float(c), "synthetic_rmse": float(e)} for n, a, b, c, e in zip(pnames, s_stat, s_sys, s_bud, s_syn)},
    "order": ["f", "cx", "cy", "k1", "k2"],
    "correlation_bootstrap": corr.round(3).tolist(),
    "covariance": cov.tolist(),
    "covariance_note": "final 1-sigma values combined with the bootstrap correlation matrix (order f, cx, cy, k1, k2)",
}
out = lens_json(K0, d0, model=main["model"], rms_reprojection_error_px=main["rms_reprojection_error_px"],
                uncertainty=unc,
                mapping_uncertainty_px={"centre": reg_out["centre"]["total_median"], "cart_band": reg_out["cart_band"]["total_median"],
                                        "corners": reg_out["corners"]["total_median"]},
                data_used=main.get("data_used"), geometry_assumptions=main.get("geometry_assumptions"))
out["uncertainty_details"] = unc_details
out["mapping_uncertainty_details"] = {
    "definition": "1-sigma displacement of the projection of a fixed viewing ray [px], median over the region (centre: r<150 px "
                  "around the image centre; cart_band: both cart footprints; corners: 200x150 px corner boxes), "
                  "rotation-compensated (a pure camera rotation is absorbed by the pose); final value = max(budget median, "
                  f"synthetic {SYN_SCEN} RMSE median)",
    "max_in_region": {k: v["total_max"] for k, v in reg_out.items()},
    "components": reg_out,
    "without_rotation_compensation": {k: v["total_median"] for k, v in reg_raw.items()},
    "without_rotation_compensation_components": reg_raw,
    "without_rotation_compensation_note": "budget (bootstrap (+) alternatives) with a camera rotation counted as error; "
                                          "dominated by the principal-point / f alternatives; only relevant when rays are "
                                          "compared without re-estimating the camera pose",
}
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
