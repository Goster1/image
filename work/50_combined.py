"""Combined estimator: stickers (drawing geometry, exact) + edges (exact cuboid lines, VP groups,
straightness) + visible sides of partly occluded stickers. Block weights by variance components.

Variants (model x data):
  combined_<model>           : all blocks
  combined_markers_robust    : stickers with a robust (Cauchy) loss instead of plain LS
Outputs: work/cache/method_combined.json (+ _alternatives), combined_full.json, plots results/combined_*.png
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from combined import Combined
from common import CACHE, RESULTS, H, W, K_from, lens_json, load_json, save_json
from evaltools import grid, mapping_displacement, mapping_stats
from linedata import load_edges
from markerdata import load_markers, marker_side_edges, to_points

NBOOT = int(sys.argv[1]) if len(sys.argv) > 1 else 100
rng = np.random.default_rng(777)
init = load_json(f"{CACHE}/initial_calib.json")
P0 = {80: np.array(init["poses"]["80"]), 310: np.array(init["poses"]["310"])}
WHICH = sys.argv[2] if len(sys.argv) > 2 else "final"
TAG = "" if WHICH == "final" else "_" + WHICH
M = load_markers(which=WHICH)
PTS = to_points(M)
SIDES = marker_side_edges(M, only_partial=True)
LINES = load_edges(verified_only=True)
# VP-group policy (same as 30_lines_methods.py): end-board edges straightness only; parallel floor lines.
FLOOR = {"scene_tapeV_left": "floor_V", "scene_tapeV_right": "floor_V", "scene_tapeH_top": "floor_H",
         "scene_tapeH_bottom": "floor_H", "scene_blueH_top": "floor_H", "scene_blueL_top": "floor_H", "scene_blueL_bottom": "floor_H"}
for e in LINES:
    if "board" in e["id"]:
        e["vp_group"] = None
        if e.get("model_line"):
            e["model_line"] = dict(e["model_line"], exact=False)  # board positions are not reliable (review)
    if e["id"] in FLOOR:
        e["vp_group"] = FLOOR[e["id"]]
EDGES = LINES + SIDES
ROBUST_C = 1.5  # px: sticker weight 1/(1+(rms/c)^2)
print(f"stickers {len(M)}, corners {len(PTS)}, partial-sticker sides {len(SIDES)}, edges total {len(EDGES)}")

SPECS = {
    "k1k2": dict(f="single", pp="free", dist=["k1", "k2"]),
    "k1k2k3": dict(f="single", pp="free", dist=["k1", "k2", "k3"]),
    "k1k2p1p2": dict(f="single", pp="free", dist=["k1", "k2", "p1", "p2"]),
    "k1k2_fxfy": dict(f="fxfy", pp="free", dist=["k1", "k2"]),
    "k1k2_ppfixed": dict(f="single", pp="fixed", dist=["k1", "k2"]),
    "k1_only": dict(f="single", pp="free", dist=["k1"]),
}


def run(spec, pts, edges, x0=None, sig=None, use=("M", "L", "V", "S"), loss="linear", robust=True):
    cb = Combined(pts, edges, spec, use=use, sig=sig)
    if x0 is None:
        K0 = K_from(1420.0, 1420.0, (W - 1) / 2, (H - 1) / 2)
        x0 = cb.x0(K0, np.array([-0.33, 0.09, 0, 0, 0]), P0)
    if robust and len(cb.mp):
        r = cb.solve_irls(x0, c=ROBUST_C, loss=loss)
    else:
        r = cb.solve(x0, loss=loss)
    return cb, r


def lens_of(cb, x):
    K, d, _, _ = cb.unpack(x)
    return K, d[:5]


summary = {}
fits = {}
xk = None
for name, spec in SPECS.items():
    cb, r = run(spec, PTS, EDGES)
    br = cb.block_rms(r.x)
    K, d = lens_of(cb, r.x)
    J = r.jac
    dof = max(len(r.fun) - len(r.x), 1)
    C = np.linalg.pinv(J.T @ J) * (2 * r.cost / dof)
    s = np.sqrt(np.diag(C))[: cb.ni]
    summary[name] = dict(names=cb.inames, x=r.x[: cb.ni].tolist(), sigma_cov=s.tolist(), block_rms=br, sig=cb.sig,
                         K=K.tolist(), dist=d.tolist())
    fits[name] = (cb, r)
    print(f"{name:14s} " + " ".join(f"{n}={v:.5g}±{e:.2g}" for n, v, e in zip(cb.inames, r.x[: cb.ni], s)),
          "| blocks", {k: (round(v[0], 3) if v[0] else None, v[1]) for k, v in br.items()})

# NON-robust variant (all stickers with full weight, block variance components only)
cb_r, r_r = run(SPECS["k1k2"], PTS, EDGES, robust=False)
Kr, dr = lens_of(cb_r, r_r.x)
summary["k1k2_nonrobust"] = dict(names=cb_r.inames, x=r_r.x[: cb_r.ni].tolist(), block_rms=cb_r.block_rms(r_r.x), K=Kr.tolist(), dist=dr.tolist(),
                                 sticker_rms=cb_r.sticker_rms(r_r.x))
print("non-robust k1k2", np.round(r_r.x[: cb_r.ni], 4), "sticker rms", {M[g]['role'] + f"@{M[g]['cart']}": round(v, 2) for g, v in cb_r.sticker_rms(r_r.x).items()})
# sticker weights / residuals of the main (robust) fit
cbm, rm = fits["k1k2"]
srm = cbm.sticker_rms(rm.x)
sticker_table = [dict(cart=M[g]["cart"], id=M[g]["id"], role=M[g]["role"], rms_px=v, weight=1.0 / (1.0 + (v / ROBUST_C) ** 2)) for g, v in sorted(srm.items())]
for t in sticker_table:
    print(f"   sticker {t['cart']}:{t['id']:3d} {t['role']:8s} rms {t['rms_px']:6.2f} px  weight {t['weight']:.3f}")
# data-subset variants (information content of each source)
for tag, use in {"edges_only": ("V", "S"), "markers_only_blockM": ("M",), "markers+sides": ("M", "L")}.items():
    try:
        cb2, r2 = run(SPECS["k1k2"], PTS, EDGES, use=use, x0=None)
        summary[f"k1k2_{tag}"] = dict(names=cb2.inames, x=r2.x[: cb2.ni].tolist(), block_rms=cb2.block_rms(r2.x))
        print(f"subset {tag:20s}", np.round(r2.x[: cb2.ni], 4), {k: v for k, v in cb2.block_rms(r2.x).items() if v[1]})
    except Exception as ex:  # noqa
        print("subset", tag, "failed", ex)

# ------------- profile of f for the main model -------------
cb, r = fits["k1k2"]
from scipy.optimize import least_squares

prof = []
for fv in np.linspace(r.x[0] * 0.85, r.x[0] * 1.15, 25):
    rr = least_squares(lambda y: cb.residuals(np.r_[fv, y]), r.x[1:], method="trf", x_scale="jac", max_nfev=200)
    b = cb.blocks(np.r_[fv, rr.x])
    prof.append([fv, 2 * rr.cost] + [float(np.sqrt(np.mean(b[k] ** 2))) if len(b[k]) else np.nan for k in ("M", "L", "V", "S")])
prof = np.array(prof)

# ------------- cluster bootstrap: resample stickers and edges -------------
mis = sorted({p["mi"] for p in PTS})
boot = []
for bi in range(NBOOT):
    pick_m = rng.choice(mis, len(mis), replace=True)
    bp = []
    for q, mi in enumerate(pick_m):
        bp += [dict(p, mi=1000 * q + mi) for p in PTS if p["mi"] == mi]
    if len({p["cart"] for p in bp}) < 2:
        continue
    pick_e = rng.integers(0, len(EDGES), len(EDGES))
    be = [dict(EDGES[i], id=f"{EDGES[i]['id']}#{q}") for q, i in enumerate(pick_e)]
    try:
        cbb, rb = run(SPECS["k1k2"], bp, be, x0=r.x.copy(), sig=dict(cb.sig))
        boot.append(rb.x[: cb.ni])
    except Exception as ex:  # noqa
        print("boot fail", ex)
    if bi % 10 == 0:
        print("boot", bi, np.round(boot[-1], 4) if boot else None)
boot = np.array(boot)
K, d = lens_of(cb, r.x)
uv, _ = grid(40)
disp = np.array([mapping_displacement(K, d, *lens_of(cb, np.r_[xb, r.x[cb.ni:]]), uv) for xb in boot])
mstats, mrms = mapping_stats(disp, uv)
disp_raw = np.array([mapping_displacement(K, d, *lens_of(cb, np.r_[xb, r.x[cb.ni:]]), uv, compensate=False) for xb in boot])
mstats_raw, _ = mapping_stats(disp_raw, uv)
print("bootstrap std", dict(zip(cb.inames, np.round(boot.std(0), 4))))
print("mapping (rot-comp) 1 sigma", {k: round(v["rms_median_px"], 3) for k, v in mstats.items()})

# ------------- plots -------------
fig, ax = plt.subplots(1, 2, figsize=(14, 5), dpi=110)
ax[0].plot(prof[:, 0], prof[:, 1] - prof[:, 1].min(), "k-o", ms=3)
ax[0].set_xlabel("f [px] (fixed, all else refit)")
ax[0].set_ylabel("delta chi^2 (block-weighted)")
ax[0].axhline(1, color="r", lw=0.8)
ax[0].grid(alpha=0.3)
for j, (k, c) in enumerate(zip("MLVS", "rbgm")):
    ax[1].plot(prof[:, 0], prof[:, 2 + j], "-o", ms=3, color=c, label={"M": "sticker corners", "L": "exact cuboid lines", "V": "VP groups", "S": "straightness"}[k])
ax[1].set_xlabel("f [px]")
ax[1].set_ylabel("block RMS [px]")
ax[1].legend()
ax[1].grid(alpha=0.3)
fig.suptitle("Combined estimator: profile of f")
fig.tight_layout()
fig.savefig(f"{RESULTS}/combined_f_profile{TAG}.png")
plt.close(fig)

unc = {f"{n}_1sigma": float(v) for n, v in zip(cb.inames, boot.std(0))}
unc.update({f"{n}_1sigma_cov": float(v) for n, v in zip(cb.inames, summary["k1k2"]["sigma_cov"])})
out = lens_json(K, d, method="combined_robust (stickers with drawing geometry, cluster-robust IRLS weights + edges: VPs incl. floor lines, straightness, sides of partly hidden stickers)",
                model="fx=fy, pp free, k1,k2 (p1=p2=k3=0)", rms_reprojection_error_px=summary["k1k2"]["block_rms"]["M"][0],
                uncertainty=unc, mapping_uncertainty_px={k: v["rms_median_px"] for k, v in mstats.items() if k != "whole_image"},
                data_used=f"{len(PTS)} valid sticker corners of {len(M)} stickers (7-frame means), {len(EDGES)} edges incl. {len(SIDES)} sides of partly hidden stickers",
                geometry_assumptions="drawing dimensions and sticker positions exactly as specified (not fitted)",
                details=dict(summary=summary, block_sigmas=cb.sig, bootstrap_n=len(boot), mapping=mstats, mapping_raw=mstats_raw,
                             sticker_table=sticker_table, robust_c_px=ROBUST_C))
save_json(out, f"{CACHE}/method_combined{TAG}.json")
alts = []
for name in SPECS:
    if name == "k1k2":
        continue
    s = summary[name]
    alts.append(lens_json(np.array(s["K"]), s["dist"], method=f"combined_{name}", model=name,
                          rms_reprojection_error_px=s["block_rms"]["M"][0],
                          uncertainty={f"{n}_1sigma_cov": v for n, v in zip(s["names"], s["sigma_cov"])},
                          details=dict(block_rms=s["block_rms"])))
save_json(alts, f"{CACHE}/method_combined_alternatives{TAG}.json")
save_json(dict(summary=summary, profile=prof.tolist(), boot=boot.tolist(), mapping=mstats, mapping_raw=mstats_raw, sticker_table=sticker_table),
          f"{CACHE}/combined_full{TAG}.json")
print("done")
