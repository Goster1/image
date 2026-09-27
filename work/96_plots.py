"""Report plots for the main lens (results/lens_result.json):

  results/residuals_stickers_main.png : detected vs predicted sticker quads (drawing geometry, poses refit per cart
                                         with the main intrinsics, least squares), residual arrows x10, per-sticker RMS
  results/residuals_edges_main.png     : straightness / VP residual of every edge point under the main lens
  results/predicted_stickers_crops.png : zoomed crops of every sticker: detected (green) vs predicted (red)
  results/f_profiles.png               : RMS / chi2 vs fixed f for stickers only, edges only and combined
"""
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from calib import Model, Problem
from common import CACHE, RESULTS, H, W, K_from, load_json, project, rodrigues, save_json
from linedata import load_edges
from lineselfcal import LineCal
from markerdata import load_markers, to_points

L = load_json(f"{RESULTS}/lens_result.json")
K = np.array(L["camera_matrix"])
d = np.array(L["dist_coeffs"])
init = load_json(f"{CACHE}/initial_calib.json")
P0 = {80: np.array(init["poses"]["80"]), 310: np.array(init["poses"]["310"])}
M = load_markers()
col = cv2.cvtColor(cv2.imread(f"{CACHE}/mean_aligned_color.png"), cv2.COLOR_BGR2RGB)

# ---------- poses per cart with the main intrinsics (least squares, same definition as rms_reprojection_error_px)
poses = {}
per_sticker = []
for c in (80, 310):
    pts = to_points(M, sel=lambda m: m["cart"] == c)
    X = np.array([p["X"] for p in pts])
    uv = np.array([p["uv"] for p in pts])

    def res(pz):
        return (project(X, K, d, rvec=pz[:3], tvec=pz[3:]) - uv).ravel()

    r = least_squares(res, P0[c])
    poses[c] = r.x
fig, ax = plt.subplots(figsize=(16, 9), dpi=110)
ax.imshow(col)
tiles = []
for mi, m in enumerate(M):
    pz = poses[m["cart"]]
    pred = project(m["corners_3d"], K, d, rvec=pz[:3], tvec=pz[3:])
    obs = m["corners_px"]
    v = m["valid"]
    e = pred - obs
    rms = float(np.sqrt(np.mean(np.sum(e[v] ** 2, 1)))) if v.any() else float("nan")
    per_sticker.append(dict(cart=m["cart"], id=m["id"], role=m["role"], rms_px=rms, mean_offset_px=e[v].mean(0).tolist() if v.any() else None))
    ax.plot(*np.vstack([pred, pred[:1]]).T, "-", color="red", lw=0.9)
    ax.plot(*np.vstack([obs, obs[:1]]).T, "-", color="lime", lw=0.9)
    for j in range(4):
        if v[j]:
            ax.annotate("", xy=obs[j] + 10 * e[j], xytext=obs[j], arrowprops=dict(arrowstyle="->", color="yellow", lw=0.8))
    ax.text(*obs.mean(0) + [0, -32], f"{m['cart']}:{m['id']} {rms:.1f}px", color="cyan", fontsize=6, ha="center")
    # crop tile
    cc = obs.mean(0)
    x0, y0 = int(np.clip(cc[0] - 45, 0, W - 90)), int(np.clip(cc[1] - 45, 0, H - 90))
    t = cv2.resize(col[y0:y0 + 90, x0:x0 + 90], None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC).copy()
    for q, colr in ((obs, (0, 255, 0)), (pred, (255, 0, 0))):
        qq = np.round(((q - [x0, y0]) + 0.5) * 4 - 0.5).astype(np.int32)
        cv2.polylines(t, [qq], True, colr, 2, cv2.LINE_AA)
    for j in range(4):
        if not v[j]:
            qq = np.round(((obs[j] - [x0, y0]) + 0.5) * 4 - 0.5).astype(int)
            cv2.circle(t, tuple(qq), 6, (255, 128, 0), 2)
    cv2.putText(t, f"{m['cart']}:{m['id']} {m['role']} {rms:.1f}px", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)
    tiles.append(t)
ax.set_title("Main lens, drawing geometry: detected sticker (green) vs predicted (red), residual x10 (yellow). "
             "Poses fitted per cart (robust).")
ax.set_axis_off()
fig.tight_layout()
fig.savefig(f"{RESULTS}/residuals_stickers_main.png")
plt.close(fig)
rows = [np.hstack(tiles[i:i + 5] + [np.zeros_like(tiles[0])] * (5 - len(tiles[i:i + 5]))) for i in range(0, len(tiles), 5)]
cv2.imwrite(f"{RESULTS}/predicted_stickers_crops.png", cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))
save_json(dict(poses={str(k): v for k, v in poses.items()}, per_sticker=per_sticker), f"{CACHE}/final_sticker_residuals.json")

# ---------- edge residuals under the main lens (straightness; VP groups with the same policy as the methods)
E = load_edges()
FLOOR = {"scene_tapeV_left": "floor_V", "scene_tapeV_right": "floor_V", "scene_tapeH_top": "floor_H", "scene_tapeH_bottom": "floor_H",
         "scene_blueH_top": "floor_H", "scene_blueL_top": "floor_H", "scene_blueL_bottom": "floor_H"}
for e in E:
    if "board" in e["id"]:
        e["vp_group"] = None
    if e["id"] in FLOOR:
        e["vp_group"] = FLOOR[e["id"]]
lc = LineCal(E, "plumb", dist_free=(), centre_free=False, f0=K[0, 0], centre_fixed=[K[0, 2], K[1, 2]], dist_fixed=d, subsample=1)
r_edges = lc.residuals(np.zeros(0))[:-1]
pe = lc.residuals(np.zeros(0), per_edge=True)
fig, ax = plt.subplots(figsize=(16, 9), dpi=110)
ax.imshow(col, alpha=0.6)
sc = ax.scatter(lc.allp[:, 0], lc.allp[:, 1], c=r_edges, cmap="coolwarm", vmin=-1, vmax=1, s=2)
for e, p in zip(E, pe):
    ax.text(*e["points"][len(e["points"]) // 2], f"{p:.2f}", fontsize=6, color="k")
fig.colorbar(sc, ax=ax, shrink=0.7, label="straightness residual [px] (distorted image)")
ax.set_title(f"Main lens: straightness residual of the {len(E)} verified edges (numbers = per-edge RMS px); overall RMS {np.sqrt(np.mean(r_edges**2)):.3f} px")
ax.set_axis_off()
fig.tight_layout()
fig.savefig(f"{RESULTS}/residuals_edges_main.png")
plt.close(fig)
print("edge rms", float(np.sqrt(np.mean(r_edges ** 2))), "sticker rms", {f"{s['cart']}:{s['id']}": round(s["rms_px"], 2) for s in per_sticker})

# ---------- f profiles
fig, ax = plt.subplots(1, 3, figsize=(18, 4.8), dpi=110)
mb = load_json(f"{CACHE}/markers_ba_full.json")
for name, pf in mb["profile"].items():
    pf = np.array(pf)
    ax[0].plot(pf[:, 0], pf[:, 1], "-o", ms=2, label=name)
ax[0].set_title("stickers only (drawing geometry): RMS vs f")
ax[0].set_xlabel("f [px]")
ax[0].set_ylabel("RMS [px]")
ax[0].legend(fontsize=7)
# edges-only joint fit (distortion + pp + rotations refit) at fixed f
lj = LineCal(E, "joint", dist_free=("k1", "k2"), pp_free=True, subsample=4)
xj = lj.x0(K, d, poses)
rj = least_squares(lj.residuals, xj, loss="huber", f_scale=0.5, x_scale="jac", max_nfev=300)
pf = []
for fv in np.linspace(rj.x[0] * 0.85, rj.x[0] * 1.15, 21):
    rr = least_squares(lambda y: lj.residuals(np.r_[fv, y]), rj.x[1:], loss="huber", f_scale=0.5, x_scale="jac", max_nfev=200)
    pf.append([fv, 2 * rr.cost, float(np.sqrt(np.mean(rr.fun[:-1] ** 2)))])
pf = np.array(pf)
save_json(dict(profile_joint=pf.tolist()), f"{CACHE}/lines_profile.json")
ax[1].plot(pf[:, 0], pf[:, 2], "-o", ms=2)
ax[1].set_ylabel("edge RMS [px]")
ax[1].set_title("edges only (joint, all else refit): RMS vs f")
ax[1].set_xlabel("f [px]")
cf = np.array(load_json(f"{CACHE}/combined_full.json")["profile"])
ax[2].plot(cf[:, 0], cf[:, 1] - cf[:, 1].min(), "k-o", ms=2, label="total (weighted chi2)")
ax2 = ax[2].twinx()
for j, (k, c) in enumerate(zip("MLVS", "rbgm")):
    ax2.plot(cf[:, 0], cf[:, 2 + j], "-", color=c, lw=0.8, label=k)
ax[2].set_title("combined (main): delta chi2 vs f; block RMS (right)")
ax[2].set_xlabel("f [px]")
ax2.legend(fontsize=7, loc="upper right")
for a in ax:
    a.axvline(K[0, 0], color="gray", ls="--", lw=0.8)
    a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{RESULTS}/f_profiles.png")
print("done")
