"""Stage 11d: diagnostics of the complete sticker set (no new corner data).

1. Edge profile analysis of the black-square sides (mean image): blur width sigma, the intensity
   level (0 = black border, 1 = white margin) at the sub-pixel gradient maximum used by the edge
   corners, and the shift that a non-linear tone curve would cause (if the camera encoded linear
   light with a 1/2.2 power law, the physical edge would sit at the 0.73 level of the encoded step,
   not at the gradient maximum) -> sensitivity of the black-square size to the tone curve.
2. Residual structure of the joint fit (f single, pp free, k1, k2; all valid corners): per marker
   observed / predicted side length ratio, mean residual vector, and a diagnostic refit with one
   free height offset per shelf level (A, B, C, common to both carts) - NOT a result, only to show
   where the inconsistency sits.
3. Plot results/markers_complete_residuals.png (residual vectors x25 on the mean image).
"""
import importlib

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates
from scipy.optimize import least_squares
from scipy.special import erf

from calib import Model, Problem
from common import CACHE, RESULTS, load_json, marker_center, marker_corners_3d, project, save_json

L = importlib.import_module("11_markers_lib")
C = importlib.import_module("11_markers_c_fit")


def edge_profiles(M, img):
    rows = []
    for m in M:
        q = np.array(m["corners_px"], float)
        side = np.linalg.norm(np.roll(q, -1, 0) - q, axis=1).mean()
        cell = side / 6
        for i, ln in enumerate(m["side_lines"]):
            if not ln["valid"]:
                continue
            a, b = q[i], q[(i + 1) % 4]
            d = (b - a) / np.linalg.norm(b - a)
            n = np.array([-d[1], d[0]])
            if np.dot(q.mean(0) - 0.5 * (a + b), n) > 0:
                n = -n  # outward
            p0, dd = np.array(ln["point"]), np.array(ln["direction"])
            nn = np.array([-dd[1], dd[0]])
            if np.dot(nn, n) < 0:
                nn = -nn
            L_ = np.linalg.norm(b - a)
            ts = np.linspace(-0.3 * L_, 0.3 * L_, 25)
            offs = np.arange(-0.9 * cell, 0.6 * cell + 1e-9, 0.1)
            # line point closest to the side centre
            c0 = p0 + np.dot(0.5 * (a + b) - p0, dd) * dd
            P = c0[None, None] + ts[:, None, None] * dd[None, None] + offs[None, :, None] * nn[None, None]
            prof = map_coordinates(img, [P[..., 1].ravel(), P[..., 0].ravel()], order=3).reshape(P.shape[:2]).mean(0)
            # erf fit on the transition region
            sel = (offs > -0.45 * cell) & (offs < 0.45 * cell)

            def f(x):
                return x[0] + (x[1] - x[0]) * 0.5 * (1 + erf((offs[sel] - x[2]) / (np.sqrt(2) * x[3]))) - prof[sel]

            r = least_squares(f, [prof.min(), prof.max(), 0.0, 1.0])
            blk, wht, x50, sig = r.x
            der = gaussian_filter1d(prof, 0.8 / 0.1, order=1)
            jm = np.argmax(der[5:-5]) + 5
            y0, y1, y2 = der[jm - 1], der[jm], der[jm + 1]
            xg = offs[jm] + 0.1 * 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
            lev_g = (np.interp(xg, offs, prof) - blk) / (wht - blk)
            # level 0.73 location (pure 1/2.2 power-law encoding of a linear 50 % edge)
            normp = (prof - blk) / (wht - blk)
            k = np.where((normp[:-1] < 0.73) & (normp[1:] >= 0.73) & (offs[:-1] > -0.4 * cell))[0]
            x73 = offs[k[0]] + 0.1 * (0.73 - normp[k[0]]) / (normp[k[0] + 1] - normp[k[0]]) if len(k) else np.nan
            rows.append(dict(cart=m["cart"], id=m["id"], side=i, sigma_px=abs(sig), gradmax_minus_x50_px=xg - x50,
                             level_at_gradmax=lev_g, x73_minus_gradmax_px=x73 - xg, cell_px=cell))
    return rows


def level_offset_diag(M, rot):
    """Refit with a free Z offset per shelf level (A, B, C), common to both carts."""
    pts = C.build_points(M, rot, "final")
    lev = {-195.0: 0, -595.0: 1, -995.0: 2}
    model = Model(C.MODEL)
    pr = Problem(model, pts, carts=[80, 310])
    x0 = C.seed_x(model)
    li = np.array([lev.get(marker_center(p["id"], p["cart"])[2], -1) for p in pts])

    def res(x):
        X = pr.X.copy()
        for k in range(3):
            X[li == k, 2] += x[-3 + k]
        pr_X = pr.X
        pr.X = X
        r = pr.residuals(x[:-3])
        pr.X = pr_X
        return r

    r0 = least_squares(pr.residuals, x0, method="lm", x_scale="jac")
    r = least_squares(res, np.r_[r0.x, 0, 0, 0], method="lm", x_scale="jac")
    rms0 = np.sqrt(np.mean(r0.fun.reshape(-1, 2) ** 2) * 2)
    rms1 = np.sqrt(np.mean(r.fun.reshape(-1, 2) ** 2) * 2)
    return dict(rms_drawing_px=float(rms0), rms_free_level_offsets_px=float(rms1),
                dz_mm={"A": float(r.x[-3]), "B": float(r.x[-2]), "C": float(r.x[-1])},
                f_drawing=float(r0.x[0]), f_free_levels=float(r.x[0]))


def main():
    M = load_json(f"{CACHE}/markers_final.json")
    S = load_json(f"{CACHE}/markers_summary.json")
    img = L.mean_gray()
    rows = edge_profiles(M, img)
    R = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    edge = dict(n_sides=len(rows), sigma_px_median=float(np.median(R["sigma_px"])),
                sigma_px_p10_p90=[float(np.percentile(R["sigma_px"], 10)), float(np.percentile(R["sigma_px"], 90))],
                level_at_gradmax_median=float(np.median(R["level_at_gradmax"])),
                gradmax_minus_x50_px_mean=float(np.mean(R["gradmax_minus_x50_px"])),
                gradmax_minus_x50_px_std=float(np.std(R["gradmax_minus_x50_px"])),
                x73_minus_gradmax_px_median=float(np.nanmedian(R["x73_minus_gradmax_px"])),
                note=("positive = outward (towards the white margin). If the tone curve were a pure 1/2.2 power law the true edge would be"
                      " x73_minus_gradmax further out on every side, i.e. the black square would be measured too small by about"
                      " twice that value per side length; a real camera tone curve is unknown -> this is a sensitivity bound only."))
    print("edge profiles:", {k: (np.round(v, 3) if not isinstance(v, str) else v[:40]) for k, v in edge.items()})

    rot = {(m["cart"], m["id"]): m["rot_k"] for m in M}
    diag = level_offset_diag(M, rot)
    print("level offset diagnostic:", diag)

    # observed / predicted size per marker in the final fit
    fin_res = S["residuals_final"]
    pts = C.build_points(load_json(f"{CACHE}/markers_corners.json"), rot, "final")
    pr, r, res = C.fit(pts)
    K, dist, poses = pr.split(r.x)
    per_marker = []
    for m in M:
        pose = poses[0 if m["cart"] == 80 else 1]
        X = marker_corners_3d(m["id"], m["cart"], m["rot_k"])
        p = project(X, K, dist, rvec=pose[:3], tvec=pose[3:])
        q = np.array(m["corners_px"], float)
        ok = np.array(m["corner_valid"])
        # side ratio from sides whose both corners are valid
        rat = [np.linalg.norm(q[(i + 1) % 4] - q[i]) / np.linalg.norm(p[(i + 1) % 4] - p[i]) for i in range(4) if ok[i] and ok[(i + 1) % 4]]
        d = (q - p)[ok]
        per_marker.append(dict(cart=m["cart"], id=m["id"], role=m["role"], n_valid=int(ok.sum()),
                               size_ratio_obs_over_model=float(np.mean(rat)) if rat else None,
                               mean_residual_px=d.mean(0) if len(d) else None,
                               rms_px=float(np.sqrt(np.mean(np.sum(d ** 2, 1)))) if len(d) else None))
        print(f"{m['cart']:4d}:{m['id']:<4d} {m['role']:8s} n={ok.sum()} size obs/model="
              f"{(np.mean(rat) if rat else float('nan')):.3f} mean res={np.round(d.mean(0), 2) if len(d) else None} rms={per_marker[-1]['rms_px']}")

    # residual plot
    col = L.mean_color()
    ov = col.copy()
    for p_, d_ in zip(pts, res):
        u = np.array(p_["uv"])
        z = marker_center(p_["id"], p_["cart"])[2]
        c = {0.0: (255, 255, 0), -195.0: (0, 0, 255), -595.0: (0, 200, 0), -995.0: (255, 0, 255)}[z]
        cv2.circle(ov, tuple(np.round(u).astype(int)), 3, c, -1)
        cv2.arrowedLine(ov, tuple(np.round(u).astype(int)), tuple(np.round(u + 25 * d_).astype(int)), c, 2, cv2.LINE_AA, tipLength=0.15)
    cv2.rectangle(ov, (0, 0), (1920, 34), (0, 0, 0), -1)
    cv2.putText(ov, f"joint fit f single, pp free, k1 k2, all {len(pts)} valid corners: rms {np.sqrt(np.mean(np.sum(res ** 2, 1))):.2f} px;"
                f" arrows = model - measured x25; cyan top plate, red shelf A, green B, magenta C", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(f"{RESULTS}/markers_complete_residuals.png", ov)
    save_json(dict(edge_profiles=edge, edge_profile_rows=rows, level_offset_diagnostic=diag, per_marker=per_marker),
              f"{CACHE}/markers_diag.json")


if __name__ == "__main__":
    main()
