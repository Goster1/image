"""Review of stage 11 (markers): empirical edge-location bias (tone curve / blur asymmetry) measured
from the INTERIOR of the codes, to replace the "unknown gamma" assumption.

Idea: every black/white transition inside a code sits at an integer cell coordinate k (15 mm cells).
If the edge criterion used for the corners (gradient maximum of the blurred profile) is displaced
by b px towards the white side (black looks dilated; b < 0 = black looks eroded), the four outer
sides are displaced outward by the same b, the corners (and the homography built from them) are
scaled by 6/(6+2*beta) (beta = b / cell), and an interior transition at k with white on the +u side
(s = +1) or on the -u side (s = -1) appears, in the cell coordinates of that homography, displaced
towards white by   delta = beta * (1 + s * (1 - k/3)).
beta is fitted by least squares over all interior transitions of the fully visible stickers
(rows and columns, 5 sample lines per cell row), on the mean image and per still.

Output: work/cache/review_markers/bias.json, printed summary.
"""
import importlib
import os

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates

from common import CACHE, load_json, save_json

L = importlib.import_module("11_markers_lib")
OUT = f"{CACHE}/review_markers"
os.makedirs(OUT, exist_ok=True)


def transitions(img, quad, grid, step=0.01, sigma_px=0.8):
    """Interior transitions (u or v direction) of one code. quad: ArUco order corners (mean frame).
    grid: 6x6 cells (1 = white), canonical orientation (row = v, col = u; corner 0 at (0,0)).
    Returns list of (k, s, delta_cells, cell_px)."""
    Hm = cv2.getPerspectiveTransform(np.array([[0, 0], [6, 0], [6, 6], [0, 6]], np.float32), np.asarray(quad, np.float32)).astype(float)
    side = np.linalg.norm(np.roll(quad, -1, 0) - quad, axis=1).mean()
    cell = side / 6.0
    rows = []
    t = np.arange(-0.6, 6.6 + 1e-9, step)
    for axis in (0, 1):  # 0: lines along u (vary u, fixed v); 1: along v
        g = grid if axis == 0 else grid.T  # g[line_cell, pos_cell]
        for lc in range(1, 5):
            ext = np.r_[1, g[lc], 1]  # white margin on both ends
            for fv in (0.3, 0.4, 0.5, 0.6, 0.7):
                w = lc + fv
                if axis == 0:
                    uv = np.column_stack([t, np.full_like(t, w)])
                else:
                    uv = np.column_stack([np.full_like(t, w), t])
                q = np.column_stack([uv, np.ones(len(uv))]) @ Hm.T
                P = q[:, :2] / q[:, 2:3]
                prof = map_coordinates(img, [P[:, 1], P[:, 0]], order=3, mode="nearest")
                ds = np.median(np.linalg.norm(np.diff(P, axis=0), axis=1))  # px per sample
                der = gaussian_filter1d(prof, sigma_px / ds, order=1)
                for k in range(1, 6):  # interior boundaries only
                    a, b = ext[k], ext[k + 1]  # cell k-1 (index k in ext) and cell k
                    if a == b:
                        continue
                    s = 1 if b == 1 else -1  # white on the +side
                    # run lengths (cells) of the white and the black side (margin counts as long run = 9)
                    def run(idx, step_):
                        v = ext[idx]
                        n = 0
                        while 0 <= idx < len(ext) and ext[idx] == v:
                            n += 9 if idx in (0, len(ext) - 1) else 1
                            idx += step_
                        return n
                    wrun = run(k + 1, 1) if s == 1 else run(k, -1)
                    brun = run(k, -1) if s == 1 else run(k + 1, 1)
                    sel = np.where(np.abs(t - k) < 0.4)[0]
                    dd = s * der[sel]
                    j = int(np.argmax(dd))
                    if j == 0 or j == len(sel) - 1:
                        continue
                    y0, y1, y2 = dd[j - 1], dd[j], dd[j + 1]
                    den = y0 - 2 * y1 + y2
                    off = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
                    u = t[sel[j]] + off * step
                    rows.append((k, s, s * (u - k), cell, wrun, brun))
    return rows


def fit_two(rows):
    """delta = beta_i + beta_o * s * (1 - k/3): separate interior-edge and outer-edge biases (cells)."""
    R = np.array(rows, float)
    k, s, d = R[:, 0], R[:, 1], R[:, 2]
    A = np.column_stack([np.ones_like(k), s * (1 - k / 3)])
    x, *_ = np.linalg.lstsq(A, d, rcond=None)
    r = d - A @ x
    C = np.linalg.inv(A.T @ A) * np.sum(r ** 2) / (len(d) - 2)
    return x, np.sqrt(np.diag(C)), float(np.sqrt(np.mean(r ** 2)))


def fit_beta(rows):
    R = np.array(rows, float)
    k, s, d = R[:, 0], R[:, 1], R[:, 2]
    a = 1 + s * (1 - k / 3)
    beta = float(np.sum(a * d) / np.sum(a * a))
    r = d - beta * a
    se = float(np.sqrt(np.sum(r ** 2) / (len(d) - 1) / np.sum(a * a)))
    return beta, se, float(np.sqrt(np.mean(r ** 2)))


def main():
    M = load_json(f"{CACHE}/markers_final.json")
    mean = L.mean_gray()
    frames = L.frames()
    out = dict(per_marker={}, note=__doc__.split("\n\n")[1].replace("\n", " "))
    allrows = []
    per_frame_beta = []
    for m in M:
        if not all(m["corner_valid"]):
            continue
        grid = L.code_grid(m["id"])
        q = np.array(m["corners_px"], float)
        rows = transitions(mean, q, grid)
        cell = rows[0][3]
        beta, se, rr = fit_beta(rows)
        x2, se2, rr2 = fit_two(rows)
        R0 = np.array(rows, float)
        iso = R0[:, 4] == 1
        out["per_marker"][f"{m['cart']}:{m['id']}"] = dict(n=len(rows), beta_cells=beta, beta_se=se, b_px=beta * cell,
                                                           cell_px=cell, resid_rms_cells=rr,
                                                           two_param_b_inner_px=x2[0] * cell, two_param_b_outer_px=x2[1] * cell,
                                                           two_param_se_px=list(se2 * cell),
                                                           delta_px_white_run1=float(R0[iso, 2].mean() * cell),
                                                           delta_px_white_run_ge2=float(R0[~iso, 2].mean() * cell))
        print(f"{m['cart']:4d}:{m['id']:<4d} n={len(rows):3d} cell={cell:5.2f}px  b(common) = {beta * cell:+.3f} +- {se * cell:.3f} px"
              f"  | b_inner {x2[0] * cell:+.3f}+-{se2[0] * cell:.3f}  b_outer {x2[1] * cell:+.3f}+-{se2[1] * cell:.3f} (rms {rr2 * cell:.3f})"
              f" | delta(white run 1) {R0[iso, 2].mean() * cell:+.3f}  (run>=2) {R0[~iso, 2].mean() * cell:+.3f}")
        allrows += [(k, s, d, c, beta * c) for k, s, d, c, *_ in rows]
    R = np.array(allrows)
    # pooled estimate in px: delta_px = b * a  (b in px common)
    a = 1 + R[:, 1] * (1 - R[:, 0] / 3)
    dpx = R[:, 2] * R[:, 3]
    b = float(np.sum(a * dpx) / np.sum(a * a))
    bm = np.array([v["b_px"] for v in out["per_marker"].values()])
    out["pooled_b_px"] = b
    out["marker_mean_b_px"] = float(bm.mean())
    out["marker_sem_b_px"] = float(bm.std(ddof=1) / np.sqrt(len(bm)))
    print(f"pooled b = {b:+.3f} px ; mean over stickers {bm.mean():+.3f} +- {bm.std(ddof=1) / np.sqrt(len(bm)):.3f} px (sem)")
    bi = np.array([v["two_param_b_inner_px"] for v in out["per_marker"].values()])
    bo = np.array([v["two_param_b_outer_px"] for v in out["per_marker"].values()])
    out["two_param_mean_px"] = dict(b_inner=float(bi.mean()), b_inner_sem=float(bi.std(ddof=1) / np.sqrt(len(bi))),
                                    b_outer=float(bo.mean()), b_outer_sem=float(bo.std(ddof=1) / np.sqrt(len(bo))))
    print("two-parameter model, mean over stickers:", {k: round(v, 3) for k, v in out["two_param_mean_px"].items()})
    if os.environ.get("NO_FRAMES"):
        save_json(out, f"{OUT}/bias.json")
        return
    # per still (corners of the still = jitter-mapped per-frame corners)
    for f, (name, g) in enumerate(frames):
        rows_f = []
        for m in M:
            if not all(m["corner_valid"]):
                continue
            q = np.array(m["per_frame"][f], float)
            q = L.mean_to_frame(q, f)
            rr = transitions(g, q, L.code_grid(m["id"]))
            rows_f += [(k, s, d * c, c) for k, s, d, c, *_ in rr]
        Rf = np.array(rows_f)
        af = 1 + Rf[:, 1] * (1 - Rf[:, 0] / 3)
        bf = float(np.sum(af * Rf[:, 2]) / np.sum(af * af))
        per_frame_beta.append(bf)
        print(f"  still {name}: b = {bf:+.3f} px")
    out["per_still_b_px"] = per_frame_beta
    out["interpretation"] = ("b > 0: the edge criterion places black/white edges towards the white side (black dilated), so "
                             "edge corners lie outside the true corners by ~b along each side normal (~b*sqrt(2) along the diagonal); "
                             "b < 0: black eroded, corners inside.")
    save_json(out, f"{OUT}/bias.json")


if __name__ == "__main__":
    main()
