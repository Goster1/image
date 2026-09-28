"""Helpers for the independent review of the scene edge set (work/cache/edges_scene.json).

render_edge(): one PNG per edge for visual checking
  row 1: context crop of the colour mean image with the stored points (red) drawn, zoomed
  row 2: straightened strip of the gray mean image along the stored points (normal offsets
         -hw..+hw, zoomed x6 across the edge) with the stored points drawn (red); the edge must be
         a straight, clean step in the strip and the points must sit on it
  row 3: residual plots: (a) to a smooth cubic in arc length (local jumps / kinks),
         (b) to a straight line after undistortion with reference lenses (distances in the
         distorted image) - only a plausibility view, never used to select points.
Reference lenses are only used for diagnostics.
"""
from __future__ import annotations

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates

from common import CACHE, K_from, load_json, undistort_points

GRAY = np.load(f"{CACHE}/mean_aligned_gray.npy").astype(np.float64)
COLOR = cv2.imread(f"{CACHE}/mean_aligned_color.png")


def ref_lenses():
    out = {}
    for name in ("lines_joint", "plumbline", "combined"):
        d = load_json(f"{CACHE}/method_{name}.json")
        if isinstance(d, list):
            d = d[0]
        K = np.array(d["camera_matrix"], float)
        out[name] = (K, np.array(d["dist_coeffs"], float))
    return out


REFS = ref_lenses()


def arclen(P):
    return np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]


def smooth_path(P, deg=3):
    s = arclen(P)
    px = np.polyfit(s, P[:, 0], deg)
    py = np.polyfit(s, P[:, 1], deg)
    return s, px, py


def normals_along(s, px, py):
    dx = np.polyval(np.polyder(px), s)
    dy = np.polyval(np.polyder(py), s)
    n = np.hypot(dx, dy)
    d = np.column_stack([dx / n, dy / n])
    return d, np.column_stack([-d[:, 1], d[:, 0]])


def local_resid(P, deg=3):
    """Signed normal residual of the points to a smooth polynomial path (px)."""
    s, px, py = smooth_path(P, deg)
    Q = np.column_stack([np.polyval(px, s), np.polyval(py, s)])
    _, nrm = normals_along(s, px, py)
    return s, np.sum((P - Q) * nrm, axis=1)


def straight_resid_distorted(P, K, dist):
    """Straight-line residual after undistortion, expressed as distances in the distorted image."""
    n = undistort_points(P, K, dist)
    U = np.column_stack([K[0, 0] * n[:, 0] + K[0, 2], K[1, 1] * n[:, 1] + K[1, 2]])
    c = U.mean(0)
    _, _, Vt = np.linalg.svd(U - c)
    d = Vt[0]
    nn = np.array([-d[1], d[0]])
    r_u = (U - c) @ nn
    # Jacobian of the undistortion map (numeric) to convert to distorted-image distances
    h = 0.5
    Ux = undistort_points(P + [h, 0], K, dist)
    Uy = undistort_points(P + [0, h], K, dist)
    Ux = np.column_stack([K[0, 0] * Ux[:, 0] + K[0, 2], K[1, 1] * Ux[:, 1] + K[1, 2]])
    Uy = np.column_stack([K[0, 0] * Uy[:, 0] + K[0, 2], K[1, 1] * Uy[:, 1] + K[1, 2]])
    J1 = (Ux - U) / h  # dU/dx
    J2 = (Uy - U) / h  # dU/dy
    g = np.column_stack([J1 @ nn, J2 @ nn])  # gradient of the normal coordinate wrt distorted pixel
    return r_u / np.linalg.norm(g, axis=1)


def strip(P, hw=8.0, zoom=6, step=1.0):
    """Straightened strip along the smooth path through P. Returns image (uint8), offsets of P."""
    s, px, py = smooth_path(P, 3)
    t = np.arange(s.min(), s.max() + 1e-9, step)
    Q = np.column_stack([np.polyval(px, t), np.polyval(py, t)])
    _, nrm = normals_along(t, px, py)
    offs = np.arange(-hw, hw + 1e-9, 1.0 / zoom)
    X = Q[:, None, 0] + nrm[:, None, 0] * offs[None]
    Y = Q[:, None, 1] + nrm[:, None, 1] * offs[None]
    prof = map_coordinates(GRAY, [Y.ravel(), X.ravel()], order=3, mode="nearest").reshape(X.shape)
    lo, hi = np.percentile(prof, [1, 99])
    img = np.clip((prof - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8).T  # rows = offsets
    # offsets of the stored points relative to the smooth path
    Qp = np.column_stack([np.polyval(px, s), np.polyval(py, s)])
    _, nrmp = normals_along(s, px, py)
    o = np.sum((P - Qp) * nrmp, axis=1)
    return img, s - s.min(), o, offs


def render_edge(e, fn, title_extra=""):
    P = np.array(e["points"], float)
    x0, y0 = np.floor(P.min(0) - 40).astype(int)
    x1, y1 = np.ceil(P.max(0) + 40).astype(int)
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, 1919), min(y1, 1079)
    crop = COLOR[y0:y1 + 1, x0:x1 + 1].copy()
    z = min(900 / max(crop.shape[:2]), 6.0)
    crop = cv2.resize(crop, None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC)
    for p in P[::2]:
        c = (int(round((p[0] - x0 + 0.5) * z - 0.5)), int(round((p[1] - y0 + 0.5) * z - 0.5)))
        cv2.circle(crop, c, 1, (0, 0, 255), -1)
    img, s, o, offs = strip(P)
    fig = plt.figure(figsize=(16, 11), dpi=90)
    gs = fig.add_gridspec(3, 2, height_ratios=[2.2, 1.1, 1.0])
    ax = fig.add_subplot(gs[0, :])
    ax.imshow(crop[:, :, ::-1])
    ax.set_title(f"{e['id']}  [{e['direction']}, {e.get('verified')}]  crop x{z:.1f} at ({x0},{y0}) {title_extra}")
    ax.axis("off")
    ax = fig.add_subplot(gs[1, :])
    ax.imshow(img, cmap="gray", aspect="auto", extent=[0, s.max(), offs[-1], offs[0]])
    ax.plot(s, o, "r.", ms=2)
    ax.set_ylabel("normal offset px")
    ax.set_title("straightened strip (gray mean), red = stored points")
    ax = fig.add_subplot(gs[2, 0])
    sl, rl = local_resid(P, 3)
    ax.plot(sl, rl, ".-", ms=2, lw=0.5)
    ax.set_title(f"resid to smooth cubic: rms {np.sqrt(np.mean(rl**2)):.2f} max {np.abs(rl).max():.2f}")
    ax.grid(alpha=0.3)
    ax = fig.add_subplot(gs[2, 1])
    for name, (K, dist) in REFS.items():
        r = straight_resid_distorted(P, K, dist)
        ax.plot(sl, r, ".-", ms=2, lw=0.5, label=f"{name}: rms {np.sqrt(np.mean(r**2)):.2f}")
    ax.legend(fontsize=8)
    ax.set_title("straight-line resid after undistortion (distorted px)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fn)
    plt.close(fig)


def window(P, centre, half=18, zoom=12, pts=True):
    """Zoomed window (gray mean, cubic) around `centre` with stored points drawn."""
    cx, cy = int(round(centre[0])), int(round(centre[1]))
    x0, y0 = max(cx - half, 0), max(cy - half, 0)
    x1, y1 = min(cx + half, 1919), min(cy + half, 1079)
    g = GRAY[y0:y1 + 1, x0:x1 + 1]
    lo, hi = np.percentile(g, [1, 99])
    g = np.clip((g - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)
    g = cv2.resize(g, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
    g = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    if pts:
        for p in P:
            if x0 <= p[0] <= x1 and y0 <= p[1] <= y1:
                c = (int(round((p[0] - x0 + 0.5) * zoom - 0.5)), int(round((p[1] - y0 + 0.5) * zoom - 0.5)))
                cv2.circle(g, c, 2, (0, 0, 255), -1)
    return g
