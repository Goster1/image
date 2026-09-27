"""Evaluation helpers shared by all methods: mapping differences / uncertainty, regions, plots.

Mapping difference between two lens models (K1,d1) and (K2,d2):
  for pixel u on a grid, take the viewing ray of model 1, rotate it by R and project with model 2;
  displacement = project2(R * ray1(u)) - u.  R = identity ("raw") or the rotation minimising the
  summed squared displacement over the whole grid ("rotation-compensated"): a pure camera rotation is
  indistinguishable from extrinsics, so the compensated field is what matters whenever the pose is
  estimated together with the lens (normal use). Both are reported.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from common import H, W, K_from, project, rodrigues, undistort_points


def grid(step=40, margin=0):
    xs = np.arange(margin, W - margin + 1e-9, step)
    ys = np.arange(margin, H - margin + 1e-9, step)
    X, Y = np.meshgrid(xs, ys)
    return np.column_stack([X.ravel(), Y.ravel()]).astype(float), X.shape


def region_defs():
    """Named image regions (boolean functions of pixel coords)."""
    cx, cy = (W - 1) / 2, (H - 1) / 2

    def centre(uv):
        return np.hypot(uv[:, 0] - cx, uv[:, 1] - cy) < 150

    def corners(uv):
        return ((uv[:, 0] < 200) | (uv[:, 0] > W - 1 - 200)) & ((uv[:, 1] < 150) | (uv[:, 1] > H - 1 - 150))

    def cart_band(uv):
        # union of the two cart footprints (quadrilaterals around both carts incl. top boards,
        # from the observed image, generous by ~20 px)
        from matplotlib.path import Path

        left = Path([(165, 100), (500, -5), (880, 560), (830, 1000), (560, 1010)])
        right = Path([(1170, 0), (1720, 30), (1660, 380), (1540, 980), (1270, 990), (1130, 600)])
        return left.contains_points(uv) | right.contains_points(uv)

    return {"centre": centre, "cart_band": cart_band, "corners": corners}


def rays(uv, K, dist):
    n = undistort_points(uv, K, dist)
    r = np.column_stack([n, np.ones(len(n))])
    return r / np.linalg.norm(r, axis=1, keepdims=True)


def mapping_displacement(K1, d1, K2, d2, uv=None, compensate=True, weights=None):
    """Displacement field (N x 2) of model 2 relative to model 1 at pixels uv (model-1 image)."""
    if uv is None:
        uv, _ = grid(40)
    r = rays(uv, K1, d1)
    ok = r[:, 2] > 0.05
    if not compensate:
        return project(r, K2, d2) - uv
    w = np.ones(len(uv)) if weights is None else weights

    def res(a):
        p = project(r @ rodrigues(a).T, K2, d2)
        return ((p - uv) * np.sqrt(w)[:, None]).ravel()

    a = least_squares(res, np.zeros(3), x_scale=1e-3).x
    return project(r @ rodrigues(a).T, K2, d2) - uv


def mapping_stats(disp_samples, uv):
    """disp_samples: S x N x 2 displacement fields of parameter samples vs the reference.
    Returns per-region 1-sigma radial RMS (median over the region's points and max)."""
    rms = np.sqrt(np.mean(np.sum(disp_samples ** 2, axis=2), axis=0))  # N
    out = {}
    for name, fn in region_defs().items():
        m = fn(uv)
        out[name] = {"rms_median_px": float(np.median(rms[m])), "rms_max_px": float(np.max(rms[m])), "n_points": int(m.sum())}
    out["whole_image"] = {"rms_median_px": float(np.median(rms)), "rms_max_px": float(np.max(rms))}
    return out, rms


def brown_from_mapping(uv, rays_true, init_f=1300.0, dist_names=("k1", "k2", "k3"), pp_free=True, fxfy=False):
    """Fit an OpenCV Brown model to a set of (pixel, ray) pairs (e.g. to convert another lens model)."""
    names = list(dist_names)

    def unpack(x):
        i = 0
        fx = x[0]
        fy = x[1] if fxfy else x[0]
        i = 2 if fxfy else 1
        cx, cy = (x[i], x[i + 1]) if pp_free else ((W - 1) / 2, (H - 1) / 2)
        i += 2 if pp_free else 0
        d = np.zeros(5)
        order = ["k1", "k2", "p1", "p2", "k3"]
        for nm in names:
            d[order.index(nm)] = x[i]
            i += 1
        return K_from(fx, fy, cx, cy), d

    x0 = [init_f] + ([init_f] if fxfy else []) + ([(W - 1) / 2, (H - 1) / 2] if pp_free else []) + [0.0] * len(names)

    def res(x):
        K, d = unpack(x)
        return (project(rays_true, K, d) - uv).ravel()

    r = least_squares(res, x0, x_scale="jac")
    K, d = unpack(r.x)
    return K, d, float(np.sqrt(np.mean(r.fun ** 2) * 2))


def quiver_on_image(ax, img, uv, vec, scale=20.0, color="r", label=None):
    ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
    ax.quiver(uv[:, 0], uv[:, 1], vec[:, 0] * scale, vec[:, 1] * scale, angles="xy", scale_units="xy", scale=1,
              color=color, width=0.002, label=label)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_axis_off()
