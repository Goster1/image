"""Synthetic edges with the real geometry (for round-trip tests of the line methods).

Each real edge is undistorted with the current estimate (K_est, d_est) to get its ideal pinhole
line; for direction-group edges the line is replaced by the line through the TRUE vanishing point
and the edge centroid; exact model lines use the projection of the 3D line with the true pose.
Points are resampled over the same extent, distorted with the TRUE model and perturbed by white
noise (sigma_px) plus an optional smooth per-edge bow (sigma_bow_px at mid-length).
"""
from __future__ import annotations

import numpy as np

from combined import AXIS, model_line_3d
from common import project, rodrigues, undistort_points


def ideal_edges(edges, K_est, d_est, K_true, d_true, poses_true=None, wv_true=None, rng=None, sigma_px=0.1,
                sigma_bow_px=0.0, spacing=2.0):
    rng = np.random.default_rng(0) if rng is None else rng
    out = []
    Kinv = np.linalg.inv(K_true)
    for e in edges:
        P = e["points"]
        n = undistort_points(P, K_est, d_est)
        U = np.column_stack([K_true[0, 0] * n[:, 0] + K_true[0, 2], K_true[1, 1] * n[:, 1] + K_true[1, 2]])
        c = U.mean(0)
        line = None
        ml = model_line_3d(e.get("model_line"))
        if ml is not None and poses_true is not None and e.get("cart") in poses_true:
            A, D = ml
            p = poses_true[e["cart"]]
            R = rodrigues(p[:3])
            a = K_true @ (R @ A + p[3:])
            b = K_true @ (R @ D)
            line = np.cross(a, b)
        elif poses_true is not None and e.get("direction") in AXIS and e.get("cart") in poses_true:
            R = rodrigues(poses_true[e["cart"]][:3])
            v = K_true @ R[:, AXIS[e["direction"]]]
            line = np.cross(v, np.r_[c, 1.0])
        elif e.get("direction") == "world_vertical" and wv_true is not None:
            v = K_true @ wv_true
            line = np.cross(v, np.r_[c, 1.0])
        if line is None:
            u, s, vt = np.linalg.svd(U - c, full_matrices=False)
            dvec = vt[0]
            nvec = vt[1]
            off = 0.0
        else:
            nvec = line[:2] / np.hypot(line[0], line[1])
            off = -line[2] / np.hypot(line[0], line[1])  # nvec . x = off
            dvec = np.array([-nvec[1], nvec[0]])
        # extent along the line (from the real points)
        t = (U - c) @ dvec
        t0, t1 = t.min(), t.max()
        base = c - ((c @ nvec) - off) * nvec if line is not None else c
        m = max(int((t1 - t0) / spacing), 5)
        ts = np.linspace(t0, t1, m)
        Q = base[None] + ts[:, None] * dvec[None]
        if sigma_bow_px > 0:
            s = (ts - t0) / max(t1 - t0, 1e-9)
            Q = Q + (rng.normal(0, sigma_bow_px) * 4 * s * (1 - s))[:, None] * nvec[None]
        # distort with the true model
        rays = np.column_stack([Q, np.ones(len(Q))]) @ Kinv.T
        D = project(rays, K_true, d_true)
        D = D + rng.normal(0, sigma_px, D.shape) * 0  # isotropic noise added along the normal below
        # noise along the local normal of the distorted curve
        tang = np.gradient(D, axis=0)
        tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-12
        nrm = np.column_stack([-tang[:, 1], tang[:, 0]])
        D = D + rng.normal(0, sigma_px, len(D))[:, None] * nrm
        e2 = dict(e)
        e2["points"] = D
        out.append(e2)
    return out
