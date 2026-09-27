"""Line-based calibration: plumb-line (distortion), vanishing points (f, pp) and the joint version.

Units: all residuals are distances in (distorted) image pixels.

Modes
-----
plumb : unknowns = distortion coefficients (+ optional distortion centre); f0 fixed (not
        identifiable from straightness: only k_i / f^(2i) is). Every straight edge gets its own
        best line (TLS) in the undistorted image.
vp    : distortion fixed; unknowns = f, cx, cy (optionally the pp fixed) + one rotation per cart
        (+ a free direction for scene verticals). Each edge of a direction group must point to
        the group's vanishing point v = K R d.
joint : both at once (distortion + f + pp + rotations): edges with a direction group must be
        straight AND point at their VP; the other straight edges only straight.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from common import H, W, K_from, rodrigues, undistort_points

DORDER = ["k1", "k2", "p1", "p2", "k3"]
AXIS = {"cartX": 0, "cartY": 1, "cartZ": 2}


def group_of(e):
    d = e["direction"]
    if d in AXIS and e.get("cart") in (80, 310):
        return f"{e['cart']}_{d}"
    if d == "world_vertical":
        return "world_vertical"
    return None


class LineCal:
    def __init__(self, edges, mode, dist_free=("k1", "k2"), centre_free=False, pp_free=True, f0=1300.0,
                 dist_fixed=None, centre_fixed=None, subsample=2, vp_groups=None, use_world_vertical=True,
                 tie_world_vertical=None):
        self.edges = edges
        self.mode = mode
        self.dist_free = list(dist_free) if mode != "vp" else []
        self.centre_free = centre_free
        self.pp_free = pp_free
        self.f0 = f0
        self.dist_fixed = np.zeros(5) if dist_fixed is None else np.asarray(dist_fixed, float)
        self.centre_fixed = np.array([(W - 1) / 2, (H - 1) / 2]) if centre_fixed is None else np.asarray(centre_fixed, float)
        self.pts = [e["points"][::subsample] for e in edges]
        self.allp = np.vstack(self.pts)
        self.idx = np.cumsum([0] + [len(p) for p in self.pts])
        self.chord = np.array([np.hypot(*(p[-1] - p[0])) for p in self.pts])
        self.resid_mode = "jacobian"  # or "chord" (older approximation, noise-biased)
        if mode == "plumb":
            self.groups = [None] * len(edges)
        else:
            self.groups = [group_of(e) for e in edges]
            if not use_world_vertical:
                self.groups = [g if g != "world_vertical" else None for g in self.groups]
            if vp_groups is not None:
                self.groups = [g if g in vp_groups else None for g in self.groups]
        self.tie = tie_world_vertical  # e.g. "80" -> world vertical = cart 80 Z
        self.carts = sorted({int(g.split("_")[0]) for g in self.groups if g and g != "world_vertical"})
        self.has_wv = any(g == "world_vertical" for g in self.groups) and self.tie is None
        self.names = []
        if mode in ("vp", "joint"):
            self.names += ["f"] + (["cx", "cy"] if pp_free else [])
        elif centre_free:
            self.names += ["cx", "cy"]
        self.names += self.dist_free
        self.n_intr = len(self.names)
        for c in self.carts:
            self.names += [f"r{c}_{i}" for i in range(3)]
        if self.has_wv:
            self.names += ["wv_a", "wv_b"]

    # ------------------------------------------------------------------
    def unpack(self, x):
        i = 0
        if self.mode in ("vp", "joint"):
            f = x[0]
            i = 1
            if self.pp_free:
                cx, cy = x[1], x[2]
                i = 3
            else:
                cx, cy = self.centre_fixed
        else:
            f = self.f0
            if self.centre_free:
                cx, cy = x[0], x[1]
                i = 2
            else:
                cx, cy = self.centre_fixed
        d = self.dist_fixed.copy()
        for nm in self.dist_free:
            d[DORDER.index(nm)] = x[i]
            i += 1
        rots = {}
        for c in self.carts:
            rots[c] = rodrigues(x[i:i + 3])
            i += 3
        wv = None
        if self.has_wv:
            a, b = x[i], x[i + 1]
            wv = np.array([np.cos(a) * np.sin(b), np.sin(a) * np.sin(b), np.cos(b)])
        return K_from(f, f, cx, cy), d, rots, wv

    def vp(self, g, K, rots, wv):
        if g == "world_vertical":
            d = rots[int(self.tie)][:, 2] if self.tie is not None else wv
        else:
            c, ax = g.split("_")
            d = rots[int(c)][:, AXIS[ax]]
        return K @ d

    def undist_px(self, K, d, pts=None):
        pts = self.allp if pts is None else pts
        n = undistort_points(pts, K, d, iters=30)
        return np.column_stack([K[0, 0] * n[:, 0] + K[0, 2], K[1, 1] * n[:, 1] + K[1, 2]])

    def undist_jac(self, K, d, h=0.5):
        """Undistorted px + 2x2 Jacobian dU/dD per point (finite differences)."""
        U = self.undist_px(K, d)
        Ux = self.undist_px(K, d, self.allp + [h, 0.0])
        Uy = self.undist_px(K, d, self.allp + [0.0, h])
        J = np.stack([(Ux - U) / h, (Uy - U) / h], axis=2)  # N x 2 (U comp) x 2 (D comp)
        return U, J

    def residuals(self, x, per_edge=False):
        """Residuals = distances in the DISTORTED image: r_d = r_u / |J^T n| (first order), which is the
        ML residual under isotropic image noise and has no noise-induced shrinkage bias."""
        K, d, rots, wv = self.unpack(x)
        if self.resid_mode == "jacobian":
            U, J = self.undist_jac(K, d)
        else:
            U, J = self.undist_px(K, d), None
        out = []
        pe = []
        for k in range(len(self.pts)):
            sl = slice(self.idx[k], self.idx[k + 1])
            P = U[sl]
            c = P.mean(0)
            g = self.groups[k]
            if g is None or self.mode == "plumb":
                if self.mode == "vp":
                    out.append(np.zeros(0))
                    pe.append(np.nan)
                    continue
                u, s, vt = np.linalg.svd(P - c, full_matrices=False)
                nvec = vt[1]
                r = (P - c) @ nvec
                nv = np.tile(nvec, (len(P), 1))
            else:
                v = self.vp(g, K, rots, wv)
                l = np.cross(v, np.r_[c, 1.0])
                nn = np.hypot(l[0], l[1])
                r = (P @ l[:2] + l[2]) / nn
                nv = np.tile(l[:2] / nn, (len(P), 1))
            if J is not None:
                jt = np.einsum("nij,ni->nj", J[sl], nv)  # J^T n
                r = r / np.maximum(np.hypot(jt[:, 0], jt[:, 1]), 1e-9)
            else:
                r = r * self.chord[k] / max(np.hypot(*(P[-1] - P[0])), 1e-9)
            out.append(r)
            pe.append(float(np.sqrt(np.mean(r ** 2))) if len(r) else np.nan)
        return pe if per_edge else np.concatenate(out)

    def x0(self, K=None, dist=None, poses=None):
        x = []
        if self.mode in ("vp", "joint"):
            x += [K[0, 0] if K is not None else 1300.0]
            if self.pp_free:
                x += [K[0, 2], K[1, 2]] if K is not None else [(W - 1) / 2, (H - 1) / 2]
        elif self.centre_free:
            x += list(self.centre_fixed)
        for nm in self.dist_free:
            x.append(dist[DORDER.index(nm)] if dist is not None else (-0.2 if nm == "k1" else 0.0))
        for c in self.carts:
            x += list(poses[c][:3]) if poses is not None else [0.0, 0.0, 0.0]
        if self.has_wv:
            z = rodrigues(poses[self.carts[0]][:3])[:, 2] if poses is not None else np.array([0, 0.3, -0.95])
            z = z / np.linalg.norm(z)
            x += [np.arctan2(z[1], z[0]), np.arccos(np.clip(z[2], -1, 1))]
        return np.array(x, float)

    def solve(self, x0, loss="huber", f_scale=0.5, bounds=None):
        kw = dict(loss=loss, f_scale=f_scale, x_scale="jac", max_nfev=400, xtol=1e-10, ftol=1e-10)
        if bounds is not None:
            kw["bounds"] = bounds
        return least_squares(self.residuals, x0, method="trf", **kw)

    def edge_groups_index(self):
        """point -> edge index (for cluster-robust covariance)."""
        return np.concatenate([np.full(len(self.pts[k]) if not (self.mode == "vp" and self.groups[k] is None) else 0, k)
                               for k in range(len(self.pts))])


def sandwich(res, groups):
    J = res.jac
    e = res.fun
    JTJi = np.linalg.pinv(J.T @ J)
    meat = np.zeros_like(JTJi)
    for g in np.unique(groups):
        s = groups == g
        v = J[s].T @ e[s]
        meat += np.outer(v, v)
    G = len(np.unique(groups))
    return JTJi @ meat @ JTJi * G / max(G - 1, 1)


def classic_cov(res):
    J = res.jac
    dof = max(len(res.fun) - len(res.x), 1)
    return np.linalg.pinv(J.T @ J) * (2 * res.cost / dof)
