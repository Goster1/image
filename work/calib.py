"""Generic bundle adjustment: shared intrinsics + one rigid pose per cart.

Observation = one image point with known 3D position in a cart frame (marker corner / centre),
optionally line observations (edge points that must lie on the projection of a known 3D line
segment) are handled in cart_model.py.

Model spec (dict):
    f      : "single" (fx = fy) | "fxfy"
    pp     : "free" | "fixed"          (fixed value = spec["pp0"], default image centre)
    dist   : list of free coefficient names from k1,k2,p1,p2,k3,k4,k5,k6 (others = spec["dist0"] or 0)
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from common import W, H, K_from, fold_margin, project, rodrigues

DNAMES = ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"]


class Model:
    def __init__(self, spec):
        self.spec = dict(spec)
        self.f = spec.get("f", "single")
        self.pp = spec.get("pp", "free")
        self.dfree = list(spec.get("dist", ["k1"]))
        self.pp0 = np.array(spec.get("pp0", [(W - 1) / 2.0, (H - 1) / 2.0]), float)
        self.dist0 = np.zeros(8)
        d0 = spec.get("dist0")
        if d0 is not None:
            self.dist0[: len(d0)] = d0
        self.names = (["f"] if self.f == "single" else ["fx", "fy"]) + (["cx", "cy"] if self.pp == "free" else []) + self.dfree
        self.n = len(self.names)

    def unpack(self, p):
        i = 0
        if self.f == "single":
            fx = fy = p[0]
            i = 1
        else:
            fx, fy = p[0], p[1]
            i = 2
        if self.pp == "free":
            cx, cy = p[i], p[i + 1]
            i += 2
        else:
            cx, cy = self.pp0
        dist = self.dist0.copy()
        for nm in self.dfree:
            dist[DNAMES.index(nm)] = p[i]
            i += 1
        return K_from(fx, fy, cx, cy), dist

    def pack(self, K, dist):
        d = np.zeros(8)
        d[: len(dist)] = dist
        p = [K[0, 0]] if self.f == "single" else [K[0, 0], K[1, 1]]
        if self.pp == "free":
            p += [K[0, 2], K[1, 2]]
        p += [d[DNAMES.index(nm)] for nm in self.dfree]
        return np.array(p, float)

    def describe(self):
        s = ("fx=fy" if self.f == "single" else "fx,fy") + ", pp " + ("free" if self.pp == "free" else "fixed") + ", dist " + "+".join(self.dfree)
        return s


def dist_trim(dist):
    """8-vector -> OpenCV 5 (or 8 when rational terms are used)."""
    d = np.asarray(dist, float)
    return d if np.any(d[5:] != 0) else d[:5]


class Problem:
    """points: list of dicts with keys cart, X (3,), uv (2,), w (weight = 1/sigma)."""

    def __init__(self, model: Model, points, carts=None):
        self.model = model
        self.carts = sorted({p["cart"] for p in points}) if carts is None else list(carts)
        self.cidx = {c: i for i, c in enumerate(self.carts)}
        self.X = np.array([p["X"] for p in points], float)
        self.uv = np.array([p["uv"] for p in points], float)
        self.w = np.array([p.get("w", 1.0) for p in points], float)
        self.ci = np.array([self.cidx[p["cart"]] for p in points])
        self.points = points

    def split(self, x):
        n = self.model.n
        K, dist = self.model.unpack(x[:n])
        poses = x[n:].reshape(-1, 6)
        return K, dist, poses

    def predict(self, x):
        K, dist, poses = self.split(x)
        out = np.zeros_like(self.uv)
        for i, _c in enumerate(self.carts):
            m = self.ci == i
            if m.any():
                out[m] = project(self.X[m], K, dist, rvec=poses[i, :3], tvec=poses[i, 3:])
        return out

    fold_barrier = False  # set True to forbid lenses that fold inside the image (non-invertible)

    def residuals(self, x):
        r = (self.predict(x) - self.uv) * self.w[:, None]
        if self.fold_barrier:
            K, d, _ = self.split(x)
            return np.r_[r.ravel(), 1000.0 * max(0.0, 0.03 - fold_margin(K, d))]
        return r.ravel()

    def solve(self, x0, loss="linear", f_scale=1.0):
        res = least_squares(self.residuals, x0, method="trf" if loss != "linear" else "lm", loss=loss,
                            f_scale=f_scale, x_scale="jac", max_nfev=20000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
        return res


def covariance(res, sigma2=None):
    """Covariance of parameters from a least_squares result (Gauss-Newton)."""
    J = res.jac
    dof = max(J.shape[0] - J.shape[1], 1)
    if sigma2 is None:
        sigma2 = 2 * res.cost / dof
    JTJ = J.T @ J
    try:
        C = np.linalg.inv(JTJ) * sigma2
    except np.linalg.LinAlgError:
        C = np.linalg.pinv(JTJ) * sigma2
    return C, sigma2, dof


def pose_from_Rt(R, t):
    import cv2
    return np.concatenate([cv2.Rodrigues(R)[0].ravel(), np.asarray(t, float).ravel()])


def camera_centre(pose):
    R = rodrigues(pose[:3])
    return -R.T @ pose[3:]
