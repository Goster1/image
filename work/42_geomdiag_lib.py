"""Helpers for the sticker-geometry diagnosis (42_geomdiag_*.py).

* plumb-line distortion variants (LineCal 'plumb' on all verified straight edges), cached;
  the undistortion is done in PIXEL units with f0 = 1300 (only k_i / f^(2i) matter, so every
  projective statement below is independent of f and of the principal point);
* undistortion + its Jacobian (to convert residuals back to distorted-image pixels);
* vertical vanishing point (nadir) from vertical edges (cartZ + world_vertical), with covariance;
* sticker columns: corresponding corners of the stacked stickers (top, A, B, C at the same X, Y)
  plus, for hidden corners, a valid traced side line through that corner.

Import with importlib (file name starts with a digit):  G = importlib.import_module("42_geomdiag_lib")
"""
from __future__ import annotations

import hashlib
import os

import numpy as np
from scipy.optimize import least_squares

from common import CACHE, H, W, K_from, load_json, marker_center, save_json, undistort_points
from linedata import load_edges
from lineselfcal import LineCal, sandwich
from markerdata import load_markers

F0 = 1300.0
CEN = np.array([(W - 1) / 2.0, (H - 1) / 2.0])
PLUMB_CACHE = f"{CACHE}/geomdiag_plumb.json"
Z_LEVEL = {"top": 0.0, "A": -195.0, "B": -595.0, "C": -995.0}
END_IDS = {0: {"top": 0, "A": 3, "B": 5, "C": 7}, 1600: {"top": 1, "A": 4, "B": 6, "C": 8}}
LEVELS = ["top", "A", "B", "C"]


# ------------------------------------------------------------------------------------------
# edges / plumb-line distortion
# ------------------------------------------------------------------------------------------
EDGE_FILES = ["edges_cart310.json", "edges_cart80.json", "edges_scene.json"]


def all_edges():
    """Verified straight edges from the final per-region files (explicit list: intermediate files such as
    edges_scene_raw.json also match edges_*.json and must not be used)."""
    out = []
    for fn in EDGE_FILES:
        p = f"{CACHE}/{fn}"
        if os.path.exists(p):
            try:
                out += load_edges(verified_only=True, source=p)
            except Exception as ex:  # noqa  (file being written by another process)
                print("could not read", fn, ex)
    return out


def _sig(edges):
    return hashlib.md5(";".join(sorted(e["id"] + f"{len(e['points'])}" for e in edges)).encode()).hexdigest()[:12]


PLUMB_VARIANTS = {
    "k1k2_centre_fixed": dict(dist=("k1", "k2"), centre_free=False),
    "k1k2_centre_free": dict(dist=("k1", "k2"), centre_free=True),
    "k1_centre_fixed": dict(dist=("k1",), centre_free=False),
    "k1k2k3_centre_free": dict(dist=("k1", "k2", "k3"), centre_free=True),
}


def plumb(variant="k1k2_centre_fixed", edges=None, force=False):
    """Return dict(centre, dist (OpenCV at F0), dist_px, rms, n_edges, sigma) for a plumb-line variant."""
    if variant == "none":
        return dict(centre=CEN.tolist(), dist=[0.0] * 5, dist_px=[0.0] * 5, rms=None, n_edges=0, sig={})
    edges = all_edges() if edges is None else edges
    sig = _sig(edges)
    cache = load_json(PLUMB_CACHE) if os.path.exists(PLUMB_CACHE) else {}
    key = f"{variant}|{sig}"
    if key in cache and not force:
        return cache[key]
    kw = PLUMB_VARIANTS[variant]
    lc = LineCal(edges, "plumb", dist_free=kw["dist"], centre_free=kw["centre_free"], f0=F0)
    r = lc.solve(lc.x0(), loss="huber", f_scale=0.5)
    K, d, _, _ = lc.unpack(r.x)
    C = sandwich(r, lc.edge_groups_index())
    sd = np.sqrt(np.abs(np.diag(C)))
    out = dict(variant=variant, centre=[float(K[0, 2]), float(K[1, 2])], dist=[float(v) for v in d],
               dist_px=[d[0] / F0 ** 2, d[1] / F0 ** 4, d[2] / F0, d[3] / F0, d[4] / F0 ** 6],
               rms=float(np.sqrt(np.mean(r.fun ** 2))), n_edges=len(edges),
               sig={n: float(s) for n, s in zip(lc.names, sd)}, x=[float(v) for v in r.x], names=lc.names,
               edge_ids=[e["id"] for e in edges])
    cache[key] = out
    save_json(cache, PLUMB_CACHE)
    return out


def lens_at(pl, f, pp=None):
    """OpenCV K, dist for focal f with the plumb distortion fixed in pixel units (centre = pp or plumb centre)."""
    dp = np.asarray(pl["dist_px"], float)
    d = np.array([dp[0] * f ** 2, dp[1] * f ** 4, dp[2] * f, dp[3] * f, dp[4] * f ** 6])
    c = pl["centre"] if pp is None else pp
    return K_from(f, f, c[0], c[1]), d


class Und:
    """Undistortion (pixel units, f0) for one plumb variant."""

    def __init__(self, pl):
        self.pl = pl
        self.K, self.d = lens_at(pl, F0)

    def __call__(self, uv):
        uv = np.asarray(uv, float).reshape(-1, 2)
        n = undistort_points(uv, self.K, self.d, iters=40)
        return np.column_stack([F0 * n[:, 0] + self.K[0, 2], F0 * n[:, 1] + self.K[1, 2]])

    def jac(self, uv, h=0.25):
        """dU/dD (N x 2 x 2)."""
        uv = np.asarray(uv, float).reshape(-1, 2)
        U = self(uv)
        Ux = self(uv + [h, 0.0])
        Uy = self(uv + [0.0, h])
        return U, np.stack([(Ux - U) / h, (Uy - U) / h], axis=2)

    def dist_px(self, U):
        """undistorted px -> distorted px (forward model)."""
        from common import distort_normalized
        U = np.asarray(U, float).reshape(-1, 2)
        xn = (U[:, 0] - self.K[0, 2]) / F0
        yn = (U[:, 1] - self.K[1, 2]) / F0
        xd, yd = distort_normalized(xn, yn, self.d)
        return np.column_stack([F0 * xd + self.K[0, 2], F0 * yd + self.K[1, 2]])


# ------------------------------------------------------------------------------------------
# vertical vanishing point from vertical edges
# ------------------------------------------------------------------------------------------
def vertical_edges(edges=None, kinds=("cartZ",)):
    """Vertical edges: by default the cart posts (cartZ) - the sticker stacks run along the cart Z axis;
    kinds=("world_vertical",) gives the scene verticals, ("cartZ", "world_vertical") both."""
    edges = all_edges() if edges is None else edges
    return [e for e in edges if e["direction"] in kinds]


def fit_vp_lines(und, edges, v0=(930.0, 60.0), subsample=1):
    """VP of a set of edges (undistorted px). Residual = distance (converted to distorted px) of
    each undistorted edge point from the line (VP, edge centroid). Returns v, cov (scaled by the
    residual variance; cluster-robust over edges), rms, per-edge rms."""
    Ps, Js, idx = [], [], [0]
    for e in edges:
        P = e["points"][::subsample]
        U, J = und.jac(P)
        Ps.append(U)
        Js.append(J)
        idx.append(idx[-1] + len(P))
    Uall = np.vstack(Ps)
    Jall = np.vstack(Js)

    def res(v):
        out = []
        for k in range(len(Ps)):
            P = Uall[idx[k]:idx[k + 1]]
            c = P.mean(0)
            dvec = c - v
            nvec = np.array([-dvec[1], dvec[0]]) / np.hypot(*dvec)
            r = (P - c) @ nvec
            jt = np.einsum("nij,i->nj", Jall[idx[k]:idx[k + 1]], nvec)
            out.append(r / np.maximum(np.hypot(jt[:, 0], jt[:, 1]), 1e-9))
        return np.concatenate(out)

    r = least_squares(res, np.asarray(v0, float), x_scale=10.0)
    g = np.concatenate([np.full(idx[k + 1] - idx[k], k) for k in range(len(Ps))])
    try:
        C = sandwich(r, g)
    except Exception:  # noqa
        C = np.full((2, 2), np.nan)
    J = r.jac
    dof = max(len(r.fun) - 2, 1)
    Ccl = np.linalg.pinv(J.T @ J) * (2 * r.cost / dof)
    pe = [float(np.sqrt(np.mean(r.fun[idx[k]:idx[k + 1]] ** 2))) for k in range(len(Ps))]
    return dict(v=r.x, cov_cluster=C, cov_classic=Ccl, rms=float(np.sqrt(np.mean(r.fun ** 2))), per_edge=pe,
                ids=[e["id"] for e in edges])


# ------------------------------------------------------------------------------------------
# markers / columns
# ------------------------------------------------------------------------------------------
def markers():
    ms = load_markers()
    raw = {(m["cart"], m["id"]): m for m in load_json(f"{CACHE}/markers_final.json")}
    for m in ms:
        r = raw[(m["cart"], m["id"])]
        m["side_lines"] = r["side_lines"]
        m["side_valid"] = r["side_valid"]
        s = np.array(m["std"], float)
        m["sig_xy"] = np.where(np.isfinite(s), s / np.sqrt(2.0), 0.2)  # per-coordinate, per-frame scatter
    return ms


def mindex(ms):
    return {(m["cart"], m["id"]): m for m in ms}


def side_line_sigma(sl):
    """(sigma_offset_px, sigma_angle_rad, half_length_px) of a traced side line (distorted image).
    Neighbouring samples along a traced edge are correlated -> effective n = n_samples / 3;
    the offset sigma is floored at 0.05 px."""
    n = max(sl.get("n_samples") or 10, 5)
    rms = max(sl.get("rms_px") or 0.15, 0.08)
    L = 1.5 * n  # traced length (step 1.5 px)
    s_o = max(rms / np.sqrt(n / 3.0), 0.05)
    s_t = rms * np.sqrt(12.0) / (L * np.sqrt(n / 3.0))
    return s_o, s_t, L / 2.0


def columns(ms):
    """Sticker columns: for every cart / end / corner position of the top sticker a dict with
    per-level observations: ('pt', uv, sigma_xy) or ('line', point, direction, s_o, s_t, half) or None."""
    mi = mindex(ms)
    cols = []
    for cart in (80, 310):
        for end, ids in END_IDS.items():
            top = mi[(cart, ids["top"])]
            for j in range(4):
                XY = top["corners_3d"][j][:2]
                col = dict(cart=cart, end=end, top_corner=j, XY=XY.tolist(), obs={}, corner_of={})
                for lev in LEVELS:
                    m = mi.get((cart, ids[lev]))
                    if m is None:
                        col["obs"][lev] = None
                        continue
                    k = int(np.argmin(np.linalg.norm(m["corners_3d"][:, :2] - XY, axis=1)))
                    assert np.linalg.norm(m["corners_3d"][k, :2] - XY) < 1e-6
                    col["corner_of"][lev] = k
                    if m["valid"][k]:
                        col["obs"][lev] = ("pt", m["corners_px"][k].copy(), float(m["sig_xy"][k]))
                    else:
                        cand = []
                        for s in ((k - 1) % 4, k):
                            sl = m["side_lines"][s]
                            if sl["valid"] and sl["point"] is not None:
                                s_o, s_t, half = side_line_sigma(sl)
                                cand.append(("line", np.array(sl["point"], float), np.array(sl["direction"], float) /
                                             np.linalg.norm(sl["direction"]), s_o, s_t, half, s))
                        col["obs"][lev] = cand if cand else None
                cols.append(col)
    return cols


# ------------------------------------------------------------------------------------------
# small geometry helpers
# ------------------------------------------------------------------------------------------
def tls_line(P, w=None):
    """TLS line through points (N x 2) -> centroid c, unit direction d, unit normal n, residuals."""
    P = np.asarray(P, float)
    w = np.ones(len(P)) if w is None else np.asarray(w, float)
    c = (P * w[:, None]).sum(0) / w.sum()
    Q = (P - c) * np.sqrt(w)[:, None]
    _, _, vt = np.linalg.svd(Q, full_matrices=False)
    d = vt[0]
    n = np.array([-d[1], d[0]])
    return c, d, n, (P - c) @ n


def intersect_lines(c1, d1, c2, d2):
    """Intersection of p = c1 + s d1 and p = c2 + t d2."""
    A = np.column_stack([d1, -d2])
    st = np.linalg.solve(A, c2 - c1)
    return c1 + st[0] * d1


def proj1d_fit(Z, t):
    """t = (a Z + b) / (c Z + 1) through >= 3 (Z, t) pairs (least squares, algebraic). Returns (a, b, c)."""
    Z, t = np.asarray(Z, float), np.asarray(t, float)
    A = np.column_stack([Z, np.ones_like(Z), -Z * t])
    return np.linalg.lstsq(A, t, rcond=None)[0]


def proj1d_inv(abc, t):
    a, b, c = abc
    return (t - b) / (a - c * t)


def cross_ratio(a, b, c, d):
    return ((c - a) * (d - b)) / ((c - b) * (d - a))


def aic_gauss(rss, n, k):
    return n * np.log(rss / n) + 2 * k


def bic_gauss(rss, n, k):
    return n * np.log(rss / n) + k * np.log(n)
