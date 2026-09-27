"""Synthetic round-trip / sensitivity test of the orchestrator pipeline (shared helpers).

The synthetic data copy the REAL observation geometry exactly:
  * stickers: the same 20 stickers and the same valid corners (62), 3D corners from the drawing;
  * edges: every verified straight edge of work/cache/edges_{cart310,cart80,scene}.json (same filter as
    linedata.load_edges) plus the traced sides of partly hidden stickers (markerdata.marker_side_edges);
    each real edge is undistorted with the TRUE lens and replaced by its ideal line
      - exact model line (cart frame, drawing)  -> projection of the 3D line with the true pose,
      - cart-axis / world-vertical direction    -> line through the true vanishing point and the edge centroid,
      - other straight edges                    -> its own TLS line,
    the real points are dropped onto that line (same extent / spacing) and distorted with the true lens;
  * cart poses: pose-only fit of the drawing geometry to the real corners with the true lens held fixed.
Noise models are built from the real residuals (see real_edge_stats / scenario generation).

Estimators = the orchestrator's code paths (calib.Problem / lineselfcal.LineCal / combined.Combined), same
settings (loss, f_scale, subsampling, reweighting, multi-start in f for the sticker fits).
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from scipy.optimize import least_squares

from calib import Model, Problem, covariance
from combined import AXIS, Combined, model_line_3d
from common import CACHE, H, W, K_from, distort_normalized, load_json, project, rodrigues, undistort_points
from evaltools import grid, mapping_displacement, rays, region_defs
from linedata import load_edges
from lineselfcal import LineCal, classic_cov, sandwich
from markerdata import load_markers, marker_side_edges

EDGE_FILES = [f"{CACHE}/edges_{s}.json" for s in ("cart310", "cart80", "scene")]
F0 = 1300.0
CEN = np.array([(W - 1) / 2.0, (H - 1) / 2.0])
UV, UV_SHAPE = grid(40)
REGIONS = region_defs()
REG_MASK = {k: fn(UV) for k, fn in REGIONS.items()}

MARKER_SPECS = {
    "k1_ppfix": dict(f="single", pp="fixed", dist=["k1"]),
    "k1k2_ppfix": dict(f="single", pp="fixed", dist=["k1", "k2"]),
    "k1_ppfree": dict(f="single", pp="free", dist=["k1"]),
    "k1k2_ppfree": dict(f="single", pp="free", dist=["k1", "k2"]),
}
COMBINED_SPEC = dict(f="single", pp="free", dist=["k1", "k2"])


# ----------------------------------------------------------------------------------------------
# real data
# ----------------------------------------------------------------------------------------------

def load_real_edges():
    """Same selection as linedata.load_edges(verified_only=True) on the three canonical edge files.
    (linedata's default glob `edges_*.json` now also matches *_prereview / *_check / *_diag / *_raw files.)"""
    out = []
    for fn in EDGE_FILES:
        out += load_edges(verified_only=True, source=fn)
    return out


def edge_review_status():
    st = {}
    for fn in EDGE_FILES:
        d = load_json(fn)
        st[os.path.basename(fn)] = "reviewed" if "review" in d else "not reviewed"
    return st


def load_real():
    M = load_markers()
    E = load_real_edges()
    S = marker_side_edges(M, only_partial=True)
    return M, E, S


def side_owner(e):
    """marker_<cart>_<id>_side<j> -> (cart, id)"""
    p = e["id"].split("_")
    return int(p[1]), int(p[2])


def edge_block(e, carts=(80, 310)):
    """Block of an edge in the Combined estimator (L / V / S) and the LineCal VP group."""
    if model_line_3d(e.get("model_line")) is not None and e.get("cart") in carts:
        return "L"
    if e.get("direction") in AXIS and e.get("cart") in carts:
        return "V"
    if e.get("direction") == "world_vertical":
        return "V"
    return "S"


# ----------------------------------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------------------------------

def und_px(P, K, d):
    n = undistort_points(P, K, d, iters=40)
    return np.column_stack([K[0, 0] * n[:, 0] + K[0, 2], K[1, 1] * n[:, 1] + K[1, 2]])


def dist_px(U, K, d):
    xn = (U[:, 0] - K[0, 2]) / K[0, 0]
    yn = (U[:, 1] - K[1, 2]) / K[1, 1]
    xd, yd = distort_normalized(xn, yn, d)
    return np.column_stack([K[0, 0] * xd + K[0, 2], K[1, 1] * yd + K[1, 2]])


def tls_line(P):
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    n = vt[1]
    return np.r_[n, -n @ c]


def foot_on_line(U, l):
    n = l[:2] / np.hypot(l[0], l[1])
    c = l[2] / np.hypot(l[0], l[1])
    dd = U @ n + c
    return U - dd[:, None] * n[None]


def curve_normals(P):
    t = np.gradient(P, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    return np.column_stack([-t[:, 1], t[:, 0]])


def fit_poses(K, d, M, P0):
    """Pose-only fit (drawing geometry) of both carts to the real valid corners with a fixed lens."""
    out = {}
    for c in (80, 310):
        X = np.vstack([m["corners_3d"][m["valid"]] for m in M if m["cart"] == c])
        uv = np.vstack([m["corners_px"][m["valid"]] for m in M if m["cart"] == c])

        def res(p):
            return (project(X, K, d, rvec=p[:3], tvec=p[3:]) - uv).ravel()

        best = None
        for s in (1.0, 0.9, 1.1):
            p0 = np.array(P0[c], float).copy()
            p0[3:] *= s
            r = least_squares(res, p0, method="lm", x_scale="jac", max_nfev=5000)
            if best is None or r.cost < best.cost:
                best = r
        out[c] = best.x
    return out


def world_vertical_dir(E, K, d):
    """Least-squares vanishing direction of the world-vertical edges (undistorted with K, d)."""
    Ls = []
    for e in E:
        if e.get("direction") != "world_vertical":
            continue
        U = und_px(e["points"], K, d)
        l = tls_line(U)
        Ls.append(l / np.hypot(l[0], l[1]))
    A = np.array(Ls)
    _, _, vt = np.linalg.svd(A)
    v = vt[-1]
    dvec = np.linalg.solve(K, v)
    return dvec / np.linalg.norm(dvec)


# ----------------------------------------------------------------------------------------------
# truth
# ----------------------------------------------------------------------------------------------

class Truth:
    def __init__(self, name, f, cx, cy, k1, k2, M, E, P0):
        self.name = name
        self.K = K_from(f, f, cx, cy)
        self.d = np.array([k1, k2, 0.0, 0.0, 0.0])
        self.poses = fit_poses(self.K, self.d, M, P0)
        self.wv = world_vertical_dir(E, self.K, self.d)
        cz = np.mean([rodrigues(self.poses[c][:3])[:, 2] for c in (80, 310)], axis=0)
        if self.wv @ cz < 0:
            self.wv = -self.wv
        self.params = dict(f=f, cx=cx, cy=cy, k1=k1, k2=k2)

    def px_units(self):
        f = self.K[0, 0]
        return np.array([self.d[0] / f ** 2, self.d[1] / f ** 4])

    def describe(self):
        p = self.params
        return f"f={p['f']:.0f}, pp=({p['cx']:.0f},{p['cy']:.0f}), k1={p['k1']:.3f}, k2={p['k2']:.3f}"


# ----------------------------------------------------------------------------------------------
# real edge residual statistics (noise / bow / direction deviations)
# ----------------------------------------------------------------------------------------------

def real_edge_stats(E, S, Kref, dref, rots_ref, wv_ref):
    """Per edge: white-noise sigma + lag-1 correlation (high-frequency part), low-frequency bow profile
    (cubic in the normalised arc parameter, constant/linear part removed), direction deviation w.r.t. the
    reference vanishing point. Residuals are distances in the distorted image (normal of the fitted line
    mapped through the local Jacobian)."""
    stats = {}
    for e in E + S:
        P = e["points"]
        U = und_px(P, Kref, dref)
        # local jacobian scaling
        h = 0.5
        Ux = und_px(P + [h, 0], Kref, dref)
        Uy = und_px(P + [0, h], Kref, dref)
        l = tls_line(U)
        n = l[:2]
        r_u = U @ n + l[2]
        jt = np.column_stack([(Ux - U) / h @ n, (Uy - U) / h @ n])
        scale = np.maximum(np.hypot(jt[:, 0], jt[:, 1]), 1e-9)
        r = r_u / scale
        tdir = np.array([-n[1], n[0]])
        s = (U - U.mean(0)) @ tdir
        L = s.max() - s.min()
        t = 2 * (s - s.min()) / max(L, 1e-9) - 1
        deg = 3 if len(P) >= 20 else 2
        c = np.polyfit(t, r, deg)
        low = np.polyval(c, t)
        # remove the constant + linear part of the bow (absorbed by any line fit)
        cl = np.polyfit(t, low, 1)
        bow = low - np.polyval(cl, t)
        hf = r - low
        sig = float(np.std(hf))
        rho = float(np.corrcoef(hf[:-1], hf[1:])[0, 1]) if len(hf) > 5 else 0.0
        spacing = float(np.median(np.hypot(*np.diff(P, axis=0).T)))
        # direction deviation w.r.t. the reference VP (only for VP-group edges)
        theta = 0.0
        blk = edge_block(e)
        if blk == "V" and rots_ref is not None:
            if e.get("direction") == "world_vertical":
                dv = wv_ref
            else:
                dv = rots_ref[int(e["cart"])][:, AXIS[e["direction"]]]
            v = Kref @ dv
            cU = U.mean(0)
            lv = np.cross(v, np.r_[cU, 1.0])
            lv = lv / np.hypot(lv[0], lv[1])
            # angle between TLS line and VP line (in the undistorted image)
            a1 = np.arctan2(-l[0], l[1])
            a2 = np.arctan2(-lv[0], lv[1])
            theta = float((a1 - a2 + np.pi / 2) % np.pi - np.pi / 2)
        stats[e["id"]] = dict(sigma=sig, rho=float(np.clip(rho, 0.0, 0.95)), bow_coef=np.polyfit(t, bow, deg).tolist(),
                              bow_rms=float(np.sqrt(np.mean(bow ** 2))), rms=float(np.sqrt(np.mean(r ** 2))), length_px=float(L),
                              spacing_px=spacing, theta_rad=theta, block=blk, n=len(P))
    return stats


# ----------------------------------------------------------------------------------------------
# synthetic observations
# ----------------------------------------------------------------------------------------------

def ideal_edges(truth, E, S, side_offsets=None, extra_line_offsets=None):
    """Noise-free synthetic edges (list of edge dicts with replaced points) for the true lens.
    side_offsets: {(cart,id): 3-vector mm} 3D shift of a sticker (moves its traced sides too)."""
    K, d = truth.K, truth.d
    Rs = {c: rodrigues(truth.poses[c][:3]) for c in (80, 310)}
    out = []
    for e in E + S:
        P = e["points"]
        U = und_px(P, K, d)
        ml = model_line_3d(e.get("model_line"))
        blk = edge_block(e)
        if blk == "L":
            A, D = ml
            A = A.copy()
            if e["id"].startswith("marker_") and side_offsets is not None:
                A = A + side_offsets.get(side_owner(e), 0.0)
            c = int(e["cart"])
            R, t = Rs[c], truth.poses[c][3:]
            a = K @ (R @ A + t)
            b = K @ (R @ D)
            l = np.cross(a, b)
        elif blk == "V":
            if e.get("direction") == "world_vertical":
                v = K @ truth.wv
            else:
                v = K @ Rs[int(e["cart"])][:, AXIS[e["direction"]]]
            l = np.cross(v, np.r_[U.mean(0), 1.0])
        else:
            l = tls_line(U)
        Ui = foot_on_line(U, l)
        Pi = dist_px(Ui, K, d)
        out.append(dict(e, points=Pi))
    return out


def ar1(n, rho, rng):
    w = rng.standard_normal(n)
    if rho <= 0 or n < 2:
        return w
    x = np.empty(n)
    x[0] = w[0]
    a = np.sqrt(1 - rho ** 2)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + a * w[i]
    return x


def noisy_edges(ideal, stats, rng, bows=False, directions=False, noise_scale=1.0):
    out = []
    for e in ideal:
        P = e["points"]
        st = stats[e["id"]]
        nrm = curve_normals(P)
        off = st["sigma"] * noise_scale * ar1(len(P), st["rho"], rng)
        tvec = np.array([nrm[:, 1], -nrm[:, 0]]).T
        s = (P - P.mean(0)) @ np.median(tvec, axis=0)
        L = s.max() - s.min()
        t = 2 * (s - s.min()) / max(L, 1e-9) - 1
        if bows:
            off = off + rng.choice([-1.0, 1.0]) * np.polyval(st["bow_coef"], t)
        if directions and st["block"] == "V":
            off = off + rng.choice([-1.0, 1.0]) * st["theta_rad"] * (s - s.mean())
        out.append(dict(e, points=P + off[:, None] * nrm))
    return out


def sticker_obs(truth, M, offsets=None, rng=None, corner_sigma=None):
    """Synthetic corner observations in the to_points format (X = drawing corners, uv = synthetic).
    offsets: {(cart,id): 3-vector mm} unknown true 3D displacement of a sticker (not known to the estimator)."""
    pts = []
    for mi, m in enumerate(M):
        X = m["corners_3d"]
        Xt = X + (offsets.get((m["cart"], m["id"]), 0.0) if offsets is not None else 0.0)
        c = m["cart"]
        uv = project(Xt, truth.K, truth.d, rvec=truth.poses[c][:3], tvec=truth.poses[c][3:])
        if rng is not None:
            sg = corner_sigma[mi]
            uv = uv + rng.standard_normal((4, 2)) * sg[:, None]
        for j in range(4):
            if m["valid"][j]:
                pts.append(dict(cart=c, id=m["id"], mi=mi, corner=j, X=X[j], uv=uv[j]))
    return pts


def corner_sigmas(M):
    """Per corner, per coordinate sigma of the 7-frame mean: radial per-frame std / sqrt(2) / sqrt(7)."""
    out = []
    med = np.nanmedian(np.concatenate([m["std"][m["valid"]] for m in M]))
    for m in M:
        s = np.array(m["std"], float)
        s = np.where(np.isfinite(s), s, med)
        out.append(s / np.sqrt(14.0))
    return out


def geometry_offsets(M, rng, sig_sticker, sig_row):
    """Random unknown 3D placement: per-sticker isotropic (mm) + per cart-row height (mm)."""
    rows = {}
    off = {}
    for m in M:
        key = (m["cart"], m["row"])
        if key not in rows:
            rows[key] = rng.standard_normal() * sig_row
        off[(m["cart"], m["id"])] = rng.standard_normal(3) * sig_sticker + np.array([0, 0, rows[key]])
    return off


def diagnosed_offsets(M):
    """Deterministic what-if: the end-board heights of the diagnostic fit (geometry_diagnosis.md sec. 3):
    top stickers raised by 80/X0 +64, 80/X1600 +71, 310/X0 +61, 310/X1600 +81 mm."""
    dz = {(80, 0): 64.0, (80, 1): 71.0, (310, 0): 61.0, (310, 1): 81.0}
    off = {}
    for m in M:
        if m["row"] == "top":
            end = int(m["corners_3d"][:, 0].mean() > 800)
            off[(m["cart"], m["id"])] = np.array([0, 0, dz[(m["cart"], end)]])
    return off


# ----------------------------------------------------------------------------------------------
# estimators (orchestrator code paths)
# ----------------------------------------------------------------------------------------------

def cluster_sandwich_pts(r, groups):
    J = r.jac
    e = r.fun
    JTJi = np.linalg.pinv(J.T @ J)
    meat = np.zeros_like(JTJi)
    g2 = np.repeat(groups, 2)
    for g in np.unique(groups):
        s = g2 == g
        v = J[s].T @ e[s]
        meat += np.outer(v, v)
    G = len(np.unique(groups))
    return JTJi @ meat @ JTJi * G / max(G - 1, 1)


def fit_markers(spec, pts, K0, d0, P0, carts=(80, 310), multistart=True):
    """= 20_markers_ba.fit (3 starts in f, lowest cost)."""
    m = Model(spec)
    pr = Problem(m, pts, carts=list(carts))
    x0 = np.r_[m.pack(K0, d0), np.ravel([P0[c] for c in carts])]
    best = None
    for fs in ((1.0, 0.85, 1.15) if multistart else (1.0,)):
        xs = x0.copy()
        xs[0] *= fs
        if m.f == "fxfy":
            xs[1] *= fs
        r = pr.solve(xs)
        if best is None or r.cost < best.cost:
            best = r
    return m, pr, best


def marker_estimate(spec, pts, K0, d0, P0):
    m, pr, r = fit_markers(spec, pts, K0, d0, P0)
    C, s2, dof = covariance(r)
    groups = np.array([p["mi"] for p in pts])
    Cs = cluster_sandwich_pts(r, groups)
    res = pr.predict(r.x) - pr.uv
    K, d = m.unpack(r.x[: m.n])
    return dict(names=m.names, x=r.x[: m.n], C=C[: m.n, : m.n], Cs=Cs[: m.n, : m.n], K=K, d=d[:5],
                rms=float(np.sqrt(np.mean(np.sum(res ** 2, 1)))), unpack=m.unpack, full_x=r.x)


def plumb_estimate(edges, centre_free=False, x0=None):
    lc = LineCal(edges, "plumb", dist_free=("k1", "k2"), centre_free=centre_free, f0=F0)
    x0 = lc.x0() if x0 is None else x0
    r = lc.solve(x0, loss="huber", f_scale=0.5)
    Cs = sandwich(r, lc.edge_groups_index())
    Cc = classic_cov(r)
    return lc, r, Cc[: lc.n_intr, : lc.n_intr], Cs[: lc.n_intr, : lc.n_intr]


def vp_estimate(edges, dist_px_units, K0, rots0):
    lc = LineCal(edges, "vp", pp_free=True)

    def unpack_fixed(x, _orig=lc.unpack):
        K, d, rots, wv = _orig(x)
        f = K[0, 0]
        d = np.array([dist_px_units[0] * f ** 2, dist_px_units[1] * f ** 4, 0, 0, 0])
        return K, d, rots, wv

    lc.unpack = unpack_fixed
    x0 = lc.x0(K0, np.zeros(5), rots0)
    r = lc.solve(x0, loss="huber", f_scale=0.5)
    Cs = sandwich(r, lc.edge_groups_index())
    Cc = classic_cov(r)
    return lc, r, Cc[: lc.n_intr, : lc.n_intr], Cs[: lc.n_intr, : lc.n_intr]


def joint_estimate(edges, K0, d0, rots0):
    lc = LineCal(edges, "joint", dist_free=("k1", "k2"), pp_free=True)
    x0 = lc.x0(K0, d0, rots0)
    r = lc.solve(x0, loss="huber", f_scale=0.5)
    Cs = sandwich(r, lc.edge_groups_index())
    Cc = classic_cov(r)
    return lc, r, Cc[: lc.n_intr, : lc.n_intr], Cs[: lc.n_intr, : lc.n_intr]


def combined_estimate(pts, edges, K0, d0, P0, spec=COMBINED_SPEC, sig=None):
    cb = Combined(pts, edges, spec, sig=sig)
    x0 = cb.x0(K0, d0, P0)
    r = cb.solve(x0)
    J = r.jac
    dof = max(len(r.fun) - len(r.x), 1)
    C = np.linalg.pinv(J.T @ J) * (2 * r.cost / dof)
    K, d, _, _ = cb.unpack(r.x)
    return cb, r, C[: cb.ni, : cb.ni], K, d[:5]


# ----------------------------------------------------------------------------------------------
# mapping evaluation
# ----------------------------------------------------------------------------------------------

def valid_mask(K, d, uv=UV, tol=0.05):
    """Pixels whose viewing ray is well defined (undistortion converged and the model is not folded)."""
    r = rays(uv, K, d)
    back = project(r, K, d)
    return (np.linalg.norm(back - uv, axis=1) < tol) & (r[:, 2] > 0.05)


def disp_field(K1, d1, K2, d2, uv=UV, compensate=True):
    """evaltools.mapping_displacement restricted to pixels with a valid ray in model 1 (NaN elsewhere)."""
    m = valid_mask(K1, d1, uv)
    D = np.full((len(uv), 2), np.nan)
    if m.sum() > 10:
        D[m] = mapping_displacement(K1, d1, K2, d2, uv[m], compensate=compensate)
    return D


def region_summary(per_point):
    """per_point: N values (NaN allowed) -> {region: median, max}."""
    out = {}
    for k, m in REG_MASK.items():
        v = per_point[m]
        v = v[np.isfinite(v)]
        out[k] = dict(median=float(np.median(v)) if len(v) else None, max=float(np.max(v)) if len(v) else None,
                      n_valid=int(len(v)), n=int(m.sum()))
    v = per_point[np.isfinite(per_point)]
    out["whole_image"] = dict(median=float(np.median(v)) if len(v) else None, max=float(np.max(v)) if len(v) else None,
                              n_valid=int(len(v)), n=int(len(per_point)))
    return out


def claimed_mapping(K, d, C, unpack_fn, x, rng, ns=30):
    """1-sigma mapping uncertainty claimed by a covariance: sample intrinsics ~ N(x, C), displacement of each
    sample vs the estimate (rotation compensated); per-point RMS over the samples -> per-region medians."""
    try:
        L = np.linalg.cholesky(C + 1e-18 * np.eye(len(C)))
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(C)
        L = V @ np.diag(np.sqrt(np.clip(w, 0, None)))
    m = valid_mask(K, d)
    if m.sum() < 10:
        return None
    Ds = []
    for _ in range(ns):
        xs = x + L @ rng.standard_normal(len(x))
        Ks, ds = unpack_fn(xs)
        Ds.append(mapping_displacement(K, d, Ks, np.asarray(ds)[:5], UV[m], compensate=True))
    Ds = np.array(Ds)
    pp = np.full(len(UV), np.nan)
    pp[m] = np.sqrt(np.mean(np.sum(Ds ** 2, axis=2), axis=0))
    return pp
