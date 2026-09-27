"""Independent line-based calibration library (plumb-line, vanishing points, joint line self-calibration).

Written independently of lineselfcal.py / combined.py (only common.py helpers for paths / JSON are used).

Coordinates
-----------
* Distorted image: pixel coordinates of the traced edge points (OpenCV convention).
* Distortion is expressed in PIXEL UNITS around a distortion centre c, with a conditioning scale S = 1000 px:
      q = (p - c) / S                     (scaled, centred coordinates)
  'poly'  (Brown / OpenCV radial [+ tangential]), forward map undistorted -> distorted:
      q_d = q_u (1 + a1 r^2 + a2 r^4 + a3 r^6) + [2 t1 x y + t2 (r^2 + 2 x^2),  t1 (r^2 + 2 y^2) + 2 t2 x y]
      with r = |q_u|.  OpenCV coefficients for a camera with focal length f (pp = c):
      k1 = a1 f^2 / S^2, k2 = a2 f^4 / S^4, k3 = a3 f^6 / S^6, p1 = t1 f / S, p2 = t2 f / S
      i.e. straightness alone determines only k_i / f^(2i) (and p_i / f).
  'div'   (division model), backward map distorted -> undistorted:
      q_u = q_d / (1 + l1 r_d^2 + l2 r_d^4)
* The undistorted pixel image  U = c + S q_u  is an ideal pinhole image with the TRUE focal length f and
  principal point c (if the distortion centre is the principal point), independent of any f0.

Residuals
---------
Distances are always measured in the DISTORTED image (first-order / Sampson distance):
      r_i = S * (n . q_u,i - rho) / |J_i^T n|,   J_i = d q_u / d q_d  at point i,
where (n, rho) is the edge's line in the undistorted plane (fitted per edge by weighted TLS - variable
projection - or constrained to pass through a vanishing point).  Measuring in the undistorted image would
make the noise grow with the undistortion and bias the fit towards weaker distortion.
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.optimize import least_squares

from common import CACHE, RESULTS, H, W, save_json, load_json  # noqa: F401

S = 1000.0
FOLD_MIN_SLOPE = 0.2  # barrier: radial map slope d r_d / d r_u (div: d r_u / d r_d) must stay > this up to the far corner
C0 = np.array([(W - 1) / 2.0, (H - 1) / 2.0])
EDGE_FILES = [f"{CACHE}/edges_cart310.json", f"{CACHE}/edges_cart80.json", f"{CACHE}/edges_scene.json"]

# ----------------------------------------------------------------------------------------------------------
# edge data
# ----------------------------------------------------------------------------------------------------------
# edges of one physical object (not independent): same bootstrap cluster
SAME_OBJECT = [
    ("scene_blue_left_upper", "scene_blue_left_lower"),
    ("scene_pole_top", "scene_pole_bottom"),
    ("scene_tapeV_left", "scene_tapeV_right"),
    ("scene_jointB_V_left", "scene_jointB_V_right"),
    ("scene_tapeH_top", "scene_tapeH_bottom", "scene_blueH_top"),
    ("scene_blueL_top", "scene_blueL_bottom", "cart310_floor_tape_blue_upper", "cart310_floor_tape_blue_lower"),
    ("scene_leg_left", "scene_leg_right"),
    ("scene_rpost_rod", "scene_rpost_rodR"),
    ("cart310_board0_inner_top", "cart310_board0_inner_bottom"),
    ("cart80_x_back_rail_out", "scene_wall_railA"),
]
# floor lines: aisle direction (runs along the aisle, image-vertical) / cross direction (across the aisle)
FLOOR_AISLE = {"scene_tapeV_left", "scene_tapeV_right", "scene_jointB_V_left", "scene_jointB_V_right"}
FLOOR_CROSS = {"scene_tapeH_top", "scene_tapeH_bottom", "scene_blueH_top", "scene_blueL_top", "scene_blueL_bottom",
               "cart310_floor_tape_blue_upper", "cart310_floor_tape_blue_lower"}
FLOOR_JOINT = {"scene_joint450"}  # saw-cut floor joint across the aisle: parallel to the blue line only by assumption


def _truthy(v):
    return v is True or (isinstance(v, str) and v.lower() == "true") or v == 1


def file_status():
    """Which edge files carry a top-level review block (i.e. are final)."""
    out = {}
    for fn in EDGE_FILES:
        d = load_json(fn)
        out[os.path.basename(fn)] = "review" in d
    return out


def load_edges_indep(min_len=40.0, drop_duplicates=True, verbose=False):
    """Own loader: straight_3d edges, verified != 'no', geometric duplicates removed (the longer one is kept).

    Returns list of dicts: id, src, cart (int|None), direction, points (N x 2), length, verified, exact,
    pair_of, cluster (str), group (VP group name or None).
    """
    raw = []
    for fn in EDGE_FILES:
        d = load_json(fn)
        for e in d["edges"]:
            if not _truthy(e.get("straight_3d", False)):
                continue
            if str(e.get("verified", "yes")).lower() == "no":
                continue
            P = np.asarray(e["points"], float)
            if len(P) < 6:
                continue
            L = float(np.sum(np.hypot(*np.diff(P, axis=0).T)))
            if L < min_len:
                continue
            cart = e.get("cart")
            try:
                cart = int(cart) if cart not in (None, "None", "") else None
            except (TypeError, ValueError):
                cart = None
            ml = e.get("model_line") if isinstance(e.get("model_line"), dict) else None
            raw.append(dict(id=e["id"], src=os.path.basename(fn), cart=cart, direction=e.get("direction", "unknown"),
                            points=P, length=L, verified=str(e.get("verified", "yes")),
                            exact=bool(ml and _truthy(ml.get("exact", False))), pair_of=e.get("pair_of"),
                            what=e.get("what", ""), crop=e.get("crop")))
    # geometric duplicates (same image edge traced twice, e.g. in two region files)
    dropped = []
    if drop_duplicates:
        keep = [True] * len(raw)
        order = np.argsort([-e["length"] for e in raw])
        for a in range(len(order)):
            i = order[a]
            if not keep[i]:
                continue
            for b in range(a + 1, len(order)):
                j = order[b]
                if not keep[j]:
                    continue
                Pi, Pj = raw[i]["points"], raw[j]["points"]
                # quick bbox test
                if (Pj[:, 0].min() > Pi[:, 0].max() + 5 or Pj[:, 0].max() < Pi[:, 0].min() - 5 or
                        Pj[:, 1].min() > Pi[:, 1].max() + 5 or Pj[:, 1].max() < Pi[:, 1].min() - 5):
                    continue
                dmin = np.min(np.hypot(Pj[:, None, 0] - Pi[None, :, 0], Pj[:, None, 1] - Pi[None, :, 1]), axis=1)
                if np.mean(dmin < 1.5) > 0.5:
                    keep[j] = False
                    dropped.append((raw[j]["id"], raw[i]["id"], float(np.median(dmin))))
        raw = [e for e, k in zip(raw, keep) if k]
    if verbose and dropped:
        print("duplicates dropped:", dropped)
    # clusters
    parent = {e["id"]: e["id"] for e in raw}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        if a in parent and b in parent:
            parent[find(a)] = find(b)

    for e in raw:
        if e["pair_of"]:
            union(e["id"], e["pair_of"])
    for grp in SAME_OBJECT:
        for a in grp[1:]:
            union(grp[0], a)
    for e in raw:
        e["cluster"] = find(e["id"])
        e["group"] = vp_group(e)
    return raw, dropped


def vp_group(e):
    """Vanishing-point group of an edge (None = straightness only)."""
    i, d, c = e["id"], e["direction"], e["cart"]
    if i in FLOOR_AISLE:
        return "FA"
    if i in FLOOR_CROSS:
        return "FC"
    if i in FLOOR_JOINT:
        return "FJ"
    if d == "world_vertical":
        return "WV"
    if c in (80, 310) and d in ("cartX", "cartY", "cartZ"):
        ax = d[-1]
        if "board" in i:  # end-board edges: board frame, not necessarily the cart frame
            return f"{c}{ax}b"
        return f"{c}{ax}"
    return None


def region_of(e):
    return "scene" if e["src"] == "edges_scene.json" else ("cart310" if e["src"] == "edges_cart310.json" else "cart80")


# ----------------------------------------------------------------------------------------------------------
# distortion models (pixel units, scaled by S, around centre c)
# ----------------------------------------------------------------------------------------------------------
class Dist:
    """kind 'poly': coeffs = [a1..an] (+ [t1, t2] if tang);  kind 'div': coeffs = [l1..ln]."""

    def __init__(self, kind="poly", coeffs=(0.0,), tang=None, centre=C0):
        self.kind = kind
        self.a = np.asarray(coeffs, float)
        self.t = None if tang is None else np.asarray(tang, float)
        self.c = np.asarray(centre, float)

    def copy(self):
        return Dist(self.kind, self.a.copy(), None if self.t is None else self.t.copy(), self.c.copy())

    # forward (poly only): q_u -> q_d and Jacobian dq_d/dq_u
    def fwd(self, qu):
        x, y = qu[:, 0], qu[:, 1]
        r2 = x * x + y * y
        R = np.ones_like(r2)
        dR = np.zeros_like(r2)
        pw = np.ones_like(r2)
        for i, ai in enumerate(self.a):
            dR = dR + (i + 1) * ai * pw
            pw = pw * r2
            R = R + ai * pw
        xd = x * R
        yd = y * R
        j11 = R + 2 * x * x * dR
        j22 = R + 2 * y * y * dR
        j12 = 2 * x * y * dR
        j21 = j12.copy()
        if self.t is not None:
            t1, t2 = self.t
            xd = xd + 2 * t1 * x * y + t2 * (r2 + 2 * x * x)
            yd = yd + t1 * (r2 + 2 * y * y) + 2 * t2 * x * y
            j11 = j11 + 2 * t1 * y + 6 * t2 * x
            j12 = j12 + 2 * t1 * x + 2 * t2 * y
            j21 = j21 + 2 * t1 * x + 2 * t2 * y
            j22 = j22 + 6 * t1 * y + 2 * t2 * x
        return np.column_stack([xd, yd]), (j11, j12, j21, j22)

    def distort_px(self, U):
        """undistorted px -> distorted px"""
        U = np.asarray(U, float).reshape(-1, 2)
        if self.kind == "poly":
            qd, _ = self.fwd((U - self.c) / S)
            return self.c + S * qd
        # division: invert numerically
        qu = (U - self.c) / S
        qd = qu.copy()
        for _ in range(60):
            qq, J = self._div_back(qd)
            ex = qq - qu
            j11, j12, j21, j22 = J
            det = j11 * j22 - j12 * j21
            dx = (j22 * ex[:, 0] - j12 * ex[:, 1]) / det
            dy = (-j21 * ex[:, 0] + j11 * ex[:, 1]) / det
            qd = qd - np.column_stack([dx, dy])
            if np.max(np.abs(dx) + np.abs(dy)) < 1e-14:
                break
        return self.c + S * qd

    def _div_back(self, qd):
        x, y = qd[:, 0], qd[:, 1]
        r2 = x * x + y * y
        L = np.ones_like(r2)
        dL = np.zeros_like(r2)
        pw = np.ones_like(r2)
        for i, li in enumerate(self.a):
            dL = dL + (i + 1) * li * pw
            pw = pw * r2
            L = L + li * pw
        qu = qd / L[:, None]
        # d(qd/L)/dqd = I/L - qd (dL/dqd)^T / L^2, dL/dqd = 2 dL/dr2 qd
        g = 2 * dL / L ** 2
        j11 = 1 / L - g * x * x
        j22 = 1 / L - g * y * y
        j12 = -g * x * y
        return qu, (j11, j12, j12.copy(), j22)

    def undistort(self, P, iters=30):
        """distorted px (N x 2) -> undistorted px U (N x 2), Jacobian JU = dU/dP as tuple (j11,j12,j21,j22), ok flag."""
        qd = (np.asarray(P, float) - self.c) / S
        if self.kind == "div":
            qu, J = self._div_back(qd)
            det = J[0] * J[3] - J[1] * J[2]
            self.valid = det > 0
            return self.c + S * qu, J, bool(np.all(det > 0))
        # Newton inversion of the forward map
        qu = qd.copy()
        if len(self.a):
            r2 = np.sum(qd * qd, 1)
            qu = qd / (1 + self.a[0] * r2)[:, None]
        for it in range(iters):
            qq, (j11, j12, j21, j22) = self.fwd(qu)
            ex = qq - qd
            det = j11 * j22 - j12 * j21
            det = np.where(np.abs(det) < 1e-9, 1e-9, det)
            dx = (j22 * ex[:, 0] - j12 * ex[:, 1]) / det
            dy = (-j21 * ex[:, 0] + j11 * ex[:, 1]) / det
            qu = qu - np.column_stack([dx, dy])
            if np.max(np.abs(dx) + np.abs(dy)) < 1e-13:
                break
        qq, (j11, j12, j21, j22) = self.fwd(qu)
        det = j11 * j22 - j12 * j21
        self.valid = (det > 1e-6) & (np.max(np.abs(qq - qd), axis=1) < 1e-8)
        ok = bool(np.all(self.valid))
        # inverse Jacobian
        detc = np.where(np.abs(det) < 1e-9, 1e-9, det)
        JU = (j22 / detc, -j12 / detc, -j21 / detc, j11 / detc)
        return self.c + S * qu, JU, ok

    def radial_displacement(self, rpx):
        """|distorted| - |undistorted| radius (px) for undistorted radius rpx (poly) or distorted radius (div)."""
        q = np.column_stack([np.asarray(rpx, float) / S, np.zeros(len(np.atleast_1d(rpx)))])
        if self.kind == "poly":
            t = self.t
            self.t = None
            qd, _ = self.fwd(q)
            self.t = t
            return S * (qd[:, 0] - q[:, 0])
        qu, _ = self._div_back(q)
        return S * (q[:, 0] - qu[:, 0])

    def fold_margin(self, min_slope=0.0):
        """(r_d at which the radial map stops being monotonic) - (distance centre -> farthest image corner), px.

        Must be > 0: otherwise the lens is not invertible inside the image and undistortion-based residuals
        collapse near the fold.  Radial part only.
        """
        corners = np.array([[0, 0], [W - 1, 0], [0, H - 1], [W - 1, H - 1]], float)
        rc = np.max(np.hypot(*(corners - self.c).T)) / S
        if self.kind == "poly":
            ru = np.linspace(0, 3.5, 3501)
            r2 = ru * ru
            R = np.ones_like(ru)
            pw = np.ones_like(ru)
            for ai in self.a:
                pw = pw * r2
                R = R + ai * pw
            rd = ru * R
            sl = np.gradient(rd, ru)
            badi = np.where(sl <= min_slope)[0]
            rfold = rd[: badi[0]].max() if len(badi) else rd.max()
            return float(S * (rfold - rc))
        rd = np.linspace(0, 2.5, 2501)
        r2 = rd * rd
        Lr = np.ones_like(rd)
        pw = np.ones_like(rd)
        for ai in self.a:
            pw = pw * r2
            Lr = Lr + ai * pw
        ru = rd / np.where(Lr > 1e-9, Lr, 1e-9)
        sl = np.gradient(ru, rd)
        badi = np.where((sl <= min_slope) | (Lr <= 1e-6))[0]
        rfold = rd[badi[0]] if len(badi) else rd.max()
        return float(S * (rfold - rc))

    def opencv(self, f):
        """OpenCV [k1,k2,p1,p2,k3] for focal length f (poly only), pp = centre."""
        k = np.zeros(5)
        a = list(self.a) + [0.0] * 3
        k[0] = a[0] * f ** 2 / S ** 2
        k[1] = a[1] * f ** 4 / S ** 4
        k[4] = a[2] * f ** 6 / S ** 6
        if self.t is not None:
            k[2] = self.t[0] * f / S
            k[3] = self.t[1] * f / S
        return k

    def invariants(self):
        """pixel-unit invariants k1/f^2 [px^-2], k2/f^4 [px^-4], k3/f^6, p1/f, p2/f."""
        a = list(self.a) + [0.0] * 3
        out = {"k1_over_f2": a[0] / S ** 2, "k2_over_f4": a[1] / S ** 4, "k3_over_f6": a[2] / S ** 6}
        if self.t is not None:
            out["p1_over_f"] = self.t[0] / S
            out["p2_over_f"] = self.t[1] / S
        return out


def opencv_to_dist(K, d):
    """OpenCV (K, [k1,k2,p1,p2,k3]) -> Dist (poly) in pixel units (requires fx == fy)."""
    f = K[0, 0]
    a = [d[0] * S ** 2 / f ** 2, d[1] * S ** 4 / f ** 4, d[4] * S ** 6 / f ** 6]
    t = None if (d[2] == 0 and d[3] == 0) else [d[2] * S / f, d[3] * S / f]
    while len(a) > 1 and a[-1] == 0:
        a = a[:-1]
    return Dist("poly", a, t, centre=(K[0, 2], K[1, 2]))


# ----------------------------------------------------------------------------------------------------------
# edge set engine: vectorised per-edge moments, free lines and lines through vanishing points
# ----------------------------------------------------------------------------------------------------------
class EdgeSet:
    def __init__(self, edges, subsample=1):
        self.edges = edges
        pts = [e["points"][::subsample] for e in edges]
        self.n = np.array([len(p) for p in pts])
        self.P = np.vstack(pts)
        self.k = np.repeat(np.arange(len(pts)), self.n)
        self.E = len(pts)
        # arc parameter in [-1, 1] per edge (for bow statistics)
        s = []
        for p in pts:
            d = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
            s.append(2 * d / max(d[-1], 1e-9) - 1)
        self.s = np.concatenate(s)
        self.groups = [e.get("group") for e in edges]

    def moments(self, X, w):
        """per-edge weighted moments of homogeneous points (x, y, 1): returns E x 3 x 3."""
        k, E = self.k, self.E
        x, y = X[:, 0], X[:, 1]
        m = np.zeros((E, 3, 3))
        sw = np.bincount(k, w, E)
        sx = np.bincount(k, w * x, E)
        sy = np.bincount(k, w * y, E)
        m[:, 0, 0] = np.bincount(k, w * x * x, E)
        m[:, 0, 1] = m[:, 1, 0] = np.bincount(k, w * x * y, E)
        m[:, 1, 1] = np.bincount(k, w * y * y, E)
        m[:, 0, 2] = m[:, 2, 0] = sx
        m[:, 1, 2] = m[:, 2, 1] = sy
        m[:, 2, 2] = sw
        return m

    @staticmethod
    def free_lines(M):
        """TLS line per edge from moments: returns lines (E x 3) with (a,b) unit normal."""
        sw = M[:, 2, 2]
        mx, my = M[:, 0, 2] / sw, M[:, 1, 2] / sw
        cxx = M[:, 0, 0] / sw - mx * mx
        cxy = M[:, 0, 1] / sw - mx * my
        cyy = M[:, 1, 1] / sw - my * my
        th = 0.5 * np.arctan2(2 * cxy, cxx - cyy)  # direction of the major axis
        a, b = -np.sin(th), np.cos(th)
        return np.column_stack([a, b, -(a * mx + b * my)])

    @staticmethod
    def vp_lines(M, V):
        """Line per edge constrained through the homogeneous point V (E x 3 unit vectors, scaled coords)."""
        E = len(M)
        # orthonormal basis of V-perp
        tmp = np.where(np.abs(V[:, [0]]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
        e1 = np.cross(V, tmp)
        e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
        e2 = np.cross(V, e1)
        Eb = np.stack([e1, e2], axis=2)  # E x 3 x 2
        A = np.einsum("eia,eij,ejb->eab", Eb, M, Eb)
        Dm = np.diag([1.0, 1.0, 0.0])
        B = np.einsum("eia,ij,ejb->eab", Eb, Dm, Eb)
        a = B[:, 0, 0] * B[:, 1, 1] - B[:, 0, 1] ** 2
        b = -(A[:, 0, 0] * B[:, 1, 1] + A[:, 1, 1] * B[:, 0, 0] - 2 * A[:, 0, 1] * B[:, 0, 1])
        c = A[:, 0, 0] * A[:, 1, 1] - A[:, 0, 1] ** 2
        disc = np.sqrt(np.maximum(b * b - 4 * a * c, 0))
        lam = 2 * c / np.maximum(-b + disc, 1e-300)
        r1 = np.stack([-(A[:, 0, 1] - lam * B[:, 0, 1]), A[:, 0, 0] - lam * B[:, 0, 0]], 1)
        r2 = np.stack([A[:, 1, 1] - lam * B[:, 1, 1], -(A[:, 0, 1] - lam * B[:, 0, 1])], 1)
        alpha = np.where((np.sum(r1 ** 2, 1) >= np.sum(r2 ** 2, 1))[:, None], r1, r2)
        L = np.einsum("eia,ea->ei", Eb, alpha)
        nrm = np.hypot(L[:, 0], L[:, 1])
        return L / np.maximum(nrm, 1e-300)[:, None]

    def residuals(self, U, JU, V=None, weights=None, return_lines=False):
        """Distorted-image distances (px) of all points to their edge line.

        U: undistorted px (N x 2); JU: Jacobian tuple dU/dP; V: None or E x 3 array of homogeneous VPs in
        scaled coordinates ((u - C0)/S, 1) with NaN rows for edges without VP constraint.
        """
        X = (U - C0) / S
        k = self.k
        w0 = np.ones(len(X)) if weights is None else weights
        M = self.moments(X, w0)
        L = self.free_lines(M)
        # weights 1/g^2 with g = |J^T n| (n from the unweighted free line)
        j11, j12, j21, j22 = JU
        n = L[k, :2]
        g = np.hypot(j11 * n[:, 0] + j21 * n[:, 1], j12 * n[:, 0] + j22 * n[:, 1])
        w = w0 / g ** 2
        M = self.moments(X, w)
        L = self.free_lines(M)
        if V is not None:
            has = ~np.isnan(V[:, 0])
            if np.any(has):
                Lv = self.vp_lines(M[has], V[has])
                L = L.copy()
                L[has] = Lv
        n = L[k, :2]
        g = np.hypot(j11 * n[:, 0] + j21 * n[:, 1], j12 * n[:, 0] + j22 * n[:, 1])
        r = S * (X[:, 0] * L[k, 0] + X[:, 1] * L[k, 1] + L[k, 2]) / g
        if return_lines:
            return r, L
        return r

    def per_edge_stats(self, r):
        """rms, bow (quadratic sagitta of the residual along the edge, px), slope term per edge."""
        out = []
        for e in range(self.E):
            m = self.k == e
            rr, ss = r[m], self.s[m]
            A = np.column_stack([np.ones_like(ss), ss, ss * ss - np.mean(ss * ss)])
            c, *_ = np.linalg.lstsq(A, rr, rcond=None)
            res = rr - A @ c
            out.append(dict(rms=float(np.sqrt(np.mean(rr ** 2))), bow=float(-c[2]), tilt=float(c[1]),
                            noise=float(np.sqrt(np.mean(res ** 2) * len(rr) / max(len(rr) - 3, 1))), n=int(m.sum())))
        return out


def cluster_ids(edges):
    names = sorted({e["cluster"] for e in edges})
    idx = {n: i for i, n in enumerate(names)}
    return np.array([idx[e["cluster"]] for e in edges]), names


# ----------------------------------------------------------------------------------------------------------
# covariance helpers
# ----------------------------------------------------------------------------------------------------------
def cov_classic(J, r, npar_extra=0):
    dof = max(len(r) - J.shape[1] - npar_extra, 1)
    s2 = float(r @ r) / dof
    return np.linalg.pinv(J.T @ J) * s2


def cov_cluster(J, r, groups):
    """Cluster-robust (sandwich) covariance with small-sample factor G/(G-1)."""
    A = np.linalg.pinv(J.T @ J)
    B = np.zeros_like(A)
    ug = np.unique(groups)
    for g in ug:
        s = groups == g
        v = J[s].T @ r[s]
        B += np.outer(v, v)
    G = len(ug)
    return A @ B @ A * G / max(G - 1, 1)


def huber_weights(r, c):
    a = np.abs(r)
    return np.where(a <= c, 1.0, c / np.maximum(a, 1e-12))


# ----------------------------------------------------------------------------------------------------------
# plumb-line fit
# ----------------------------------------------------------------------------------------------------------
class PlumbModel:
    """Parameter packing for plumb-line fits."""

    def __init__(self, kind="poly", nrad=2, tang=False, centre_free=False, centre=C0):
        self.kind, self.nrad, self.tang, self.centre_free = kind, nrad, tang, centre_free
        self.centre = np.asarray(centre, float)
        self.names = (["cx", "cy"] if centre_free else []) + [("a%d" if kind == "poly" else "l%d") % (i + 1) for i in range(nrad)] + (["t1", "t2"] if tang else [])

    def dist(self, x):
        i = 0
        c = self.centre
        if self.centre_free:
            c = np.array([x[0], x[1]])
            i = 2
        a = x[i:i + self.nrad]
        i += self.nrad
        t = x[i:i + 2] if self.tang else None
        return Dist(self.kind, a, t, c)

    def x0(self, prev=None):
        x = []
        if self.centre_free:
            x += list(self.centre)
        if self.kind == "poly":
            a0 = [-0.10] if self.nrad == 1 else [-0.18, 0.03, 0.0, 0.0][: self.nrad]
        else:
            a0 = [-0.2, 0.0, 0.0][: self.nrad]
        if prev is not None:
            pa = list(prev.a) + [0.0] * 4
            a0 = pa[: self.nrad] if prev.kind == self.kind else a0
        x += a0
        if self.tang:
            x += [0.0, 0.0]
        return np.array(x, float)

    def x_scale(self):
        return np.array(([100.0, 100.0] if self.centre_free else []) + [0.1] * self.nrad + ([0.01, 0.01] if self.tang else []))


def fit_plumb(es, model, x0=None, loss="huber", f_scale=0.3, weights=None, max_nfev=200):
    """Plumb-line fit on EdgeSet es. Returns dict with x, dist, r, J, per-edge stats, covariances."""
    bad = np.full(len(es.P), 50.0)

    def fun(x):
        d = model.dist(x)
        if d.fold_margin(FOLD_MIN_SLOPE) < 0:
            return bad
        U, JU, ok = d.undistort(es.P)
        if not ok:
            return bad
        return es.residuals(U, JU, None, weights)

    x0 = model.x0() if x0 is None else x0
    res = least_squares(fun, x0, loss=loss, f_scale=f_scale, x_scale=model.x_scale(), method="trf",
                        diff_step=1e-6, max_nfev=max_nfev, xtol=1e-12, ftol=1e-12, gtol=1e-12)
    r = fun(res.x)
    # raw Jacobian (not loss-scaled) by central differences for covariance
    J = numjac(fun, res.x, model.x_scale() * 1e-4)
    w = huber_weights(r, f_scale) if loss == "huber" else np.ones_like(r)
    sw = np.sqrt(w)
    Jw, rw = J * sw[:, None], r * sw
    out = dict(x=res.x, names=model.names, dist=model.dist(res.x), r=r, J=J, w=w, fold_margin_px=model.dist(res.x).fold_margin(),
               barrier_margin_px=model.dist(res.x).fold_margin(FOLD_MIN_SLOPE),
               rms=float(np.sqrt(np.mean(r ** 2))), cost=float(res.cost), nfev=res.nfev, success=bool(res.success))
    out["cov_classic"] = cov_classic(Jw, rw, npar_extra=2 * es.E)
    cl = np.array([e["cluster"] for e in es.edges])
    _, inv = np.unique(cl, return_inverse=True)
    out["cov_cluster"] = cov_cluster(Jw, rw, inv[es.k])
    out["per_edge"] = es.per_edge_stats(r)
    return out


def numjac(fun, x, h):
    f0 = fun(x)
    J = np.zeros((len(f0), len(x)))
    for i in range(len(x)):
        xp, xm = x.copy(), x.copy()
        xp[i] += h[i]
        xm[i] -= h[i]
        J[:, i] = (fun(xp) - fun(xm)) / (2 * h[i])
    return J


def robust_edge_rejection(edges, model, max_iter=8, fac=3.0, bow_abs=0.6, rms_abs=0.5, verbose=True, f_scale=0.3):
    """Iteratively fit and reject edges that stay curved / noisy after correction (monotone: no re-admission).

    Per edge (evaluated for ALL edges with the current fit): residual bow b (quadratic sagitta along the edge)
    and rms.  Rejected if |b - median(b_active)| > max(bow_abs, fac * 1.4826 * MAD(b_active))
    or rms > max(rms_abs, fac * median(rms_active)).
    """
    active = np.ones(len(edges), bool)
    hist = []
    x = None
    for it in range(max_iter):
        es = EdgeSet([e for e, a in zip(edges, active) if a])
        fit = fit_plumb(es, model, x0=x, f_scale=f_scale)
        x = fit["x"]
        esa = EdgeSet(edges)
        d = fit["dist"]
        U, JU, _ = d.undistort(esa.P)
        r = esa.residuals(U, JU)
        st = esa.per_edge_stats(r)
        bows = np.array([s["bow"] for s in st])
        rmss = np.array([s["rms"] for s in st])
        med = np.median(bows[active])
        mad = 1.4826 * np.median(np.abs(bows[active] - med))
        tb = max(bow_abs, fac * mad)
        tr = max(rms_abs, fac * np.median(rmss[active]))
        bad = (np.abs(bows - med) > tb) | (rmss > tr)
        new = active & ~bad
        hist.append(dict(iter=it, n_active=int(active.sum()), bow_median=float(med), bow_thr=float(tb), rms_thr=float(tr),
                         flagged=[edges[i]["id"] for i in np.where(bad)[0]], x=x.tolist()))
        if verbose:
            print(f"  reject iter {it}: active {active.sum()} -> {new.sum()}, bow median {med:+.2f} thr {tb:.2f}, rms thr {tr:.2f};",
                  "flagged:", [edges[i]["id"] for i in np.where(bad)[0]])
        if np.array_equal(new, active):
            break
        active = new
    return active, hist, st


# ----------------------------------------------------------------------------------------------------------
# vanishing points
# ----------------------------------------------------------------------------------------------------------
def to_scaled_h(v_px_h):
    """homogeneous pixel point -> homogeneous scaled coords ((u-C0)/S, 1), normalised to unit length."""
    v = np.asarray(v_px_h, float)
    out = np.array([(v[0] - C0[0] * v[2]) / S, (v[1] - C0[1] * v[2]) / S, v[2]])
    return out / np.linalg.norm(out)


def from_scaled_h(V):
    V = np.asarray(V, float)
    return np.array([S * V[0] + C0[0] * V[2], S * V[1] + C0[1] * V[2], V[2]])


def vp_basis(V0):
    tmp = np.array([1.0, 0, 0]) if abs(V0[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(V0, tmp)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(V0, e1)
    return e1, e2


def vp_init(L):
    """least-squares intersection of lines (rows, (a,b) unit) -> unit homogeneous point."""
    _, _, vt = np.linalg.svd(L)
    v = vt[-1]
    return v / np.linalg.norm(v)


def fit_vp_group(U, JU, es_group, loss="linear", f_scale=0.5):
    """MLE of one vanishing point from the (undistorted) points of the edges of one group.

    es_group: EdgeSet of the group's edges; U, JU for its points. Returns V (unit, scaled homog coords),
    cov (2x2 on the tangent plane basis e1,e2 at V), per-edge stats, residuals and the cluster covariance.
    """
    r0, Lf = es_group.residuals(U, JU, None, return_lines=True)
    V0 = vp_init(Lf)
    e1, e2 = vp_basis(V0)

    def Vof(d):
        v = V0 + d[0] * e1 + d[1] * e2
        return v / np.linalg.norm(v)

    def fun(d):
        V = np.tile(Vof(d), (es_group.E, 1))
        return es_group.residuals(U, JU, V)

    res = least_squares(fun, np.zeros(2), loss=loss, f_scale=f_scale, x_scale=np.array([1e-3, 1e-3]), xtol=1e-14, ftol=1e-14, gtol=1e-14)
    # re-centre the tangent basis at the solution
    V1 = Vof(res.x)
    e1, e2 = vp_basis(V1)
    V0 = V1

    r = fun(np.zeros(2))
    J = numjac(fun, np.zeros(2), np.array([1e-6, 1e-6]))
    cl = np.array([e["cluster"] for e in es_group.edges])
    _, inv = np.unique(cl, return_inverse=True)
    out = dict(V=V1, basis=(e1, e2), r=r, rms=float(np.sqrt(np.mean(r ** 2))), rms_free=float(np.sqrt(np.mean(r0 ** 2))),
               cov_classic=cov_classic(J, r, npar_extra=es_group.E), n_edges=es_group.E,
               n_clusters=int(len(np.unique(inv))), per_edge=es_group.per_edge_stats(r),
               per_edge_free=es_group.per_edge_stats(r0))
    out["cov_cluster"] = cov_cluster(J, r, inv[es_group.k]) if len(np.unique(inv)) > 2 else None
    out["V_px"] = from_scaled_h(V1)
    return out


def ray(Vs, f, pp):
    """unit viewing direction of a homogeneous scaled VP for K(f, pp)."""
    v = from_scaled_h(Vs)
    d = np.array([(v[0] - pp[0] * v[2]) / f, (v[1] - pp[1] * v[2]) / f, v[2]])
    return d / np.linalg.norm(d)


def f_from_pair(Va, Vb, pp):
    """f from two orthogonal VPs with known pp (homogeneous-safe); returns nan if not real."""
    va, vb = from_scaled_h(Va), from_scaled_h(Vb)
    aa = va[:2] - pp * va[2]
    bb = vb[:2] - pp * vb[2]
    den = va[2] * vb[2]
    f2 = -(aa @ bb) / den if abs(den) > 1e-15 else np.nan
    return float(np.sqrt(f2)) if f2 > 0 else np.nan


def orthocentre(V1, V2, V3):
    """orthocentre and f from three mutually orthogonal (finite) VPs (pixel coords)."""
    a, b, c = [from_scaled_h(v) for v in (V1, V2, V3)]
    a, b, c = a[:2] / a[2], b[:2] / b[2], c[:2] / c[2]
    # p satisfies (p - a).(b - c) = 0, (p - b).(a - c) = 0
    A = np.array([b - c, a - c])
    rhs = np.array([a @ (b - c), b @ (a - c)])
    p = np.linalg.solve(A, rhs)
    f2 = -((a - p) @ (b - p))
    return p, (float(np.sqrt(f2)) if f2 > 0 else np.nan)


# ----------------------------------------------------------------------------------------------------------
# rotations
# ----------------------------------------------------------------------------------------------------------
def rodrigues(w):
    w = np.asarray(w, float)
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx


def rot_to_vec(R):
    th = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if th < 1e-12:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(th))
    return w * th


def rot_from_two(dx, dz):
    """rotation with columns (X, Y, Z) from approximate X and Z directions (Z kept, X orthogonalised)."""
    z = dz / np.linalg.norm(dz)
    x = dx - (dx @ z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


# ----------------------------------------------------------------------------------------------------------
# mapping comparison for arbitrary models (ray functions) - same definition as evaltools (rotation-compensated)
# ----------------------------------------------------------------------------------------------------------
def grid_uv(step=40):
    xs = np.arange(0, W + 1e-9, step)
    ys = np.arange(0, H + 1e-9, step)
    X, Y = np.meshgrid(xs, ys)
    return np.column_stack([X.ravel(), Y.ravel()]).astype(float)


def model_rays(uv, f, pp, dist):
    """viewing rays (unit) for pixels uv: undistort with dist (pixel units), then pinhole K(f, pp)."""
    U, _, _ = dist.undistort(uv)
    r = np.column_stack([(U[:, 0] - pp[0]) / f, (U[:, 1] - pp[1]) / f, np.ones(len(U))])
    return r / np.linalg.norm(r, axis=1, keepdims=True)


def model_project(rays_, f, pp, dist):
    x = rays_[:, 0] / rays_[:, 2]
    y = rays_[:, 1] / rays_[:, 2]
    U = np.column_stack([pp[0] + f * x, pp[1] + f * y])
    return dist.distort_px(U)


def fit_opencv_to_model(f, pp, dist, nk=3, tang=False, step=40):
    """Least-squares OpenCV (fx=fy=f', pp', k1,k2[,k3][,p1,p2]) mapping of an arbitrary pixel-unit model."""
    from common import project, K_from

    uv = grid_uv(step)
    R = model_rays(uv, f, pp, dist)

    def unpack(x):
        K = K_from(x[0], x[0], x[1], x[2])
        d = np.zeros(5)
        d[0] = x[3]
        if nk > 1:
            d[1] = x[4]
        if nk > 2:
            d[4] = x[5]
        if tang:
            d[2], d[3] = x[3 + nk], x[4 + nk]
        return K, d

    x0 = np.r_[f, pp, dist.opencv(f)[[0, 1, 4]][:nk], [0.0, 0.0] if tang else []]

    def fun(x):
        K, d = unpack(x)
        return (project(R, K, d) - uv).ravel()

    r = least_squares(fun, x0, x_scale=np.r_[10, 10, 10, [0.05] * nk, [1e-3, 1e-3] if tang else []])
    K, d = unpack(r.x)
    res = fun(r.x).reshape(-1, 2)
    return K, d, float(np.sqrt(np.mean(np.sum(res ** 2, 1)))), float(np.max(np.hypot(res[:, 0], res[:, 1])))


def mapping_disp_rays(uv, rays1, proj2, compensate=True):
    """displacement proj2(R rays1) - uv, R minimising the summed squared displacement (evaltools convention)."""
    if not compensate:
        return proj2(rays1) - uv

    def res(a):
        return (proj2(rays1 @ rodrigues(a).T) - uv).ravel()

    a = least_squares(res, np.zeros(3), x_scale=1e-3).x
    return proj2(rays1 @ rodrigues(a).T) - uv


def save(obj, name):
    save_json(obj, f"{CACHE}/{name}")


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


# ----------------------------------------------------------------------------------------------------------
# joint line self-calibration model (distortion centre = principal point, OpenCV-compatible)
# ----------------------------------------------------------------------------------------------------------
def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


class DirParam:
    """unit 3-vector near d0: d = normalize(d0 + a e1 + b e2)"""

    def __init__(self, d0):
        self.d0 = unit(d0)
        self.e1, self.e2 = vp_basis(self.d0)

    def __call__(self, ab):
        return unit(self.d0 + ab[0] * self.e1 + ab[1] * self.e2)


class JointModel:
    """Unknowns: f, (cx, cy) [= distortion centre], radial coeffs (poly a_i or div l_i), then direction params.

    frames:
      'tied'  : common vertical z (2) + azimuth of cart 310 X and cart 80 X in the plane _|_ z (1 + 1);
                groups 310X, 80X -> x_c;  310Z, 80Z -> z
      'free'  : independent cart frames: cart c has X direction (2) and Z (1 = rotation of Z about X, Z _|_ X)
    wv: None (WV edges straightness only) | 'free' (own direction, 2) | 'tied' (WV -> z; frames 'tied' only)
    """

    def __init__(self, edges, kind="poly", nrad=2, pp_free=True, pp_fixed=C0, frames="tied", wv="free", init=None,
                 use_groups=("310X", "80X", "310Z", "80Z", "WV"), vp_mode="re", sig_pt=0.2, sig_psi_deg=0.157):
        """vp_mode 're': residuals = [point distances to each edge's FREE line / sig_pt] + [one angular residual
        per VP edge: psi_e / sqrt(sig_psi^2 + noise_e^2)] (per-edge direction error treated as random effect);
        'pt': every point of a VP edge is measured against the line through the VP (no direction error)."""
        self.kind, self.nrad, self.pp_free, self.frames, self.wv = kind, nrad, pp_free, frames, wv
        self.vp_mode, self.sig_pt, self.sig_psi = vp_mode, sig_pt, np.radians(sig_psi_deg)
        self.pp_fixed = np.asarray(pp_fixed, float)
        self.edges = edges
        self.es = EdgeSet(edges)
        self.use = set(use_groups)
        if wv is None:
            self.use.discard("WV")
        self.gr = [e["group"] if e["group"] in self.use else None for e in edges]
        init = init or {}
        self.f0 = init.get("f", 1450.0)
        pp0 = np.asarray(init.get("pp", C0 if not pp_free else (930.0, 440.0)), float)
        self.a0 = list(init.get("a", [-0.16, 0.02, 0.0][:nrad] if kind == "poly" else [-0.19, -0.02][:nrad]))
        K0 = np.array([[self.f0, 0, pp0[0]], [0, self.f0, pp0[1]], [0, 0, 1.0]])
        Ki = np.linalg.inv(K0)
        vpx = init.get("vp_px", {})  # homogeneous pixel VPs for initial directions

        def d_of(g, default):
            if g in vpx:
                return unit(Ki @ np.asarray(vpx[g], float))
            return unit(default)

        z = d_of("CZ", [0.0, -0.33, 0.94])
        if z[2] < 0:
            z = -z
        self.zp = DirParam(z)
        self.xref = {}
        self.xp = {}
        for c in (310, 80):
            x = d_of(f"{c}X", [0.3, 0.95, 0.1] if c == 310 else [-0.1, 0.99, 0.1])
            self.xref[c] = x
            self.xp[c] = DirParam(x)
        self.wvp = DirParam(d_of("WV", z))
        self.names = ["f"] + (["cx", "cy"] if pp_free else []) + [f"a{i + 1}" for i in range(nrad)]
        self.nin = len(self.names)
        if frames == "tied":
            self.names += ["z_a", "z_b", "psi310", "psi80"]
        else:
            self.names += ["x310_a", "x310_b", "rz310", "x80_a", "x80_b", "rz80"]
        if wv == "free":
            self.names += ["wv_a", "wv_b"]
        self.pp0 = pp0
        # per-edge angular noise (point noise only) for the random-effect residuals
        self.gidx = np.array([k for k, g in enumerate(self.gr) if g is not None], int)
        self.noise_psi = np.zeros(len(self.gidx))
        if vp_mode == "re" and len(self.gidx):
            d0 = Dist(kind, self.a0, None, pp0)
            U, JU, _ = d0.undistort(self.es.P)
            r, Ls = self.es.residuals(U, JU, return_lines=True)
            st = self.es.per_edge_stats(r)
            for j, k in enumerate(self.gidx):
                m = self.es.k == k
                Lpx = float(np.hypot(*(U[m][-1] - U[m][0])))
                sig_th = st[k]["noise"] * np.sqrt(12.0 / m.sum()) / max(Lpx, 1.0)
                # d psi / d theta for a rotation of the line about the edge centroid (d = any direction on the line's
                # interpretation plane is irrelevant: |dn/dtheta| projected -> use the plane-normal rotation rate)
                a, b, c = Ls[k]
                lp = np.array([a / S, b / S, c - (a * C0[0] + b * C0[1]) / S])
                cen = U[m].mean(0)
                th = np.arctan2(-lp[0], lp[1])
                ns = []
                for dth in (0.0, 1e-4):
                    nn = np.array([-np.sin(th + dth), np.cos(th + dth)])
                    l2 = np.array([nn[0], nn[1], -nn @ cen])
                    nv = np.array([self.f0 * l2[0], self.f0 * l2[1], l2[0] * pp0[0] + l2[1] * pp0[1] + l2[2]])
                    ns.append(nv / np.linalg.norm(nv))
                rate = np.linalg.norm(ns[1] - ns[0]) / 1e-4
                self.noise_psi[j] = sig_th * rate

    def x0(self):
        x = [self.f0] + (list(self.pp0) if self.pp_free else []) + list(self.a0)
        x += [0.0, 0.0, 0.0, 0.0] if self.frames == "tied" else [0.0] * 6
        if self.wv == "free":
            x += [0.0, 0.0]
        return np.array(x, float)

    def x_scale(self):
        s = [100.0] + ([30.0, 30.0] if self.pp_free else []) + [0.05] * self.nrad
        s += [1e-3] * (4 if self.frames == "tied" else 6)
        if self.wv == "free":
            s += [1e-3, 1e-3]
        return np.array(s)

    def unpack(self, x):
        f = x[0]
        i = 1
        if self.pp_free:
            pp = np.array([x[1], x[2]])
            i = 3
        else:
            pp = self.pp_fixed
        a = x[i:i + self.nrad]
        i += self.nrad
        dist = Dist(self.kind, a, None, pp)
        dirs = {}
        if self.frames == "tied":
            z = self.zp(x[i:i + 2])
            for c, ps in ((310, x[i + 2]), (80, x[i + 3])):
                u = unit(self.xref[c] - (self.xref[c] @ z) * z)
                v = np.cross(z, u)
                xd = np.cos(ps) * u + np.sin(ps) * v
                dirs[f"{c}X"] = xd
                dirs[f"{c}Z"] = z
            dirs["CZ"] = z
            i += 4
        else:
            for c in (310, 80):
                xd = self.xp[c](x[i:i + 2])
                # Z _|_ X: start from the tied initial vertical, orthogonalise, rotate about X by rz
                z0 = unit(self.zp.d0 - (self.zp.d0 @ xd) * xd)
                w = np.cross(xd, z0)
                zd = np.cos(x[i + 2]) * z0 + np.sin(x[i + 2]) * w
                dirs[f"{c}X"] = xd
                dirs[f"{c}Z"] = zd
                i += 3
        if self.wv == "free":
            dirs["WV"] = self.wvp(x[i:i + 2])
            i += 2
        elif self.wv == "tied":
            dirs["WV"] = dirs["310Z"]
        return f, pp, dist, dirs

    def vp_scaled(self, f, pp, d):
        v = np.array([f * d[0] + pp[0] * d[2], f * d[1] + pp[1] * d[2], d[2]])
        return to_scaled_h(v)

    def residuals(self, x):
        f, pp, dist, dirs = self.unpack(x)
        nbad = len(self.es.P) + (len(self.gidx) if self.vp_mode == "re" else 0)
        if dist.fold_margin(FOLD_MIN_SLOPE) < 0 or f < 300:
            return np.full(nbad, 50.0)
        U, JU, ok = dist.undistort(self.es.P)
        if not ok:
            return np.full(nbad, 50.0)
        if self.vp_mode == "re":
            r, Ls = self.es.residuals(U, JU, None, return_lines=True)
            psi = self.psi(Ls, f, pp, dirs)
            return np.concatenate([r / self.sig_pt, psi / np.sqrt(self.sig_psi ** 2 + self.noise_psi ** 2)])
        V = np.full((self.es.E, 3), np.nan)
        for k, g in enumerate(self.gr):
            if g is not None and g in dirs:
                V[k] = self.vp_scaled(f, pp, dirs[g])
        return self.es.residuals(U, JU, V)

    def psi(self, Ls, f, pp, dirs):
        """angle between each VP edge's interpretation plane (free line) and its group direction (rad)."""
        out = np.zeros(len(self.gidx))
        for j, k in enumerate(self.gidx):
            a, b, c = Ls[k]
            lp = np.array([a / S, b / S, c - (a * C0[0] + b * C0[1]) / S])  # line in pixel coords
            n = np.array([f * lp[0], f * lp[1], lp[0] * pp[0] + lp[1] * pp[1] + lp[2]])  # K^T l
            d = dirs[self.gr[k]]
            out[j] = np.arcsin(np.clip(n @ d / np.linalg.norm(n), -1, 1))
        return out

    def fit(self, x0=None, loss="huber", f_scale=0.3, max_nfev=300):
        x0 = self.x0() if x0 is None else x0
        res = least_squares(self.residuals, x0, loss=loss, f_scale=f_scale, x_scale=self.x_scale(), method="trf",
                            diff_step=1e-6, max_nfev=max_nfev, xtol=1e-12, ftol=1e-12, gtol=1e-12)
        return res

    def opencv(self, x):
        f, pp, dist, _ = self.unpack(x)
        return f, pp, dist


# ----------------------------------------------------------------------------------------------------------
# per-edge direction-error model (random effect) and perturbation for the bootstrap
# ----------------------------------------------------------------------------------------------------------
def edge_angle_rates(edge_list, dist, f, pp):
    """per edge: (rate = |d n / d theta| of the interpretation-plane normal for a rotation of the undistorted line
    about its centroid, noise angle of the free line from the point scatter [rad of image angle])."""
    es = EdgeSet(edge_list)
    U, JU, _ = dist.undistort(es.P)
    r, Ls = es.residuals(U, JU, return_lines=True)
    st = es.per_edge_stats(r)
    out = {}
    for k, e in enumerate(edge_list):
        m = es.k == k
        Lpx = float(np.hypot(*(U[m][-1] - U[m][0])))
        sig_th = st[k]["noise"] * np.sqrt(12.0 / m.sum()) / max(Lpx, 1.0)
        a, b, c = Ls[k]
        lp = np.array([a / S, b / S, c - (a * C0[0] + b * C0[1]) / S])
        cen = U[m].mean(0)
        th = np.arctan2(-lp[0], lp[1])
        ns = []
        for dth in (0.0, 1e-4):
            nn = np.array([-np.sin(th + dth), np.cos(th + dth)])
            l2 = np.array([nn[0], nn[1], -nn @ cen])
            nv = np.array([f * l2[0], f * l2[1], l2[0] * pp[0] + l2[1] * pp[1] + l2[2]])
            ns.append(nv / np.linalg.norm(nv))
        rate = float(np.linalg.norm(ns[1] - ns[0]) / 1e-4)
        out[e["id"]] = dict(rate=rate, sig_theta=float(sig_th), noise_psi=float(sig_th * rate))
    return out


def perturb_directions(edge_list, rates, sig_psi, rng_, groups):
    """rotate the (distorted) points of every edge whose group is in `groups` about their centroid by a random
    image angle ~ N(0, sig_psi^2 + noise_psi^2) / rate (random direction error of the physical member)."""
    out = []
    for e in edge_list:
        g = e.get("vgroup", e.get("group"))
        if g not in groups or e["id"] not in rates:
            out.append(e)
            continue
        r = rates[e["id"]]
        s = np.sqrt(sig_psi ** 2 + r["noise_psi"] ** 2)
        dth = rng_.normal(0, s) / max(r["rate"], 1e-9)
        P = e["points"]
        c = P.mean(0)
        R = np.array([[np.cos(dth), -np.sin(dth)], [np.sin(dth), np.cos(dth)]])
        out.append(dict(e, points=(P - c) @ R.T + c))
    return out


def stratified_resample(edge_list, rng_, rich=("310X", "80X"), small=("CZ", "WV", "FA", "FC")):
    """resample physical members (clusters) with replacement within the strata rich[i] and 'rest'; members of the
    small VP groups are kept (they get a parametric direction perturbation instead)."""
    def vg(e):
        return e.get("vgroup", e.get("group"))

    out = [e for e in edge_list if vg(e) in small]
    for st in list(rich) + ["rest"]:
        pool = [e for e in edge_list if (vg(e) == st if st != "rest" else (vg(e) not in small and vg(e) not in rich))]
        names = sorted({e["cluster"] for e in pool})
        if not names:
            continue
        pick = rng_.integers(0, len(names), len(names))
        for q, c in enumerate(pick):
            out += [dict(e, cluster=f"{e['cluster']}#{st}{q}") for e in pool if e["cluster"] == names[c]]
    return out


def vp_estimates(edge_list, dist, groups):
    """VP MLE per group (merged groups: 'CZ' = 310Z + 80Z)."""
    out = {}
    for g in groups:
        sub = [e for e in edge_list if (e["group"] in ("310Z", "80Z") if g == "CZ" else e["group"] == g)]
        if len(sub) < 2:
            continue
        es = EdgeSet(sub)
        U, JU, _ = dist.undistort(es.P)
        out[g] = fit_vp_group(U, JU, es)
    return out


def f_orthogonal(vps, pairs, pp, f0=1450.0, weights=None):
    """f with fixed pp from orthogonal VP pairs (least squares on the cosines, optional weights)."""
    pairs = [p for p in pairs if p[0] in vps and p[1] in vps]
    w = np.ones(len(pairs)) if weights is None else np.asarray(weights, float)

    def fun(x):
        return w * np.array([ray(vps[a]["V"], x[0], pp) @ ray(vps[b]["V"], x[0], pp) for a, b in pairs])

    r = least_squares(fun, [f0], x_scale=[100.0])
    return float(r.x[0])
