"""Lens-model comparison library (sub-task 43): generic lens models + one estimator for all data blocks.

Lens models (all share f / (fx, fy), cx, cy; the distortion centre is the principal point):
  brown : OpenCV pinhole + Brown-Conrady (k1,k2,k3,p1,p2) incl. the rational terms k4..k6
          r_d = r_u (1 + k1 s + k2 s^2 + k3 s^3) / (1 + k4 s + k5 s^2 + k6 s^3),  s = r_u^2
  kb    : Kannala-Brandt (cv2.fisheye): th = atan(r_u), r_d = th (1 + k1 th^2 + k2 th^4 + k3 th^6 + k4 th^8)
  div   : division model defined by its UNdistortion: r_u = r_d / (1 + l1 r_d^2 + l2 r_d^4)
All radii in normalised units (pixels / f). Closed form in one direction, Newton inverse in the other.

Estimator blocks (each a distance in DISTORTED image px, divided by a block sigma):
  M : sticker corner reprojection (drawing geometry, exact)
  L : exact 3-D model lines (sides of partly hidden stickers; board edges with an exact model line)
  V : edges of a cart-axis direction group (or scene verticals) must point to the group's vanishing point
  S : straight edges (own TLS line in the undistorted image)
Edge residuals: r_d = r_u / |J^T n| (J = Jacobian of the undistortion, n = line normal) - the same
unbiased form as lineselfcal / combined.
Poses: 6-DoF per cart when M or L data of that cart are present, otherwise only a rotation (edges only).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from common import CACHE, H, W, distort_normalized, load_json, rodrigues, undistort_points

AXIS = {"cartX": 0, "cartY": 1, "cartZ": 2}
C0 = np.array([(W - 1) / 2.0, (H - 1) / 2.0])
BORDER = ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"]
INIT = load_json(f"{CACHE}/initial_calib.json")
P0 = {80: np.array(INIT["poses"]["80"]), 310: np.array(INIT["poses"]["310"])}


# =====================================================================================
# lens models
# =====================================================================================

def _newton_inverse(fun, target, x0, iters=60, lo=0.0, hi=50.0):
    """Solve fun(x) = target (vectorised, monotone fun expected) by damped Newton."""
    x = np.array(x0, float)
    t = np.asarray(target, float)
    for _ in range(iters):
        e = 1e-7 * np.maximum(1.0, np.abs(x))
        f0 = fun(x) - t
        d = (fun(x + e) - fun(x - e)) / (2 * e)
        d = np.where(d < 1e-4, 1e-4, d)
        step = f0 / d
        x = np.clip(x - step, lo, hi)
        if np.max(np.abs(step)) < 1e-15:
            break
    return x


class Lens:
    COEFS = {"brown": BORDER, "kb": ["k1", "k2", "k3", "k4"], "div": ["l1", "l2"]}

    def __init__(self, kind="brown", coef=("k1", "k2"), f="single", pp="free", pp0=None, fixed=None, name=None):
        self.kind = kind
        self.coef = list(coef)
        for c in self.coef:
            assert c in self.COEFS[kind], (kind, c)
        self.fmode = f
        self.ppmode = pp
        self.pp0 = C0.copy() if pp0 is None else np.asarray(pp0, float)
        self.fixed = dict(fixed or {})  # fixed (non-zero) values of coefficients that are not free
        self.names = (["f"] if f == "single" else ["fx", "fy"]) + (["cx", "cy"] if pp == "free" else []) + self.coef
        self.ni = len(self.names)
        self.tangential = kind == "brown" and ("p1" in self.coef or "p2" in self.coef or
                                               self.fixed.get("p1", 0) != 0 or self.fixed.get("p2", 0) != 0)
        self.name = name or f"{kind}_{''.join(self.coef)}_{f}_{pp}"

    def spec(self):
        return dict(kind=self.kind, coef=self.coef, f=self.fmode, pp=self.ppmode, pp0=self.pp0.tolist(), fixed=self.fixed, name=self.name)

    # ---- parameters -------------------------------------------------------------
    def P(self, th):
        th = np.asarray(th, float)
        i = 0
        if self.fmode == "single":
            fx = fy = th[0]
            i = 1
        else:
            fx, fy = th[0], th[1]
            i = 2
        if self.ppmode == "free":
            cx, cy = th[i], th[i + 1]
            i += 2
        else:
            cx, cy = self.pp0
        c = {k: 0.0 for k in self.COEFS[self.kind]}
        c.update(self.fixed)
        for nm in self.coef:
            c[nm] = th[i]
            i += 1
        return dict(fx=fx, fy=fy, cx=cx, cy=cy, **c)

    def pack(self, P):
        th = [P["fx"]] if self.fmode == "single" else [P["fx"], P["fy"]]
        if self.ppmode == "free":
            th += [P["cx"], P["cy"]]
        th += [P.get(nm, 0.0) for nm in self.coef]
        return np.array(th, float)

    def brown_dist(self, P):
        return np.array([P.get(k, 0.0) for k in BORDER])

    # ---- radial functions (normalised units) -----------------------------------------
    def g(self, ru, P):
        """r_u -> r_d (closed form for brown / kb)."""
        if self.kind == "brown":
            s = ru * ru
            num = 1 + P["k1"] * s + P["k2"] * s ** 2 + P["k3"] * s ** 3
            den = 1 + P["k4"] * s + P["k5"] * s ** 2 + P["k6"] * s ** 3
            return ru * num / den
        if self.kind == "kb":
            th = np.arctan(ru)
            t2 = th * th
            return th * (1 + P["k1"] * t2 + P["k2"] * t2 ** 2 + P["k3"] * t2 ** 3 + P["k4"] * t2 ** 4)
        # division: forward by inversion of h
        return _newton_inverse(lambda rd: self.h(rd, P), ru, ru)

    def h(self, rd, P):
        """r_d -> r_u (closed form for div)."""
        if self.kind == "div":
            s = rd * rd
            return rd / (1 + P["l1"] * s + P["l2"] * s * s)
        return _newton_inverse(lambda ru: self.g(ru, P), rd, rd)

    # ---- normalised forward / inverse ------------------------------------------------
    def fwd(self, xu, yu, P):
        if self.tangential:
            return distort_normalized(xu, yu, self.brown_dist(P))
        ru = np.hypot(xu, yu)
        rd = self.g(ru, P)
        s = np.where(ru > 1e-12, rd / np.maximum(ru, 1e-12), 1.0)
        return xu * s, yu * s

    def inv(self, xd, yd, P):
        if self.tangential:
            n = undistort_points(np.column_stack([xd, yd]), np.eye(3), self.brown_dist(P), iters=60)
            return n[:, 0], n[:, 1]
        rd = np.hypot(xd, yd)
        ru = self.h(rd, P)
        s = np.where(rd > 1e-12, ru / np.maximum(rd, 1e-12), 1.0)
        return xd * s, yd * s

    # ---- pixel level -----------------------------------------------------------------
    def project(self, Xc, P):
        xu = Xc[:, 0] / Xc[:, 2]
        yu = Xc[:, 1] / Xc[:, 2]
        xd, yd = self.fwd(xu, yu, P)
        return np.column_stack([P["fx"] * xd + P["cx"], P["fy"] * yd + P["cy"]])

    def norm_und(self, uv, P):
        xd = (uv[:, 0] - P["cx"]) / P["fx"]
        yd = (uv[:, 1] - P["cy"]) / P["fy"]
        xu, yu = self.inv(xd, yd, P)
        return np.column_stack([xu, yu])

    def undist_px(self, uv, P):
        n = self.norm_und(uv, P)
        return np.column_stack([P["fx"] * n[:, 0] + P["cx"], P["fy"] * n[:, 1] + P["cy"]])

    def rays(self, uv, P):
        n = self.norm_und(uv, P)
        r = np.column_stack([n, np.ones(len(n))])
        return r / np.linalg.norm(r, axis=1, keepdims=True)

    def K(self, P):
        return np.array([[P["fx"], 0, P["cx"]], [0, P["fy"], P["cy"]], [0, 0, 1.0]])

    def monotone_ok(self, P, rmax_px=None):
        """True if r_d(r_u) is monotone over the image (checks the radial function along the diagonal)."""
        if self.tangential:
            P2 = dict(P, p1=0.0, p2=0.0)
        else:
            P2 = P
        rmax = (rmax_px or np.hypot(W, H) * 0.6) / min(P["fx"], P["fy"])
        rd = np.linspace(1e-4, rmax, 400)
        if self.kind == "div":
            ru = self.h(rd, P2)
            return bool(np.all(np.diff(ru) > 0) and np.all(ru > 0))
        ru = np.linspace(1e-4, 3 * rmax, 1200)
        gd = self.g(ru, P2)
        ok = np.all(np.diff(gd) > 0)
        return bool(ok or gd[np.argmax(np.diff(gd) <= 0)] > rmax)


# =====================================================================================
# data
# =====================================================================================

def model_line_3d(ml):
    if not ml or not ml.get("exact", False):
        return None
    A = np.zeros(3)
    for k, v in ml["fixed"].items():
        A["XYZ".index(k)] = float(v)
    D = np.zeros(3)
    D["XYZ".index(ml["free"])] = 1.0
    return A, D


def vp_group(e, policy="noboard"):
    """Direction group of an edge for the V block (None -> straightness only)."""
    d = e.get("direction")
    if d in AXIS and e.get("cart") in (80, 310):
        if policy == "noboard" and "board" in e["id"]:
            return None
        return (int(e["cart"]), AXIS[d])
    if d == "world_vertical":
        return ("wv", 2)
    return None


def load_all(include_sides=True, edge_source=None, drop_edge_ids=()):
    from linedata import load_edges
    from markerdata import load_markers, marker_side_edges, to_points

    M = load_markers()
    pts = to_points(M)
    sides = marker_side_edges(M, only_partial=True) if include_sides else []
    if edge_source is None:
        edges = load_edges(verified_only=True)
    else:
        edges = []
        for fn in edge_source:
            edges += load_edges(verified_only=True, source=fn)
    edges = [e for e in edges if e["id"] not in set(drop_edge_ids)]
    return dict(markers=M, pts=pts, sides=sides, edges=edges)


def tile_of(e, nx=3, ny=3):
    c = e["points"].mean(0)
    return f"t{min(int(c[0] / (W / nx)), nx - 1)}{min(int(c[1] / (H / ny)), ny - 1)}"


def build(D, dataset, drop_sticker=None, drop_edges=(), policy="noboard"):
    """dataset: 'M' (stickers: corners + sides), 'E' (edges: V + S), 'ME' (all)."""
    out = dict(M=[], L=[], V=[], S=[])
    drop_edges = set(drop_edges)
    if "M" in dataset:
        out["M"] = [p for p in D["pts"] if drop_sticker is None or (p["cart"], p["id"]) != drop_sticker]
        for s in D["sides"]:
            if drop_sticker is not None and s["id"].startswith(f"marker_{drop_sticker[0]}_{drop_sticker[1]}_"):
                continue
            ml = model_line_3d(s["model_line"])
            out["L"].append(dict(id=s["id"], P=s["points"], cart=s["cart"], info=ml, cl=f"S{s['cart']}_{s['id'].split('_')[2]}"))
    if "E" in dataset:
        for e in D["edges"]:
            if e["id"] in drop_edges:
                continue
            ml = model_line_3d(e.get("model_line"))
            if dataset == "ME" and ml is not None and e.get("cart") in (80, 310):
                out["L"].append(dict(id=e["id"], P=e["points"], cart=e["cart"], info=ml, cl=e["id"]))
                continue
            g = vp_group(e, policy)
            if g is not None:
                out["V"].append(dict(id=e["id"], P=e["points"], cart=e.get("cart"), info=g, cl=e["id"]))
            else:
                out["S"].append(dict(id=e["id"], P=e["points"], cart=e.get("cart"), info=None, cl=e["id"]))
    return out


# =====================================================================================
# estimator
# =====================================================================================

class Est:
    BLK = ("M", "L", "V", "S")

    def __init__(self, lens: Lens, data, sig, subsample=2, carts=(80, 310)):
        self.lens = lens
        self.sig = dict(sig)
        self.mp = data["M"]
        self.muv = np.array([p["uv"] for p in self.mp], float).reshape(-1, 2)
        self.mX = np.array([p["X"] for p in self.mp], float).reshape(-1, 3)
        self.mcart = np.array([p["cart"] for p in self.mp], int)
        self.mcl = [f"S{p['cart']}_{p['id']}" for p in self.mp]
        # edges
        self.E = []
        for b in ("L", "V", "S"):
            for e in data[b]:
                self.E.append(dict(e, blk=b, Ps=np.asarray(e["P"], float)[::subsample]))
        self.nE = len(self.E)
        if self.nE:
            self.allp = np.vstack([e["Ps"] for e in self.E])
            self.cnt = np.array([len(e["Ps"]) for e in self.E])
            self.start = np.r_[0, np.cumsum(self.cnt)[:-1]]
            self.eid = np.repeat(np.arange(self.nE), self.cnt)
            self.eblk = np.array([e["blk"] for e in self.E])
        else:
            self.allp = np.zeros((0, 2))
            self.eid = np.zeros(0, int)
            self.eblk = np.zeros(0, str)
        # carts / pose layout
        need_t = {c for c in carts if (self.mcart == c).any() or any(e["blk"] == "L" and e["cart"] == c for e in self.E)}
        need_r = need_t | {e["info"][0] for e in self.E if e["blk"] == "V" and e["info"][0] != "wv"}
        self.carts = [c for c in carts if c in need_r]
        self.need_t = need_t
        self.has_wv = any(e["blk"] == "V" and e["info"][0] == "wv" for e in self.E)
        self.pnames = list(lens.names)
        self.poff = {}
        for c in self.carts:
            self.poff[c] = len(self.pnames)
            self.pnames += [f"r{c}_{i}" for i in range(3)] + ([f"t{c}_{i}" for i in range(3)] if c in need_t else [])
        if self.has_wv:
            self.pnames += ["wv_a", "wv_b"]
        self.n = len(self.pnames)
        self._cache = {}
        # per-edge static info
        self.Lidx = [k for k, e in enumerate(self.E) if e["blk"] == "L"]
        self.Vidx = [k for k, e in enumerate(self.E) if e["blk"] == "V"]
        self.Sidx = [k for k, e in enumerate(self.E) if e["blk"] == "S"]

    # ------------------------------------------------------------------------------
    def unpack(self, x):
        P = self.lens.P(x[: self.lens.ni])
        Rs, ts = {}, {}
        for c in self.carts:
            o = self.poff[c]
            Rs[c] = rodrigues(x[o:o + 3])
            ts[c] = x[o + 3:o + 6] if c in self.need_t else None
        wv = None
        if self.has_wv:
            a, b = x[-2], x[-1]
            wv = np.array([np.cos(a) * np.sin(b), np.sin(a) * np.sin(b), np.cos(b)])
        return P, Rs, ts, wv

    def _und(self, x):
        key = np.asarray(x[: self.lens.ni], float).tobytes()
        if key in self._cache:
            return self._cache[key]
        P = self.lens.P(x[: self.lens.ni])
        hh = 0.5
        U = self.lens.undist_px(self.allp, P)
        Ux = self.lens.undist_px(self.allp + [hh, 0.0], P)
        Uy = self.lens.undist_px(self.allp + [0.0, hh], P)
        J = np.stack([(Ux - U) / hh, (Uy - U) / hh], axis=2)  # N x 2 (U) x 2 (D)
        if len(self._cache) > 6:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = (U, J)
        return U, J

    def edge_lines(self, x, U=None):
        """Per-edge undistorted-image line (unit normal n, offset o: n.U + o = 0)."""
        P, Rs, ts, wv = self.unpack(x)
        if U is None:
            U, _ = self._und(x)
        K = self.lens.K(P)
        cnt = self.cnt
        c = np.add.reduceat(U, self.start, axis=0) / cnt[:, None]
        nrm = np.zeros((self.nE, 2))
        off = np.zeros(self.nE)
        if self.Sidx:
            d = U - c[self.eid]
            sxx = np.add.reduceat(d[:, 0] ** 2, self.start)
            syy = np.add.reduceat(d[:, 1] ** 2, self.start)
            sxy = np.add.reduceat(d[:, 0] * d[:, 1], self.start)
            th = 0.5 * np.arctan2(2 * sxy, sxx - syy)
            nS = np.column_stack([-np.sin(th), np.cos(th)])
            s = np.array(self.Sidx)
            nrm[s] = nS[s]
            off[s] = -np.sum(nS[s] * c[s], axis=1)
        for k in self.Vidx:
            ci, ax = self.E[k]["info"]
            dvec = wv if ci == "wv" else Rs[ci][:, ax]
            v = K @ dvec
            l = np.cross(v, np.r_[c[k], 1.0])
            nn = np.hypot(l[0], l[1])
            nrm[k] = l[:2] / nn
            off[k] = l[2] / nn
        for k in self.Lidx:
            A, Dd = self.E[k]["info"]
            ci = self.E[k]["cart"]
            a = K @ (Rs[ci] @ A + ts[ci])
            b = K @ (Rs[ci] @ Dd)
            l = np.cross(a, b)
            nn = np.hypot(l[0], l[1])
            nrm[k] = l[:2] / nn
            off[k] = l[2] / nn
        return nrm, off

    def blocks(self, x):
        P, Rs, ts, wv = self.unpack(x)
        out = {}
        if len(self.mp):
            pred = np.zeros_like(self.muv)
            for c in self.carts:
                s = self.mcart == c
                if s.any():
                    pred[s] = self.lens.project(self.mX[s] @ Rs[c].T + ts[c], P)
            out["M"] = (pred - self.muv).ravel()
        else:
            out["M"] = np.zeros(0)
        if self.nE:
            U, J = self._und(x)
            nrm, off = self.edge_lines(x, U)
            n_pt = nrm[self.eid]
            ru = np.sum(U * n_pt, axis=1) + off[self.eid]
            jt = np.einsum("nij,ni->nj", J, n_pt)
            rd = ru / np.maximum(np.hypot(jt[:, 0], jt[:, 1]), 1e-9)
            for b in ("L", "V", "S"):
                out[b] = rd[self.eblk[self.eid] == b]
            out["_edges"] = rd
        else:
            for b in ("L", "V", "S"):
                out[b] = np.zeros(0)
            out["_edges"] = np.zeros(0)
        return out

    def residuals(self, x):
        b = self.blocks(x)
        parts = []
        if len(b["M"]):
            parts.append(b["M"] / self.sig["M"])
        if self.nE:
            sg = {k: self.sig.get(k, 1.0) for k in ("L", "V", "S")}
            parts.append(b["_edges"] / np.array([sg[k] for k in self.eblk])[self.eid])
        return np.concatenate(parts) if parts else np.zeros(0)

    def clusters(self):
        """Cluster label per residual (sticker for corners, edge id for edge points)."""
        lab = []
        for c in self.mcl:
            lab += [c, c]
        lab += [self.E[k]["cl"] for k in self.eid]
        return np.array(lab)

    def block_of_resid(self):
        return np.array(["M"] * (2 * len(self.mp)) + list(self.eblk[self.eid]))

    # ------------------------------------------------------------------------------
    def x0(self, P, poses=None, wv=None):
        x = list(self.lens.pack(P))
        poses = P0 if poses is None else poses
        for c in self.carts:
            x += list(poses[c][:3])
            if c in self.need_t:
                x += list(poses[c][3:6])
        if self.has_wv:
            z = rodrigues(poses[80][:3])[:, 2] if wv is None else wv
            z = z / np.linalg.norm(z)
            x += [np.arctan2(z[1], z[0]), np.arccos(np.clip(z[2], -1, 1))]
        return np.array(x, float)

    def poses_of(self, x):
        out = {}
        for c in self.carts:
            o = self.poff[c]
            out[c] = np.r_[x[o:o + 3], x[o + 3:o + 6] if c in self.need_t else P0[c][3:]]
        return out

    def wv_of(self, x):
        if not self.has_wv:
            return None
        return self.unpack(x)[3]

    def solve(self, x0, max_nfev=400, loss="linear", f_scale=1.0):
        return least_squares(self.residuals, x0, method="trf", x_scale="jac", loss=loss, f_scale=f_scale,
                             max_nfev=max_nfev, xtol=1e-11, ftol=1e-11, gtol=1e-11)

    def varcomp(self, x0, iters=6, verbose=False):
        x = x0
        r = None
        for _ in range(iters):
            r = self.solve(x)
            x = r.x
            b = self.blocks(x)
            new = {k: max(float(np.sqrt(np.mean(b[k] ** 2))), 0.05) for k in self.BLK if len(b[k]) > 10}
            if verbose:
                print("   varcomp", {k: round(v, 3) for k, v in new.items()})
            ch = any(abs(new[k] / self.sig[k] - 1) > 0.02 for k in new)
            self.sig.update(new)
            if not ch:
                break
        return self.solve(x)

    def block_stats(self, x):
        b = self.blocks(x)
        out = {}
        for k in self.BLK:
            v = b[k]
            if not len(v):
                continue
            if k == "M":
                e = np.hypot(v[0::2], v[1::2])
                out[k] = dict(rms=float(np.sqrt(np.mean(v ** 2))), rms_radial=float(np.sqrt(np.mean(e ** 2))), max_radial=float(e.max()), n=int(len(v)))
            else:
                out[k] = dict(rms=float(np.sqrt(np.mean(v ** 2))), max=float(np.abs(v).max()), n=int(len(v)))
        return out


def sandwich_cov(res, clusters):
    J = res.jac
    e = res.fun
    A = J.T @ J
    Ai = np.linalg.pinv(A)
    meat = np.zeros_like(A)
    labs, inv = np.unique(clusters, return_inverse=True)
    S = np.zeros((len(labs), J.shape[1]))
    np.add.at(S, inv, J * e[:, None])
    meat = S.T @ S
    G = len(labs)
    return Ai @ meat @ Ai * G / max(G - 1, 1)


def naive_cov(res, sigma2=None):
    J = res.jac
    dof = max(len(res.fun) - len(res.x), 1)
    s2 = (2 * res.cost / dof) if sigma2 is None else sigma2
    return np.linalg.pinv(J.T @ J) * s2


# =====================================================================================
# mapping comparison between arbitrary lens models
# =====================================================================================

def pixel_grid(step=40, margin=0):
    xs = np.arange(margin, W - margin + 1e-9, step)
    ys = np.arange(margin, H - margin + 1e-9, step)
    X, Y = np.meshgrid(xs, ys)
    return np.column_stack([X.ravel(), Y.ravel()]).astype(float), X.shape


def mapping_disp(l1, P1, l2, P2, uv, compensate=True):
    """Displacement of lens 2 relative to lens 1 at pixels uv (rays of 1, rotated, projected by 2)."""
    r = l1.rays(uv, P1)
    if not compensate:
        return l2.project(r, P2) - uv

    def res(a):
        return (l2.project(r @ rodrigues(a).T, P2) - uv).ravel()

    a = least_squares(res, np.zeros(3), x_scale=1e-3, xtol=1e-12, ftol=1e-12).x
    return l2.project(r @ rodrigues(a).T, P2) - uv


def region_stats(disp, uv):
    from evaltools import region_defs

    m = np.hypot(disp[:, 0], disp[:, 1])
    out = {}
    for name, fn in region_defs().items():
        s = fn(uv)
        out[name] = dict(median=float(np.median(m[s])), rms=float(np.sqrt(np.mean(m[s] ** 2))), max=float(m[s].max()))
    out["whole_image"] = dict(median=float(np.median(m)), rms=float(np.sqrt(np.mean(m ** 2))), max=float(m.max()))
    return out


def convert(l_from, P_from, l_to, P_init, uv=None, rotate=False):
    """Fit the intrinsics of l_to to reproduce the mapping of l_from (same rays -> same pixels)."""
    if uv is None:
        uv, _ = pixel_grid(30)
    r = l_from.rays(uv, P_from)
    th0 = l_to.pack(P_init)

    def res(th):
        P = l_to.P(th[: l_to.ni])
        rr = r @ rodrigues(th[l_to.ni:]).T if rotate else r
        return (l_to.project(rr, P) - uv).ravel()

    x0 = np.r_[th0, np.zeros(3)] if rotate else th0
    q = least_squares(res, x0, x_scale="jac", method="trf", xtol=1e-12, ftol=1e-12, max_nfev=2000)
    return l_to.P(q.x[: l_to.ni]), float(np.sqrt(np.mean(q.fun ** 2) * 2))
