"""Joint estimator: sticker corners + edges (exact cuboid lines, VP direction groups, straightness).

Parameters: intrinsics (f or fx,fy; cx,cy; distortion subset) + 6-DoF pose per cart
            (+ free direction of the scene verticals unless tied to a cart).
Residual blocks (all in px, each divided by its own sigma):
  M  marker corners               : reprojection error
  L  exact model lines             : distance of undistorted edge points to the pinhole projection of the
                                     known 3D cart line (cart frame, drawing dimensions)
  V  direction groups (not exact)  : distance of undistorted edge points to the line through the group's
                                     vanishing point and the edge centroid
  S  other straight edges          : straightness (own TLS line)
Block sigmas can be re-estimated iteratively (variance components) so that every block has
reduced chi2 ~ 1; this is the default weighting.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from calib import DNAMES
from common import H, W, K_from, distort_normalized, fold_margin, project, rodrigues, undistort_points

AXIS = {"cartX": 0, "cartY": 1, "cartZ": 2}


def model_line_3d(ml):
    """model_line dict -> (A, D) in cart frame (point, unit direction) or None."""
    if not ml or not ml.get("exact", False):
        return None
    fixed = ml["fixed"]
    free = ml["free"]
    A = np.zeros(3)
    for k, v in fixed.items():
        A["XYZ".index(k)] = float(v)
    D = np.zeros(3)
    D["XYZ".index(free)] = 1.0
    return A, D


class Combined:
    def __init__(self, markers_pts, edges, spec, use=("M", "L", "V", "S"), sig=None, subsample=2, tie_wv=None,
                 carts=(80, 310), common_vertical=False):
        self.spec = spec
        self.f_mode = spec.get("f", "single")
        self.pp_free = spec.get("pp", "free") == "free"
        self.dfree = list(spec.get("dist", ["k1", "k2"]))
        self.pp0 = np.array(spec.get("pp0", [(W - 1) / 2, (H - 1) / 2]))
        self.dist0 = np.zeros(8)
        if spec.get("dist0") is not None:
            self.dist0[: len(spec["dist0"])] = spec["dist0"]
        self.carts = list(carts)
        self.cidx = {c: i for i, c in enumerate(self.carts)}
        self.use = set(use)
        self.sig = dict(M=0.5, L=0.5, V=0.5, S=0.3) if sig is None else dict(sig)
        # markers
        self.mp = [p for p in markers_pts if p["cart"] in self.cidx] if "M" in self.use else []
        self.mX = np.array([p["X"] for p in self.mp]).reshape(-1, 3)
        self.muv = np.array([p["uv"] for p in self.mp]).reshape(-1, 2)
        self.mci = np.array([self.cidx[p["cart"]] for p in self.mp], int)
        self.mgrp = np.array([p.get("mi", -1) for p in self.mp], int)  # sticker index per point
        self.mw = np.ones(len(self.mp))  # per-point weights (IRLS over stickers), default 1
        # edges -> blocks
        self.E = []
        for e in edges:
            P = e["points"][::subsample]
            ml = model_line_3d(e.get("model_line"))
            blk = None
            vg = e.get("vp_group", "__none__")
            if ml is not None and e.get("cart") in self.cidx and "L" in self.use:
                blk = ("L", ml)
            elif vg is None:  # explicit: straightness only
                blk = ("S", None) if "S" in self.use else None
            elif isinstance(vg, str) and vg.startswith("floor_") and "V" in self.use:
                blk = ("V", (vg, -1))
            elif e.get("direction") in AXIS and e.get("cart") in self.cidx and "V" in self.use:
                blk = ("V", (e["cart"], AXIS[e["direction"]]))
            elif e.get("direction") == "world_vertical" and "V" in self.use:
                blk = ("V", ("wv", 2))
            elif "S" in self.use:
                blk = ("S", None)
            if blk is None:
                continue
            self.E.append(dict(id=e["id"], P=P, blk=blk[0], info=blk[1], cart=e.get("cart"), chord=np.hypot(*(P[-1] - P[0])),
                               grp=int(e.get("sticker_mi", -1)), w=1.0))
        self.tie_wv = tie_wv
        # common_vertical: the Z (post) edges of both carts share the scene vertical wv (carts upright on one floor)
        self.common_vertical = bool(common_vertical)
        self.floor = sorted({e["info"][0] for e in self.E if e["blk"] == "V" and isinstance(e["info"][0], str) and e["info"][0].startswith("floor_")})
        self.has_wv = (any(e["blk"] == "V" and e["info"][0] == "wv" for e in self.E) or len(self.floor) > 0 or self.common_vertical) and tie_wv is None
        if self.E:
            self.allp = np.vstack([e["P"] for e in self.E])
            self.eidx = np.cumsum([0] + [len(e["P"]) for e in self.E])
        # parameter names
        self.inames = (["f"] if self.f_mode == "single" else ["fx", "fy"]) + (["cx", "cy"] if self.pp_free else []) + self.dfree
        self.ni = len(self.inames)
        self.names = self.inames + [f"pose{c}_{i}" for c in self.carts for i in range(6)] + (["wv_a", "wv_b"] if self.has_wv else []) + [f"{g}_ang" for g in self.floor]

    # ---------------------------------------------------------------------
    def unpack(self, x):
        i = 0
        if self.f_mode == "single":
            fx = fy = x[0]
            i = 1
        else:
            fx, fy = x[0], x[1]
            i = 2
        if self.pp_free:
            cx, cy = x[i], x[i + 1]
            i += 2
        else:
            cx, cy = self.pp0
        d = self.dist0.copy()
        for nm in self.dfree:
            d[DNAMES.index(nm)] = x[i]
            i += 1
        poses = x[self.ni:self.ni + 6 * len(self.carts)].reshape(-1, 6)
        wv = None
        j = self.ni + 6 * len(self.carts)
        if self.has_wv:
            a, b = x[j], x[j + 1]
            wv = np.array([np.cos(a) * np.sin(b), np.sin(a) * np.sin(b), np.cos(b)])
            j += 2
        self._floor_dirs = {}
        if self.floor:
            v = rodrigues(poses[self.cidx[self.tie_wv], :3])[:, 2] if self.tie_wv is not None else wv
            e1 = np.cross(v, [1.0, 0.0, 0.0])
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(v, e1)
            for g in self.floor:
                self._floor_dirs[g] = np.cos(x[j]) * e1 + np.sin(x[j]) * e2
                j += 1
        return K_from(fx, fy, cx, cy), d, poses, wv

    def blocks(self, x):
        K, d, poses, wv = self.unpack(x)
        out = {"M": np.zeros(0), "L": [], "V": [], "S": []}
        if len(self.mp):
            pred = np.zeros_like(self.muv)
            for i in range(len(self.carts)):
                s = self.mci == i
                if s.any():
                    pred[s] = project(self.mX[s], K, d, rvec=poses[i, :3], tvec=poses[i, 3:])
            out["M"] = ((pred - self.muv) * np.sqrt(self.mw)[:, None]).ravel()
        if self.E:
            def und(p):
                n = undistort_points(p, K, d, iters=30)
                return np.column_stack([K[0, 0] * n[:, 0] + K[0, 1] * n[:, 1] + K[0, 2], K[1, 1] * n[:, 1] + K[1, 2]])

            U = und(self.allp)
            Rs = [rodrigues(p[:3]) for p in poses]
            for k, e in enumerate(self.E):
                sl = slice(self.eidx[k], self.eidx[k + 1])
                P = U[sl]
                c = P.mean(0)
                if e["blk"] == "S":
                    u, s, vt = np.linalg.svd(P - c, full_matrices=False)
                    r = (P - c) @ vt[1]
                    nv = vt[1]
                elif e["blk"] == "V":
                    ci, ax = e["info"]
                    if isinstance(ci, str) and ci.startswith("floor_"):
                        dvec = self._floor_dirs[ci]
                    elif ci == "wv":
                        dvec = Rs[self.cidx[self.tie_wv]][:, 2] if self.tie_wv is not None else wv
                    elif self.common_vertical and ax == 2:
                        dvec = wv
                    else:
                        dvec = Rs[self.cidx[ci]][:, ax]
                    v = K @ dvec
                    l = np.cross(v, np.r_[c, 1.0])
                    r = (P @ l[:2] + l[2]) / np.hypot(l[0], l[1])
                    nv = l[:2] / np.hypot(l[0], l[1])
                else:  # L
                    A, D = e["info"]
                    ci = self.cidx[e["cart"]]
                    R = Rs[ci]
                    t = poses[ci, 3:]
                    a = K @ (R @ A + t)
                    b = K @ (R @ D)
                    l = np.cross(a, b)
                    r = (P @ l[:2] + l[2]) / np.hypot(l[0], l[1])
                    nv = l[:2] / np.hypot(l[0], l[1])
                # forward residual in the DISTORTED image: observed point vs distorted image of its projection
                # onto the (undistorted) line; fold-safe, no noise-shrinkage bias
                q = P - r[:, None] * nv[None, :]
                xn = (q[:, 0] - K[0, 2]) / K[0, 0]
                yn = (q[:, 1] - K[1, 2]) / K[1, 1]
                xd, yd = distort_normalized(xn, yn, d)
                ee = self.allp[sl] - np.column_stack([K[0, 0] * xd + K[0, 2], K[1, 1] * yd + K[1, 2]])
                out[e["blk"]].append(np.sign(r) * np.hypot(ee[:, 0], ee[:, 1]) * np.sqrt(e["w"]))
        for b in ("L", "V", "S"):
            out[b] = np.concatenate(out[b]) if len(out[b]) else np.zeros(0)
        return out

    def residuals(self, x):
        b = self.blocks(x)
        K, d, _, _ = self.unpack(x)
        barrier = np.array([1000.0 * max(0.0, 0.03 - fold_margin(K, d))]) if self.E else np.zeros(0)
        return np.concatenate([b[k] / self.sig[k] for k in ("M", "L", "V", "S") if len(b[k])] + [barrier])

    def x0(self, K, dist, poses, wv=None):
        x = [K[0, 0]] if self.f_mode == "single" else [K[0, 0], K[1, 1]]
        if self.pp_free:
            x += [K[0, 2], K[1, 2]]
        dd = np.zeros(8)
        dd[: len(dist)] = dist
        x += [dd[DNAMES.index(nm)] for nm in self.dfree]
        for c in self.carts:
            x += list(poses[c])
        if self.has_wv:
            z = rodrigues(poses[self.carts[0]][:3])[:, 2] if wv is None else wv
            x += [np.arctan2(z[1], z[0]), np.arccos(np.clip(z[2], -1, 1))]
        x += [0.0] * len(self.floor)
        x = np.array(x, float)
        n0 = len(x) - len(self.floor)
        for j, g in enumerate(self.floor):  # grid init of each floor direction angle
            best = None
            for a in np.linspace(0, np.pi, 91):
                x[n0 + j] = a
                v = self.blocks(x)["V"]
                c = float(np.sum(v ** 2))
                if best is None or c < best[0]:
                    best = (c, a)
            x[n0 + j] = best[1]
        return x

    def solve(self, x0, loss="linear", f_scale=3.0, reweight=True, iters=40, vc_tol=0.005, verbose=False):
        """Least squares with block weights by variance components, iterated to convergence (every block RMS within
        vc_tol of its sigma); self.vc_iters records the number of iterations used."""
        x = x0
        r = None
        for it in range(iters if reweight else 1):
            r = least_squares(self.residuals, x, method="trf", loss=loss, f_scale=f_scale, x_scale="jac",
                              max_nfev=300, xtol=1e-10, ftol=1e-10)
            x = r.x
            if not reweight:
                break
            b = self.blocks(x)
            new = {}
            for k in ("M", "L", "V", "S"):
                if len(b[k]) > 10:
                    new[k] = max(float(np.sqrt(np.mean(b[k] ** 2))), 0.05)
            if verbose:
                print("  block rms:", {k: round(v, 3) for k, v in new.items()})
            changed = any(abs(new[k] / self.sig[k] - 1) > vc_tol for k in new)
            self.sig.update(new)
            self.vc_iters = it + 1
            if not changed:
                break
        return r

    def sticker_rms(self, x):
        """Unweighted RMS error per sticker group (corners [px, 2D] + its traced sides [px, normal])."""
        w = self.mw
        ew = [e["w"] for e in self.E]
        self.mw = np.ones(len(self.mp))
        for e in self.E:
            e["w"] = 1.0
        b = self.blocks(x)
        self.mw = w
        for e, v in zip(self.E, ew):
            e["w"] = v
        r = b["M"].reshape(-1, 2)
        sq, n = {}, {}
        for g in np.unique(self.mgrp):
            s = self.mgrp == g
            sq[int(g)] = float(np.sum(r[s] ** 2))
            n[int(g)] = int(s.sum())
        # L residuals of sticker sides: recompute per edge from block order (L edges in E order)
        Lr = b["L"]
        k = 0
        for e in self.E:
            if e["blk"] != "L":
                continue
            m = len(e["P"])
            if e["grp"] >= 0:
                g = e["grp"]
                sq[g] = sq.get(g, 0.0) + 2 * float(np.sum(Lr[k:k + m] ** 2))  # 1D residual ~ half of a 2D one
                n[g] = n.get(g, 0) + m
            k += m
        return {g: float(np.sqrt(sq[g] / max(n[g], 1))) for g in sq}

    def solve_irls(self, x0, c=1.5, iters=6, verbose=False, **kw):
        """Cluster-robust fit: sticker weight w = 1/(1+(rms_sticker/c)^2) (Cauchy on the sticker RMS)."""
        r = self.solve(x0, **kw)
        for it in range(iters):
            sr = self.sticker_rms(r.x)
            w = {g: 1.0 / (1.0 + (v / c) ** 2) for g, v in sr.items()}
            new = np.array([w[g] for g in self.mgrp])
            if verbose:
                print("  irls", it, {g: round(v, 2) for g, v in sr.items()})
            old_e = np.array([e["w"] for e in self.E]) if self.E else np.zeros(0)
            new_e = np.array([w.get(e["grp"], 1.0) if e["grp"] >= 0 else 1.0 for e in self.E]) if self.E else np.zeros(0)
            if np.max(np.abs(new - self.mw), initial=0) < 0.01 and np.max(np.abs(new_e - old_e), initial=0) < 0.01:
                break
            self.mw = new
            for e, v in zip(self.E, new_e):
                e["w"] = float(v)
            r = self.solve(r.x, **kw)
        return r

    def block_rms(self, x):
        b = self.blocks(x)
        return {k: (float(np.sqrt(np.mean(v ** 2))) if len(v) else None, len(v)) for k, v in b.items()}
