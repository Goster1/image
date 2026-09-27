"""Cuboid cart model with shelves (drawing dimensions exact) - helpers for the 41_cuboid_* scripts.

Contents
--------
* cart_lines()          : every modelled straight element of a cart (top-plate outline, 4 posts,
                          shelf front / back edges at the 5 shelf heights, floor outline) as 3D lines
                          in the cart frame, drawing dimensions only.
* EDGE_MAP              : which traced edge (edges_cart*.json) belongs to which modelled element,
                          with the natural "implied offset" coordinates to report.
* CuboidFit             : joint estimator (sticker corners + sticker sides + exact structure lines,
                          optional extra lines with nuisance offsets, optional DIAGNOSTIC geometry
                          parameters).  Line residuals are distances in the DISTORTED image
                          (r_u / |J^T n|, same convention as combined.py).  Blocks are weighted by
                          variance components; optionally each edge counts as n_eff observations
                          (cluster weighting) instead of one observation per traced point.
* line offsets          : signed px offset of traced points from the projected model line (in the
                          distorted image) and the implied 3D shift of the line (mm) with the camera
                          held fixed.
Conventions: cart frame of spec/cart-marker-layout.json (X along the 1600 mm front, Y into the cart,
Z up, Z = 0 top plate).  Poses: x_cam = R(rvec) X + t.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from calib import DNAMES
from common import (CACHE, CART_D, CART_W, FLOOR_Z, SHELF_Z, H, W, K_from, load_json, project, rodrigues,
                    undistort_points)
from markerdata import load_markers, marker_side_edges, to_points

EDGE_FILES = {310: f"{CACHE}/edges_cart310.json", 80: f"{CACHE}/edges_cart80.json"}
LEVEL_Z = {"top": 0.0, **{k: float(v) for k, v in SHELF_Z.items()}, "floor": float(FLOOR_Z)}


# ------------------------------------------------------------------------------------------------
# modelled cart lines (drawing only)
# ------------------------------------------------------------------------------------------------
def cart_lines():
    """dict name -> dict(A, D, t0, t1, level, kind). Line = A + t D, t in [t0, t1] (mm)."""
    L = {}
    ex, ey, ez = np.eye(3)
    # top plate outline (Z = 0)
    L["top_front"] = dict(A=np.array([0, 0, 0.0]), D=ex, t0=0, t1=CART_W, level="top", kind="top")
    L["top_back"] = dict(A=np.array([0, CART_D, 0.0]), D=ex, t0=0, t1=CART_W, level="top", kind="top")
    L["top_x0"] = dict(A=np.array([0, 0, 0.0]), D=ey, t0=0, t1=CART_D, level="top", kind="top")
    L["top_x1600"] = dict(A=np.array([CART_W, 0, 0.0]), D=ey, t0=0, t1=CART_D, level="top", kind="top")
    # corner posts
    for X in (0.0, CART_W):
        for Y in (0.0, CART_D):
            L[f"post_x{int(X)}_y{int(Y)}"] = dict(A=np.array([X, Y, 0.0]), D=ez, t0=FLOOR_Z, t1=0.0, level="post", kind="post")
    # shelf front / back edges
    for row, z in SHELF_Z.items():
        L[f"shelf{row}_front"] = dict(A=np.array([0, 0, z]), D=ex, t0=0, t1=CART_W, level=row, kind="shelf")
        L[f"shelf{row}_back"] = dict(A=np.array([0, CART_D, z]), D=ex, t0=0, t1=CART_W, level=row, kind="shelf")
    # floor outline (castor contact plane, Z = -1800)
    L["floor_front"] = dict(A=np.array([0, 0, FLOOR_Z]), D=ex, t0=0, t1=CART_W, level="floor", kind="floor")
    L["floor_back"] = dict(A=np.array([0, CART_D, FLOOR_Z]), D=ex, t0=0, t1=CART_W, level="floor", kind="floor")
    L["floor_x0"] = dict(A=np.array([0, 0, FLOOR_Z]), D=ey, t0=0, t1=CART_D, level="floor", kind="floor")
    L["floor_x1600"] = dict(A=np.array([CART_W, 0, FLOOR_Z]), D=ey, t0=0, t1=CART_D, level="floor", kind="floor")
    return L


# traced edge id -> (model line name, implied-offset axes to report, short physical label)
# axes: the coordinates (other than the free one) along which a pure shift of the line is reported.
EDGE_MAP = {
    # ---- cart 310
    "cart310_board0_outer": ("top_x0", "XZ", "top plate X=0 end edge (board)"),
    "cart310_board1600_outer": ("top_x1600", "XZ", "top plate X=1600 end edge (board)"),
    "cart310_x_board0_front": ("top_front", "YZ", "top plate front edge at X=0 board"),
    "cart310_x_board1600_front": ("top_front", "YZ", "top plate front edge at X=1600 board (lower edge of front face)"),
    "cart310_x_A_front_fall": ("shelfA_front", "ZY", "shelf A front lip"),
    "cart310_x_B_front_fall": ("shelfB_front", "ZY", "shelf B front lip"),
    "cart310_x_B_front_rise": ("shelfB_front", "ZY", "shelf B front lip (other side of the dark line)"),
    "cart310_x_C_front_fall": ("shelfC_front", "ZY", "shelf C front lip"),
    "cart310_x_C_front_rise": ("shelfC_front", "ZY", "shelf C front lip (other side of the dark line)"),
    "cart310_x_D_front_fall": ("shelfD_front", "ZY", "shelf D front lip"),
    "cart310_x_E_front_fall": ("shelfE_front", "ZY", "shelf E front lip"),
    "cart310_x_B_inner_fall": ("shelfB_front", "ZY", "shelf B front strip inner line (lip fold)"),
    "cart310_x_back_inner": ("top_back", "ZY", "back longitudinal member (identity uncertain)"),
    "cart310_z_post_front_left_right": ("post_x0_y0", "XY", "front post at X=0 (aisle silhouette)"),
    "cart310_z_post_front_right": ("post_x1600_y0", "XY", "front post at X=1600 (seen end-on)"),
    # ---- cart 80
    "cart80_x_board0_front": ("top_front", "YZ", "top plate front edge at X=0 board"),
    "cart80_x_board1600_front": ("top_front", "YZ", "top plate front edge at X=1600 board"),
    "cart80_y_board0_outer": ("top_x0", "XZ", "top plate X=0 end edge (board)"),
    "cart80_y_board1600_outer": ("top_x1600", "XZ", "top plate X=1600 end edge (board)"),
    "cart80_x_back_rail_out": ("top_back", "ZY", "back top rail, outer silhouette"),
    "cart80_x_back_rail_in": ("top_back", "ZY", "back top rail, inner boundary"),
    "cart80_x_back_low": ("top_back", "ZY", "back lower longitudinal member"),
    "cart80_x_A_front_out": ("shelfA_front", "ZY", "shelf A front lip (outer outline)"),
    "cart80_x_A_front_in": ("shelfA_front", "ZY", "shelf A front lip (inner boundary of lip top)"),
    "cart80_x_B_front": ("shelfB_front", "ZY", "shelf B front lip"),
    "cart80_x_C_front": ("shelfC_front", "ZY", "shelf C front lip"),
    "cart80_x_D_front": ("shelfD_front", "ZY", "shelf D front lip"),
    "cart80_x_E_front": ("shelfE_front", "ZY", "shelf E front lip (outline)"),
    "cart80_x_E_front_in": ("shelfE_front", "ZY", "shelf E front lip (inner boundary of lip top)"),
    "cart80_z_post1600_sil": ("post_x1600_y0", "XY", "front post at X=1600 (aisle silhouette, upper)"),
    "cart80_z_post1600_sil_low": ("post_x1600_y0", "XY", "front post at X=1600 (aisle silhouette, lower)"),
}


def load_cart_edges_raw():
    """All traced cart edges (both files), incl. non-straight / non-exact ones; points as arrays."""
    out = {}
    for cart, fn in EDGE_FILES.items():
        d = load_json(fn)
        for e in d["edges"]:
            e = dict(e)
            e["points"] = np.array(e["points"], float)
            e["cart"] = cart
            out[e["id"]] = e
    return out


def edges_review_state():
    return {c: ("review" in load_json(fn)) for c, fn in EDGE_FILES.items()}


def exact_structure_edges(raw=None, exclude=()):
    """Traced cart edges with model_line.exact == True (verified != 'no', straight_3d)."""
    raw = load_cart_edges_raw() if raw is None else raw
    out = []
    for eid, e in raw.items():
        ml = e.get("model_line") or {}
        if not ml.get("exact", False) or e.get("verified", "yes") == "no" or not e.get("straight_3d", False):
            continue
        if eid in exclude:
            continue
        A, D = ml_to_AD(ml)
        out.append(dict(id=eid, cart=e["cart"], points=e["points"], A=A, D=D, level="top" if abs(A[2]) < 1e-9 and D[2] == 0 else "?",
                        kind="exact_edge", what=e.get("what", "")))
    return out


def ml_to_AD(ml):
    A = np.zeros(3)
    for k, v in ml["fixed"].items():
        A["XYZ".index(k)] = float(v)
    D = np.zeros(3)
    D["XYZ".index(ml["free"])] = 1.0
    return A, D


def sticker_side_lines(markers=None):
    """Sub-pixel traced sides of partly hidden stickers (exact by the sticker geometry)."""
    M = load_markers() if markers is None else markers
    out = []
    for e in marker_side_edges(M, only_partial=True):
        A, D = ml_to_AD(e["model_line"])
        z = A[2]
        lvl = {0.0: "top", -195.0: "A", -595.0: "B", -995.0: "C"}.get(float(z), "?")
        out.append(dict(id=e["id"], cart=e["cart"], points=np.asarray(e["points"], float), A=A, D=D, level=lvl,
                        kind="sticker_side", what=e["what"]))
    return out


# ------------------------------------------------------------------------------------------------
# camera helpers
# ------------------------------------------------------------------------------------------------
class Camera:
    def __init__(self, K, dist, poses, name=""):
        self.K = np.asarray(K, float)
        d = np.zeros(8)
        dd = np.asarray(dist, float).ravel()
        d[: len(dd)] = dd
        self.dist = d
        self.poses = {int(c): np.asarray(p, float) for c, p in poses.items()}
        self.name = name

    def R(self, cart):
        return rodrigues(self.poses[cart][:3])

    def t(self, cart):
        return self.poses[cart][3:]

    def proj(self, X, cart):
        return project(X, self.K, self.dist, rvec=self.poses[cart][:3], tvec=self.poses[cart][3:])

    def centre(self, cart):
        return -self.R(cart).T @ self.t(cart)


def fit_poses(K, dist, pts, P0, carts=(80, 310)):
    """Poses per cart with the intrinsics held fixed (LM on all given corner points)."""
    out = {}
    for c in carts:
        X = np.array([p["X"] for p in pts if p["cart"] == c])
        uv = np.array([p["uv"] for p in pts if p["cart"] == c])

        def res(x):
            return (project(X, K, dist, rvec=x[:3], tvec=x[3:]) - uv).ravel()

        r = least_squares(res, np.asarray(P0[c], float), method="lm", x_scale="jac", xtol=1e-12, ftol=1e-12)
        out[c] = r.x
    return out


def markers_only_camera():
    """Marker-only fit of 20_markers_ba.py (method_markers.json): intrinsics + poses of its best model."""
    mm = load_json(f"{CACHE}/method_markers.json")
    s = mm["details"]["summary"]
    return Camera(np.array(mm["camera_matrix"]), mm["dist_coeffs"], {int(k): v for k, v in s["poses"].items()},
                  name="markers-only (" + mm["model"] + ")")


# ------------------------------------------------------------------------------------------------
# projection of model lines; offsets of traced edges
# ------------------------------------------------------------------------------------------------
def project_segment(cam, cart, A, D, t0, t1, n=600, extend=0.0):
    t = np.linspace(t0 - extend, t1 + extend, n)
    X = A[None] + t[:, None] * D[None]
    Xc = X @ cam.R(cart).T + cam.t(cart)
    ok = Xc[:, 2] > 50
    return cam.proj(X[ok], cart), t[ok]


def undist_px(P, K, d):
    n = undistort_points(P, K, d, iters=40)
    return np.column_stack([K[0, 0] * n[:, 0] + K[0, 2], K[1, 1] * n[:, 1] + K[1, 2]])


def line_residuals_px(P, K, d, R, t, A, D, jac=True):
    """Signed distance (distorted-image px) of traced points P to the projection of the 3D line (A, D).

    Sign convention: positive = to the side of the image-line normal nv (nv = l[:2]/|l[:2]|, with
    l = (K(RA+t)) x (KRD)); use `oriented_sign` to map to a physical direction.
    Returns r_d (N), nv (2,), and the undistorted points.
    """
    U = undist_px(P, K, d)
    a = K @ (R @ A + t)
    b = K @ (R @ D)
    l = np.cross(a, b)
    nrm = np.hypot(l[0], l[1])
    nv = l[:2] / nrm
    r = (U @ l[:2] + l[2]) / nrm
    if jac:
        hh = 0.5
        Jx = (undist_px(P + [hh, 0.0], K, d) - U) / hh
        Jy = (undist_px(P + [0.0, hh], K, d) - U) / hh
        s = np.hypot(Jx @ nv, Jy @ nv)
        r = r / np.maximum(s, 1e-9)
    return r, nv, U


def physical_normal_sign(cam, cart, A, D, P, axis_vec, step=5.0):
    """+1 if moving the 3D line by +step mm along axis_vec moves its image towards +nv (at the points P)."""
    R, t = cam.R(cart), cam.t(cart)
    r0, nv, _ = line_residuals_px(P, cam.K, cam.dist, R, t, A, D, jac=False)
    r1, _, _ = line_residuals_px(P, cam.K, cam.dist, R, t, A + step * axis_vec, D, jac=False)
    # the line moved by +step: residual of the same points changes by -(shift along nv)
    return -np.sign(np.mean(r1 - r0))


def implied_shift(cam, cart, A, D, P, axis_vec):
    """Best 1-D shift s (mm) of the 3D line along axis_vec so that it passes through the traced points
    (camera fixed); returns s, rms after the shift, px per mm (mean sensitivity at the points)."""
    R, t = cam.R(cart), cam.t(cart)

    def res(s):
        return line_residuals_px(P, cam.K, cam.dist, R, t, A + s[0] * axis_vec, D)[0]

    r0 = res([0.0])
    r1 = res([1.0])
    sens = float(np.mean(np.abs(r1 - r0)))
    rr = least_squares(res, [0.0], x_scale=[10.0])
    return float(rr.x[0]), float(np.sqrt(np.mean(rr.fun ** 2))), sens


def implied_shift_2d(cam, cart, A, D, P, ax1, ax2):
    """Joint 2-D shift (both non-free axes): returns (s1, s2), their Gauss-Newton sigmas (from the
    rms of the residual, points treated as independent - optimistic) and the correlation."""
    R, t = cam.R(cart), cam.t(cart)

    def res(s):
        return line_residuals_px(P, cam.K, cam.dist, R, t, A + s[0] * ax1 + s[1] * ax2, D)[0]

    rr = least_squares(res, [0.0, 0.0], x_scale=[10.0, 10.0])
    J = rr.jac
    dof = max(len(rr.fun) - 2, 1)
    s2 = 2 * rr.cost / dof
    try:
        C = np.linalg.inv(J.T @ J) * s2
    except np.linalg.LinAlgError:
        C = np.full((2, 2), np.nan)
    sd = np.sqrt(np.diag(C))
    return rr.x, sd, float(C[0, 1] / (sd[0] * sd[1])), float(np.sqrt(np.mean(rr.fun ** 2)))


# ------------------------------------------------------------------------------------------------
# joint estimator
# ------------------------------------------------------------------------------------------------
class CuboidFit:
    """Intrinsics + one pose per cart from sticker corners and exact 3D lines.

    lines: list of dicts(id, cart, points, A, D, level, kind[, nuis]) - `nuis` = list of
           (param_name, axis_vector) nuisance shifts of the 3D line (e.g. a common lip height).
    geo:   list of DIAGNOSTIC geometry parameter names; supported:
             'dz_top'        : Z shift of everything at the top plate (top stickers, top lines)
             'dz_<row>'      : Z shift of shelf row <row> (stickers on it and its lines)
             'dz_shelves'    : common Z shift of all shelf rows A..E (relative to the top)
    blocks: 'M' corners, 'S' sticker sides, 'E' exact structure edges, 'X' extra (nuisance) lines.
    edge weighting: weight_mode 'point' (every traced point = one observation, points subsampled)
           or 'cluster' (every line counts as n_eff observations).
    """

    def __init__(self, pts, lines, spec, geo=(), nuis=(), subsample=2, weight_mode="point", n_eff=4.0,
                 carts=(80, 310), sig=None):
        self.f_mode = spec.get("f", "single")
        self.pp_free = spec.get("pp", "free") == "free"
        self.dfree = list(spec.get("dist", ["k1", "k2"]))
        self.pp0 = np.array(spec.get("pp0", [(W - 1) / 2, (H - 1) / 2]), float)
        self.dist0 = np.zeros(8)
        if spec.get("dist0") is not None:
            self.dist0[: len(spec["dist0"])] = spec["dist0"]
        self.carts = list(carts)
        self.cidx = {c: i for i, c in enumerate(self.carts)}
        self.pts = [p for p in pts if p["cart"] in self.cidx]
        self.mX = np.array([p["X"] for p in self.pts], float).reshape(-1, 3)
        self.muv = np.array([p["uv"] for p in self.pts], float).reshape(-1, 2)
        self.mci = np.array([self.cidx[p["cart"]] for p in self.pts], int)
        self.mlevel = np.array([{0.0: "top", -195.0: "A", -595.0: "B", -995.0: "C"}.get(float(x[2]), "?") for x in self.mX])
        self.lines = []
        for ln in lines:
            if ln["cart"] not in self.cidx:
                continue
            P = np.asarray(ln["points"], float)[::subsample]
            blk = {"sticker_side": "S", "exact_edge": "E"}.get(ln.get("kind"), "X")
            self.lines.append(dict(ln, P=P, blk=blk))
        self.geo = list(geo)
        self.nuis = list(nuis)
        self.weight_mode = weight_mode
        self.n_eff = n_eff
        if self.lines:
            self.allp = np.vstack([ln["P"] for ln in self.lines])
            self.lidx = np.cumsum([0] + [len(ln["P"]) for ln in self.lines])
            w = []
            for ln in self.lines:
                n = len(ln["P"])
                w.append(np.full(n, np.sqrt(self.n_eff / n) if weight_mode == "cluster" else 1.0))
            self.lw = np.concatenate(w)
        self.inames = (["f"] if self.f_mode == "single" else ["fx", "fy"]) + (["cx", "cy"] if self.pp_free else []) + self.dfree
        self.ni = len(self.inames)
        self.names = self.inames + [f"pose{c}_{i}" for c in self.carts for i in range(6)] + self.geo + self.nuis
        self.sig = dict(M=1.0, S=0.5, E=0.5, X=1.0) if sig is None else dict(sig)

    # -------------------------------------------------------------
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
        j = self.ni + 6 * len(self.carts)
        g = dict(zip(self.geo, x[j:j + len(self.geo)]))
        nu = dict(zip(self.nuis, x[j + len(self.geo):]))
        return K_from(fx, fy, cx, cy), d, poses, g, nu

    def _dz(self, g, level):
        dz = 0.0
        if level == "top":
            dz += g.get("dz_top", 0.0)
        elif level in ("A", "B", "C", "D", "E"):
            dz += g.get(f"dz_{level}", 0.0) + g.get("dz_shelves", 0.0)
        return dz

    def camera(self, x):
        K, d, poses, _, _ = self.unpack(x)
        return Camera(K, d, {c: poses[i] for c, i in self.cidx.items()})

    def blocks(self, x, per_line=False):
        K, d, poses, g, nu = self.unpack(x)
        out = {"M": np.zeros(0), "S": [], "E": [], "X": []}
        wts = {"S": [], "E": [], "X": []}
        if len(self.pts):
            X = self.mX.copy()
            if self.geo:
                for lv in ("top", "A", "B", "C"):
                    X[self.mlevel == lv, 2] += self._dz(g, lv)
            pred = np.zeros_like(self.muv)
            for i in range(len(self.carts)):
                s = self.mci == i
                if s.any():
                    pred[s] = project(X[s], K, d, rvec=poses[i, :3], tvec=poses[i, 3:])
            out["M"] = (pred - self.muv).ravel()
        perl = []
        if self.lines:
            U = undist_px(self.allp, K, d)
            hh = 0.5
            Jx = (undist_px(self.allp + [hh, 0.0], K, d) - U) / hh
            Jy = (undist_px(self.allp + [0.0, hh], K, d) - U) / hh
            Rs = [rodrigues(p[:3]) for p in poses]
            for k, ln in enumerate(self.lines):
                sl = slice(self.lidx[k], self.lidx[k + 1])
                ci = self.cidx[ln["cart"]]
                A = ln["A"].copy()
                A[2] += self._dz(g, ln.get("level", "?")) if self.geo else 0.0
                for nm, vec in ln.get("nuis", []):
                    A = A + nu.get(nm, 0.0) * np.asarray(vec, float)
                a = K @ (Rs[ci] @ A + poses[ci, 3:])
                b = K @ (Rs[ci] @ ln["D"])
                l = np.cross(a, b)
                nrm = np.hypot(l[0], l[1])
                nv = l[:2] / nrm
                r = (U[sl] @ l[:2] + l[2]) / nrm
                s = np.hypot(Jx[sl] @ nv, Jy[sl] @ nv)
                r = r / np.maximum(s, 1e-9)
                out[ln["blk"]].append(r)
                wts[ln["blk"]].append(self.lw[sl])
                perl.append(r)
        for b in ("S", "E", "X"):
            out[b] = np.concatenate(out[b]) if len(out[b]) else np.zeros(0)
            wts[b] = np.concatenate(wts[b]) if len(wts[b]) else np.zeros(0)
        if per_line:
            return out, wts, perl
        return out, wts

    def residuals(self, x):
        b, w = self.blocks(x)
        parts = [b["M"] / self.sig["M"]]
        for k in ("S", "E", "X"):
            if len(b[k]):
                parts.append(b[k] * w[k] / self.sig[k])
        return np.concatenate(parts)

    def x0(self, K, dist, poses, geo0=None, nuis0=None):
        x = [K[0, 0]] if self.f_mode == "single" else [K[0, 0], K[1, 1]]
        if self.pp_free:
            x += [K[0, 2], K[1, 2]]
        dd = np.zeros(8)
        dd[: len(dist)] = np.asarray(dist).ravel()
        x += [dd[DNAMES.index(nm)] for nm in self.dfree]
        for c in self.carts:
            x += list(poses[c])
        x += [0.0 if geo0 is None else geo0.get(nm, 0.0) for nm in self.geo]
        x += [0.0 if nuis0 is None else nuis0.get(nm, 0.0) for nm in self.nuis]
        return np.array(x, float)

    def block_rms(self, x):
        b, w = self.blocks(x)
        out = {}
        if len(b["M"]):
            r = b["M"].reshape(-1, 2)
            out["M"] = dict(rms_coord=float(np.sqrt(np.mean(r ** 2))), rms_point=float(np.sqrt(np.mean(np.sum(r ** 2, 1)))),
                            max_point=float(np.max(np.hypot(r[:, 0], r[:, 1]))), n=len(r))
        for k in ("S", "E", "X"):
            if len(b[k]):
                ww = w[k] ** 2
                out[k] = dict(rms=float(np.sqrt(np.mean(b[k] ** 2))), rms_w=float(np.sqrt(np.sum(ww * b[k] ** 2) / np.sum(ww))),
                              max=float(np.max(np.abs(b[k]))), n=len(b[k]))
        return out

    def solve(self, x0, reweight=True, iters=6, loss="linear", f_scale=3.0, fixed_sig=None, max_nfev=400):
        x = np.asarray(x0, float)
        if fixed_sig is not None:
            self.sig.update(fixed_sig)
        r = None
        for it in range(iters if reweight else 1):
            r = least_squares(self.residuals, x, method="trf", loss=loss, f_scale=f_scale, x_scale="jac",
                              max_nfev=max_nfev, xtol=1e-10, ftol=1e-10)
            x = r.x
            if not reweight:
                break
            b, w = self.blocks(x)
            new = {}
            if len(b["M"]) > 10:
                new["M"] = max(float(np.sqrt(np.mean(b["M"] ** 2))), 0.05)
            for k in ("S", "E", "X"):
                if len(b[k]) > 5:
                    ww = w[k] ** 2
                    new[k] = max(float(np.sqrt(np.sum(ww * b[k] ** 2) / np.sum(ww))), 0.05)
            changed = any(abs(new[k] / self.sig[k] - 1) > 0.03 for k in new)
            self.sig.update(new)
            if not changed:
                break
        return r


def cov_from(res):
    J = res.jac
    dof = max(J.shape[0] - J.shape[1], 1)
    s2 = 2 * res.cost / dof
    C = np.linalg.pinv(J.T @ J) * s2
    return C


def base_data():
    """Sticker corners, sticker sides, exact structure edges, initial poses."""
    M = load_markers()
    pts = to_points(M)
    sides = sticker_side_lines(M)
    exact = exact_structure_edges()
    init = load_json(f"{CACHE}/initial_calib.json")
    P0 = {80: np.array(init["poses"]["80"]), 310: np.array(init["poses"]["310"])}
    return M, pts, sides, exact, P0
