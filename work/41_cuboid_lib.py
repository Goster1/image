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


def foot_params(P, K, d, R, t, A, D):
    """Line parameter t_i (mm) of the point of the 3D line A + t D closest to the viewing ray of
    each image point P_i (undistorted with (K, d)); camera frame x = R X + t."""
    n = undistort_points(P, K, d, iters=40)
    r = np.column_stack([n, np.ones(len(n))])
    r /= np.linalg.norm(r, axis=1, keepdims=True)
    Ac = R @ A + t
    Dc = R @ D
    # minimise |Ac + s Dc - u r| over s, u  (per point)
    b = r @ Dc
    dd = r @ Ac
    e = Dc @ Ac
    den = 1.0 - b * b
    s = (b * dd - e) / np.maximum(den, 1e-12)
    return s


def tangent_residuals(P, s, K, d, R, t, A, D, h=5.0):
    """Signed distance (distorted-image px) of the points P from the projected 3D line, evaluated at the
    fixed line parameters s (foot points): r = (P - p(s)) . n,  n = left normal of the local tangent.
    Accurate to second order in the foot-point error (projected-line curvature is tiny)."""
    X0 = A[None] + s[:, None] * D[None]
    p0 = project(X0, K, d, R=R, tvec=t)
    p1 = project(X0 + h * D[None], K, d, R=R, tvec=t)
    tg = p1 - p0
    tg /= np.linalg.norm(tg, axis=1, keepdims=True)
    nv = np.column_stack([-tg[:, 1], tg[:, 0]])
    return np.sum((P - p0) * nv, axis=1), nv


def line_offsets(cam, cart, A, D, P):
    """Signed px distances of P from the projected line (A, D) with camera `cam` (+ normals)."""
    R, t = cam.R(cart), cam.t(cart)
    s = foot_params(P, cam.K, cam.dist, R, t, A, D)
    return tangent_residuals(P, s, cam.K, cam.dist, R, t, A, D)


def physical_sign(cam, cart, A, D, P, axis_vec, step=5.0):
    """+1 if the points' residual DEcreases when the 3D line is moved by +step along axis_vec, i.e. if a
    positive residual means 'the traced edge lies on the +axis side of the predicted line'."""
    R, t = cam.R(cart), cam.t(cart)
    s = foot_params(P, cam.K, cam.dist, R, t, A, D)
    r0, _ = tangent_residuals(P, s, cam.K, cam.dist, R, t, A, D)
    r1, _ = tangent_residuals(P, s, cam.K, cam.dist, R, t, A + step * axis_vec, D)
    return float(np.sign(np.mean(r0 - r1)))


def implied_shift(cam, cart, A, D, P, axis_vec):
    """Best 1-D shift s (mm) of the 3D line along axis_vec so that it passes through the traced points
    (camera fixed). Returns shift, rms after the shift [px], sensitivity [px/mm] (mean at the points)."""
    R, t = cam.R(cart), cam.t(cart)
    s = foot_params(P, cam.K, cam.dist, R, t, A, D)
    r0, _ = tangent_residuals(P, s, cam.K, cam.dist, R, t, A, D)
    r1, _ = tangent_residuals(P, s, cam.K, cam.dist, R, t, A + axis_vec, D)
    g = r1 - r0  # linear in the shift (to high accuracy)
    k = float(-np.sum(g * r0) / np.sum(g * g))
    rr = r0 + k * g
    return k, float(np.sqrt(np.mean(rr ** 2))), float(np.mean(np.abs(g)))


def implied_shift_2d(cam, cart, A, D, P, ax1, ax2):
    """Joint 2-D shift along both non-free axes (linearised, exact to high accuracy): values, formal
    sigmas (points treated as independent, scaled by the residual rms - optimistic), correlation, rms."""
    R, t = cam.R(cart), cam.t(cart)
    s = foot_params(P, cam.K, cam.dist, R, t, A, D)
    r0, _ = tangent_residuals(P, s, cam.K, cam.dist, R, t, A, D)
    g1 = tangent_residuals(P, s, cam.K, cam.dist, R, t, A + ax1, D)[0] - r0
    g2 = tangent_residuals(P, s, cam.K, cam.dist, R, t, A + ax2, D)[0] - r0
    J = np.column_stack([g1, g2])
    k, *_ = np.linalg.lstsq(J, -r0, rcond=None)
    rr = r0 + J @ k
    dof = max(len(rr) - 2, 1)
    C = np.linalg.pinv(J.T @ J) * np.sum(rr ** 2) / dof
    sd = np.sqrt(np.diag(C))
    return k, sd, float(C[0, 1] / (sd[0] * sd[1])), float(np.sqrt(np.mean(rr ** 2)))


# ------------------------------------------------------------------------------------------------
# joint estimator
# ------------------------------------------------------------------------------------------------
class CuboidFit:
    """Intrinsics + one pose per cart from sticker corners and exact 3D lines.

    lines: list of dicts(id, cart, points, A, D, level, kind[, nuis]) - `nuis` = list of
           (param_name, axis_vector) nuisance shifts of the 3D line (e.g. a common lip offset).
    geo:   list of DIAGNOSTIC geometry parameter names (never used in a main estimate):
             'dz_top'        : Z shift of everything at the top plate (top stickers, top lines)
             'dz_<row>'      : Z shift of shelf row <row> (stickers on it and its lines)
             'dz_shelves'    : common Z shift of all shelf rows A..E (relative to the top)
             'sz_shelves'    : relative scale of the shelf depths below the top (Z -> Z (1 + s/1000))
             'code_mm'       : sticker code size (default 90) - changes the corner positions
             'sx', 'sy'      : relative scale (1/1000) of the cart width / depth (sticker centres and lines)
    blocks: 'M' corners, 'S' sticker sides, 'E' exact structure edges, 'X' extra (nuisance) lines.
    Line residual = signed distance in the DISTORTED image to the projected 3D line (local tangent at
    the foot point; the foot points are updated between the outer iterations).
    edge weighting: weight_mode 'point' (every traced point = one observation) or 'cluster' (every line
    counts as n_eff observations, n_eff = clip(chord / corr_px, 2, 6)).
    """

    def __init__(self, pts, lines, spec, geo=(), nuis=(), subsample=2, weight_mode="cluster", corr_px=25.0,
                 carts=(80, 310), sig=None, markers=None):
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
        # sticker centre per corner (for a code-size diagnostic)
        from common import marker_center
        self.mC = np.array([marker_center(p["id"], p["cart"]) for p in self.pts], float).reshape(-1, 3) if self.pts else np.zeros((0, 3))
        self.lines = []
        for ln in lines:
            if ln["cart"] not in self.cidx:
                continue
            P = np.asarray(ln["points"], float)[::subsample]
            blk = {"sticker_side": "S", "exact_edge": "E"}.get(ln.get("kind"), "X")
            chord = float(np.hypot(*(P[-1] - P[0])))
            self.lines.append(dict(ln, P=P, blk=blk, chord=chord, s=None))
        self.geo = list(geo)
        self.nuis = list(nuis)
        self.weight_mode = weight_mode
        for ln in self.lines:
            n = len(ln["P"])
            neff = float(np.clip(ln["chord"] / corr_px, 2.0, 6.0))
            ln["neff"] = neff
            ln["w"] = np.sqrt(neff / n) if weight_mode == "cluster" else 1.0
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

    def _zmap(self, g, Z, level, cart=None):
        """Diagnostic height changes (drawing: identity). Keys may be cart specific: 'dz_top@80'."""
        if not self.geo:
            return Z
        Z = np.array(Z, float)

        def gg(k):
            return g.get(k, 0.0) + (g.get(f"{k}@{cart}", 0.0) if cart is not None else 0.0)

        if level == "top":
            Z = Z + gg("dz_top")
        elif level in ("A", "B", "C", "D", "E"):
            Z = Z * (1.0 + gg("sz_shelves") / 1000.0) + gg(f"dz_{level}") + gg("dz_shelves")
        return Z

    def camera(self, x):
        K, d, poses, _, _ = self.unpack(x)
        return Camera(K, d, {c: poses[i] for c, i in self.cidx.items()})

    def _line_AD(self, ln, g, nu):
        A = np.array(ln["A"], float).copy()
        if self.geo:
            A[2] = float(self._zmap(g, A[2], ln.get("level", "?"), ln["cart"]))
            A[0] *= 1.0 + g.get("sx", 0.0) / 1000.0
            A[1] *= 1.0 + g.get("sy", 0.0) / 1000.0
        for nm, vec in ln.get("nuis", []):
            A = A + nu.get(nm, 0.0) * np.asarray(vec, float)
        return A, np.asarray(ln["D"], float)

    def update_feet(self, x):
        K, d, poses, g, nu = self.unpack(x)
        for ln in self.lines:
            ci = self.cidx[ln["cart"]]
            A, D = self._line_AD(ln, g, nu)
            ln["s"] = foot_params(ln["P"], K, d, rodrigues(poses[ci, :3]), poses[ci, 3:], A, D)

    def marker_pred(self, x):
        K, d, poses, g, nu = self.unpack(x)
        X = self.mX.copy()
        if self.geo:
            if "code_mm" in g:
                X = self.mC + (X - self.mC) * (g["code_mm"] / 90.0)
            if "sx" in g or "sy" in g:
                X[:, 0] += self.mC[:, 0] * g.get("sx", 0.0) / 1000.0
                X[:, 1] += self.mC[:, 1] * g.get("sy", 0.0) / 1000.0
            for lv in ("top", "A", "B", "C"):
                for i, c in enumerate(self.carts):
                    m = (self.mlevel == lv) & (self.mci == i)
                    if m.any():
                        X[m, 2] = self._zmap(g, X[m, 2], lv, c)
        pred = np.zeros_like(self.muv)
        for i in range(len(self.carts)):
            s = self.mci == i
            if s.any():
                pred[s] = project(X[s], K, d, rvec=poses[i, :3], tvec=poses[i, 3:])
        return pred

    def blocks(self, x, per_line=False):
        K, d, poses, g, nu = self.unpack(x)
        out = {"M": np.zeros(0), "S": [], "E": [], "X": []}
        wts = {"S": [], "E": [], "X": []}
        if len(self.pts):
            out["M"] = (self.marker_pred(x) - self.muv).ravel()
        perl = []
        if self.lines:
            if self.lines[0]["s"] is None:
                self.update_feet(x)
            # vectorised over all lines of a cart: foot points and tangents projected in one call
            AD = [self._line_AD(ln, g, nu) for ln in self.lines]
            res_l = [None] * len(self.lines)
            hstep = 5.0
            for ci, c in enumerate(self.carts):
                ids = [k for k, ln in enumerate(self.lines) if ln["cart"] == c]
                if not ids:
                    continue
                X0 = np.vstack([AD[k][0][None] + self.lines[k]["s"][:, None] * AD[k][1][None] for k in ids])
                Dr = np.vstack([np.repeat(AD[k][1][None], len(self.lines[k]["s"]), axis=0) for k in ids])
                R = rodrigues(poses[ci, :3])
                pp = project(np.vstack([X0, X0 + hstep * Dr]), K, d, R=R, tvec=poses[ci, 3:])
                n = len(X0)
                p0, p1 = pp[:n], pp[n:]
                tg = p1 - p0
                tg /= np.linalg.norm(tg, axis=1, keepdims=True)
                P = np.vstack([self.lines[k]["P"] for k in ids])
                r_all = (P[:, 0] - p0[:, 0]) * (-tg[:, 1]) + (P[:, 1] - p0[:, 1]) * tg[:, 0]
                o = 0
                for k in ids:
                    m = len(self.lines[k]["s"])
                    res_l[k] = r_all[o:o + m]
                    o += m
            for ln, r in zip(self.lines, res_l):
                out[ln["blk"]].append(r)
                wts[ln["blk"]].append(np.full(len(r), ln["w"]))
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
        g0 = {"code_mm": 90.0}
        if geo0:
            g0.update(geo0)
        x += [g0.get(nm, 0.0) for nm in self.geo]
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

    def per_line_rms(self, x):
        _, _, perl = self.blocks(x, per_line=True)
        return {ln["id"]: dict(rms=float(np.sqrt(np.mean(r ** 2))), mean=float(np.mean(r)), n=len(r)) for ln, r in zip(self.lines, perl)}

    def solve(self, x0, reweight=True, iters=8, loss="linear", f_scale=3.0, fixed_sig=None, max_nfev=200):
        """Outer loop: foot points -> least squares -> variance components (if reweight)."""
        x = np.asarray(x0, float)
        if fixed_sig is not None:
            self.sig.update(fixed_sig)
        r = None
        prev = None
        for it in range(iters):
            if self.lines:
                self.update_feet(x)
            r = least_squares(self.residuals, x, method="trf", loss=loss, f_scale=f_scale, x_scale="jac",
                              max_nfev=max_nfev, xtol=1e-10, ftol=1e-10)
            x = r.x
            changed = False
            if reweight:
                b, w = self.blocks(x)
                new = {}
                if len(b["M"]) > 10:
                    new["M"] = max(float(np.sqrt(np.mean(b["M"] ** 2))), 0.05)
                for k in ("S", "E", "X"):
                    if len(b[k]) > 5:
                        ww = w[k] ** 2
                        new[k] = max(float(np.sqrt(np.sum(ww * b[k] ** 2) / np.sum(ww))), 0.05)
                changed = any(abs(new[k] / self.sig[k] - 1) > 0.02 for k in new)
                self.sig.update(new)
            moved = prev is None or np.max(np.abs(x[:self.ni] - prev[:self.ni]) / np.maximum(np.abs(prev[:self.ni]), 1e-3)) > 1e-5
            prev = x.copy()
            if not changed and not moved:
                break
        if self.lines:
            self.update_feet(x)
            r = least_squares(self.residuals, x, method="trf", loss=loss, f_scale=f_scale, x_scale="jac",
                              max_nfev=max_nfev, xtol=1e-10, ftol=1e-10)
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


def md_replace_section(path, header, lines):
    """Replace (or append) the section starting with `header` (a '## ...' line) in a markdown file."""
    import os
    txt = open(path).read() if os.path.exists(path) else ""
    parts = txt.split("\n## ")
    keep = [parts[0]] + ["## " + p for p in parts[1:] if not ("## " + p).startswith(header)]
    new = "\n".join(k.rstrip("\n") + "\n" for k in keep).rstrip("\n") + "\n\n" + header + "\n\n" + "\n".join(lines) + "\n"
    open(path, "w").write(new)
