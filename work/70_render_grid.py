"""Visual check of the final lens: render a grid of floor locations, the cart models and straight-line extensions
into the photo with the main lens (results/lens_result.json) and the cart poses of the main combined fit.

  results/grid_overview.png    : mean image + floor grid (500 mm, cart-310 frame, floor = Z -1800 mm) + wireframes of
                                 both carts from the drawing (top plate, shelves A-E front/back, posts, footprint) +
                                 predicted sticker squares; everything drawn with the full distortion model
  results/grid_lines.png       : every verified structural edge: traced points + the straight line fitted to them in the
                                 undistorted frame, mapped back through the distortion and EXTENDED across the image
                                 (if the lens is right, the curve follows the edge along its whole length)
  results/grid_undistorted.png : undistorted image (same K) with the same floor grid and wireframes as straight lines
  results/grid_crops.png       : zoomed crops (overview | line extensions) at the places where the check is most sensitive
Usage: python3 70_render_grid.py
"""
import sys

import cv2
import numpy as np

from common import CACHE, H, RESULTS, W, load_json, marker_corners_3d, project, undistort_points
from markerdata import load_markers

L = load_json(f"{RESULTS}/lens_result.json")
K = np.array(L["camera_matrix"], float)
D = np.array(L["dist_coeffs"], float)
FIT = load_json(f"{CACHE}/combined_fit.json")
X = np.array(FIT["x_main"])
NI = len(FIT["inames"])
POSES = {80: X[NI:NI + 6], 310: X[NI + 6:NI + 12]}  # Combined carts order (80, 310); rvec, tvec: cart -> camera
IMG = cv2.imread(f"{CACHE}/mean_aligned_color.png")
M = load_markers()
SHELF_Z = {"top": 0.0, "A": -195.0, "B": -595.0, "C": -995.0, "D": -1300.0, "E": -1605.0}
FLOOR_Z = -1800.0
CART = dict(w=1600.0, d=450.0)
# undistorted radius of the farthest image corner (+10 %): beyond it the polynomial model is not valid (fold)
_cor = np.array([[0, 0], [W - 1, 0], [0, H - 1], [W - 1, H - 1]], float)
_n = undistort_points(_cor, K, D)  # normalised undistorted coordinates
R_LIM = 1.10 * float(np.max(np.hypot(_n[:, 0], _n[:, 1])))
COL = {80: (255, 0, 255), 310: (0, 165, 255), "grid": (0, 255, 255), "grid_major": (0, 200, 255), "stk": (0, 255, 0),
       "line": (0, 0, 255), "pts": (255, 255, 0)}


def cam_of(Xc, cart):
    rv, tv = POSES[cart][:3], POSES[cart][3:]
    R = cv2.Rodrigues(rv.reshape(3, 1))[0]
    return Xc @ R.T + tv


def proj(Xc, cart, distort=True):
    """cart-frame points -> pixels; NaN where behind the camera or beyond the valid radius of the distortion model."""
    C = cam_of(np.asarray(Xc, float).reshape(-1, 3), cart)
    z = C[:, 2]
    xn = C[:, 0] / np.where(z > 1e-6, z, np.nan)
    yn = C[:, 1] / np.where(z > 1e-6, z, np.nan)
    ok = (z > 1e-6) & (np.hypot(xn, yn) < R_LIM)
    if distort:
        uv = project(np.asarray(Xc, float).reshape(-1, 3), K, D, rvec=POSES[cart][:3], tvec=POSES[cart][3:])
    else:
        uv = np.column_stack([K[0, 0] * xn + K[0, 2], K[1, 1] * yn + K[1, 2]])
    uv[~ok] = np.nan
    return uv


def polyline(img, uv, col, th=1, scale=1.0, off=(0, 0)):
    uv = (uv - np.asarray(off, float)) * scale
    good = np.isfinite(uv).all(1)
    seg = []
    for p, g in zip(uv, good):
        if g and abs(p[0]) < 1e5 and abs(p[1]) < 1e5:
            seg.append(p)
        elif len(seg) > 1:
            cv2.polylines(img, [np.round(np.array(seg) * 8).astype(np.int32)], False, col, th, cv2.LINE_AA, shift=3)
            seg = []
        else:
            seg = []
    if len(seg) > 1:
        cv2.polylines(img, [np.round(np.array(seg) * 8).astype(np.int32)], False, col, th, cv2.LINE_AA, shift=3)


def seg3d(A, B, n=120):
    t = np.linspace(0, 1, n)[:, None]
    return np.asarray(A, float)[None] * (1 - t) + np.asarray(B, float)[None] * t


def _to_floor(uv, cart=310, z=FLOOR_Z):
    rv, tv = POSES[cart][:3], POSES[cart][3:]
    R = cv2.Rodrigues(rv.reshape(3, 1))[0]
    C = -R.T @ tv
    n = undistort_points(np.asarray(uv, float), K, D)
    r = np.column_stack([n, np.ones(len(n))]) @ R
    return C[None] + ((z - C[2]) / r[:, 2])[:, None] * r


def _centre_line(a, b):
    """centre line of a tape from its two traced edges, on the floor of cart 310: point, unit direction (2D)."""
    E = {e["id"]: e for bl in load_json(f"{RESULTS}/edges.json")["regions"] for e in bl["edges"]}
    fits = []
    for k in (a, b):  # each edge separately (the two edges have different traced extents)
        Q = _to_floor(E[k]["points"])[:, :2]
        c = Q.mean(0)
        _, _, vt = np.linalg.svd(Q - c)
        fits.append((c, vt[0]))
    d = fits[0][1] + np.sign(fits[0][1] @ fits[1][1]) * fits[1][1]
    d /= np.linalg.norm(d)
    nrm = np.array([-d[1], d[0]])
    off = (fits[1][0] - fits[0][0]) @ nrm
    return fits[0][0] + 0.5 * off * nrm, d


# floor frame aligned to the white floor tape: origin at the crossing of the two tape centre lines, u along the long
# (image-vertical) tape, v along the transverse tape; lies in the floor plane of cart 310 (Z = -1800 mm)
_cv, _dv = _centre_line("scene_tapeV_left", "scene_tapeV_right")
_ch, _dh = _centre_line("scene_tapeH_top", "scene_tapeH_bottom")
_A = np.column_stack([_dv, -_dh])
_st = np.linalg.solve(_A, _ch - _cv)
TAPE_O = _cv + _st[0] * _dv
TAPE_U = _dv if _dv[1] > 0 else -_dv          # towards the camera side of the junction (image down)
TAPE_V = np.array([-TAPE_U[1], TAPE_U[0]])
TAPE_ANGLE = float(np.degrees(np.arccos(abs(_dv @ _dh))))


def tape_to_cart(u, v):
    P = TAPE_O[None] + np.outer(u, TAPE_U) + np.outer(v, TAPE_V)
    return np.column_stack([P, np.full(len(P), FLOOR_Z)])


def tape_grid(step=500.0, ur=(-5000, 5000), vr=(-6000, 6000)):
    out = []
    for u in np.arange(ur[0], ur[1] + 1, step):
        out.append((tape_to_cart(np.full(400, u), np.linspace(vr[0], vr[1], 400)), abs(u) % 1000 < 1, u == 0))
    for v in np.arange(vr[0], vr[1] + 1, step):
        out.append((tape_to_cart(np.linspace(ur[0], ur[1], 400), np.full(400, v)), abs(v) % 1000 < 1, v == 0))
    return out


def floor_grid(step=500.0, xr=(-4000, 6000), yr=(-5000, 5000)):
    """Lines of a floor grid in the cart-310 frame at Z = FLOOR_Z; returns [(pts3d, major)]."""
    out = []
    for x in np.arange(xr[0], xr[1] + 1, step):
        out.append((seg3d([x, yr[0], FLOOR_Z], [x, yr[1], FLOOR_Z], 400), abs(x) % 1000 < 1))
    for y in np.arange(yr[0], yr[1] + 1, step):
        out.append((seg3d([xr[0], y, FLOOR_Z], [xr[1], y, FLOOR_Z], 400), abs(y) % 1000 < 1))
    return out


def wireframe():
    w, dp = CART["w"], CART["d"]
    L_ = []
    for z in list(SHELF_Z.values()) + [FLOOR_Z]:
        L_ += [seg3d([0, 0, z], [w, 0, z]), seg3d([0, dp, z], [w, dp, z])]
        if z in (0.0, FLOOR_Z):
            L_ += [seg3d([0, 0, z], [0, dp, z]), seg3d([w, 0, z], [w, dp, z])]
    for x in (0.0, w):
        for y in (0.0, dp):
            L_.append(seg3d([x, y, 0], [x, y, FLOOR_Z]))
    return L_


def draw_scene(img, distort=True, scale=1.0, off=(0, 0), th=1):
    for P, major, axis in tape_grid():
        polyline(img, proj(P, 310, distort), (0, 0, 255) if axis else (COL["grid_major"] if major else COL["grid"]),
                 th + (1 if (major or axis) else 0), scale, off)
    for c in (80, 310):
        for P in wireframe():
            polyline(img, proj(P, c, distort), COL[c], th + 1, scale, off)
    for m in M:
        C = marker_corners_3d(m["id"], m["cart"], m["rot_k"])
        P = np.vstack([seg3d(C[i], C[(i + 1) % 4], 20) for i in range(4)])
        polyline(img, proj(P, m["cart"], distort), COL["stk"], th, scale, off)


def label_grid(img, distort=True):
    for x in np.arange(-5000, 5001, 1000):
        for y in np.arange(-6000, 6001, 1000):
            uv = proj(tape_to_cart(np.array([x]), np.array([y])), 310, distort)[0]
            if np.isfinite(uv).all() and 0 <= uv[0] < W and 0 <= uv[1] < H:
                cv2.circle(img, (int(uv[0]), int(uv[1])), 4, COL["grid_major"], -1, cv2.LINE_AA)
                cv2.putText(img, f"{x / 1000:+.0f},{y / 1000:+.0f}", (int(uv[0]) + 5, int(uv[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(img, f"{x / 1000:+.0f},{y / 1000:+.0f}", (int(uv[0]) + 5, int(uv[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL["grid_major"], 1, cv2.LINE_AA)


def with_header(img, lines):
    """image with a dark header bar holding the legend (the image content is not covered)."""
    hb = 14 + 26 * len(lines)
    bar = np.full((hb, img.shape[1], 3), 30, np.uint8)
    y = 28
    for txt, col in lines:
        cv2.putText(bar, txt, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.66, col, 1, cv2.LINE_AA)
        y += 26
    return np.vstack([bar, img])


# ---------------------------------------------------------------- straight-line extensions of the traced edges
EJ = load_json(f"{RESULTS}/edges.json")
EDGES = [e for b in EJ["regions"] for e in b["edges"] if e.get("straight_3d", False) and e.get("verified", "yes") != "no" and len(e["points"]) > 5]


def line_extension(P, ext=1.2, n=600):
    """TLS line through the undistorted points, extended by ext x chord on both sides, mapped back to pixels."""
    U = undistort_points(np.asarray(P, float), K, D)  # normalised
    c = U.mean(0)
    _, _, vt = np.linalg.svd(U - c)
    dvec = vt[0]
    s = (U - c) @ dvec
    lo, hi = s.min(), s.max()
    span = hi - lo
    t = np.linspace(lo - ext * span, hi + ext * span, n)
    xn = c[None] + t[:, None] * dvec[None]
    r2 = np.sum(xn ** 2, 1)
    rad = 1 + D[0] * r2 + D[1] * r2 ** 2 + D[4] * r2 ** 3
    uv = K[:2, 2] + K[0, 0] * xn * rad[:, None]
    uv[np.sqrt(r2) > R_LIM] = np.nan
    # residual of the traced points to the distorted line (px): distance to the nearest curve sample
    dd = np.min(np.linalg.norm(np.asarray(P, float)[:, None, :] - uv[None, np.isfinite(uv).all(1), :], axis=2), axis=1)
    return uv, float(np.sqrt(np.mean(dd ** 2))), float(np.max(dd))


def main():
    over = IMG.copy()
    draw_scene(over)
    label_grid(over)
    LEG_O = [("floor location grid 0.5 m aligned to the white floor tape (red = tape centre lines, origin at their crossing), "
              "floor plane of the carts; labels = u,v [m]", COL["grid_major"]),
                  ("cart 80 model from the drawing (top plate, shelves A-E, posts, footprint)", COL[80]),
                  ("cart 310 model from the drawing", COL[310]), ("predicted sticker code squares (drawing positions)", COL["stk"]),
                  (f"main lens f {K[0, 0]:.1f}, pp ({K[0, 2]:.1f}, {K[1, 2]:.1f}), k1 {D[0]:+.3f}, k2 {D[1]:+.3f}; cart poses of the main fit", (255, 255, 255))]
    cv2.imwrite(f"{RESULTS}/grid_overview.png", with_header(over, LEG_O))

    lines = IMG.copy()
    stats = []
    for e in EDGES:
        P = np.asarray(e["points"], float)
        uv, rms, mx = line_extension(P)
        polyline(lines, uv, COL["line"], 1)
        for p in P[::2]:
            cv2.circle(lines, (int(round(p[0])), int(round(p[1]))), 1, COL["pts"], -1)
        stats.append(dict(id=e["id"], rms=rms, max=mx, n=len(P)))
    LEG_L = [("traced edge points (verified straight structural edges)", COL["pts"]),
             ("straight 3D line through them, drawn with the main lens and extended 1.2x its length on both sides", COL["line"])]
    cv2.imwrite(f"{RESULTS}/grid_lines.png", with_header(lines, LEG_L))

    und = cv2.undistort(IMG, K, D, None, K)
    draw_scene(und, distort=False)
    for e in EDGES:
        U = K[:2, 2] + K[0, 0] * undistort_points(np.asarray(e["points"], float), K, D)
        for p in U[::2]:
            if 0 <= p[0] < W and 0 <= p[1] < H:
                cv2.circle(und, (int(round(p[0])), int(round(p[1]))), 1, COL["pts"], -1)
    cv2.imwrite(f"{RESULTS}/grid_undistorted.png", with_header(und, [("UNDISTORTED image (main lens, same K): floor grid, cart models and all "
                                                                   "straight edges (cyan) must be straight lines here", (255, 255, 255))] + LEG_O[:4]))
    crops(over, lines, CROPS)

    st = np.array([s["rms"] for s in stats])
    print(f"edges {len(stats)}: straight-line residual RMS median {np.median(st):.3f} px, max {max(s['max'] for s in stats):.2f} px")
    worst = sorted(stats, key=lambda s: -s["max"])[:6]
    print("largest deviations:", [(s["id"], round(s["max"], 2)) for s in worst])
    from common import save_json
    save_json(dict(lens=dict(K=K.tolist(), dist=D.tolist()), poses={str(k): v.tolist() for k, v in POSES.items()}, r_lim=R_LIM,
                   edge_line_residuals=stats), f"{CACHE}/grid_render.json")


CROPS = [(200, 30, 520, 260, "cart 310, X0 end board (top plate)"), (470, 820, 440, 260, "cart 310, X1600 end board"),
         (1220, 0, 540, 180, "cart 80, X0 end board"), (1250, 840, 420, 240, "cart 80, X1600 end board"),
         (300, 170, 520, 400, "cart 310, shelf lips A-E"), (1150, 170, 520, 420, "cart 80, shelf lips A-E"),
         (840, 600, 380, 480, "floor: white tape T-junction"), (0, 600, 560, 440, "floor: blue line, left"),
         (1600, 80, 320, 720, "scene verticals / rails, right edge"), (0, 0, 300, 640, "left edge: rack")]


def crops(over, lin, spec, out=f"{RESULTS}/grid_crops.png", z=1.5):
    """zoomed crops: overview (grid + drawing models) | straight-line extensions, side by side, one row per region."""
    tiles = []
    for x0, y0, w, h, lab in spec:
        a = cv2.resize(over[y0:y0 + h, x0:x0 + w], None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC)
        b = cv2.resize(lin[y0:y0 + h, x0:x0 + w], None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC)
        t = np.hstack([a, np.full((a.shape[0], 8, 3), 255, np.uint8), b])
        bar = np.full((34, t.shape[1], 3), 30, np.uint8)
        cv2.putText(bar, f"{lab}  (x {x0}-{x0 + w}, y {y0}-{y0 + h}; zoom {z:g}x)  left: grid + drawing models, right: straight-line extensions",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(np.vstack([bar, t]))
    wmax = max(t.shape[1] for t in tiles)
    tiles = [np.hstack([t, np.zeros((t.shape[0], wmax - t.shape[1], 3), np.uint8)]) for t in tiles]
    cv2.imwrite(out, np.vstack([np.vstack([t, np.full((8, wmax, 3), 255, np.uint8)]) for t in tiles]))
    return tiles


if __name__ == "__main__":
    main()
