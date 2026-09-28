"""Helpers for the scene (non-cart) edge extraction (scripts 10_edges_scene_*.py).

Everything works on the jitter-compensated mean image (work/cache/mean_aligned_gray.npy).
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates

from common import CACHE, RESULTS, project, load_json, K_from
import edgelib

REGION = "scene"
CROPDIR = f"{CACHE}/edgecrops_{REGION}"
IMG = np.load(f"{CACHE}/mean_aligned_gray.npy").astype(np.float64)
H, W = IMG.shape

DIRCOL = {"cartX": (0, 0, 255), "cartY": (0, 200, 0), "cartZ": (255, 60, 0), "world_vertical": (255, 0, 255),
          "floor_plane": (0, 255, 255), "horizontal_other": (0, 160, 255), "unknown": (200, 200, 200)}


def color_image():
    return cv2.imread(f"{CACHE}/mean_aligned_color.png")


# ------------------------------------------------------------------ initial calibration / cart model
def init_calib():
    d = load_json(f"{CACHE}/initial_calib.json")
    K = np.array(d["K"], float)
    dist = np.array(d["dist"], float)
    poses = {int(k): (np.array(v[:3]), np.array(v[3:])) for k, v in d["poses"].items()}
    return K, dist, poses


def cart_cuboid_points(step=25.0):
    """Dense samples on the surface of the cart bounding cuboid (cart frame, mm)."""
    X = np.arange(0, 1600 + 1e-9, step)
    Y = np.arange(0, 450 + 1e-9, step)
    Z = np.arange(-1800, 0 + 1e-9, step)
    pts = []
    for x in (0, 1600):
        yy, zz = np.meshgrid(Y, Z)
        pts.append(np.column_stack([np.full(yy.size, x), yy.ravel(), zz.ravel()]))
    for y in (0, 450):
        xx, zz = np.meshgrid(X, Z)
        pts.append(np.column_stack([xx.ravel(), np.full(xx.size, y), zz.ravel()]))
    for z in (0, -1800):
        xx, yy = np.meshgrid(X, Y)
        pts.append(np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, z)]))
    return np.vstack(pts)


def cart_masks(dilate=12):
    """Boolean masks (H x W) of the projected cart cuboids (convex hull, dilated)."""
    K, dist, poses = init_calib()
    P3 = cart_cuboid_points()
    out = {}
    for cart, (rv, tv) in poses.items():
        uv = project(P3, K, dist, rv, tv)
        ok = np.isfinite(uv).all(1) & (np.abs(uv[:, 0]) < 5000) & (np.abs(uv[:, 1]) < 5000)
        hull = cv2.convexHull(uv[ok].astype(np.float32))
        m = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(m, hull.astype(np.int32), 1)
        if dilate:
            m = cv2.dilate(m, np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8))
        out[cart] = m.astype(bool)
    return out


def cart_wireframe(cart, n=80):
    """List of (name, Nx2 polyline) of the cuboid edges and shelf-front/back lines of a cart."""
    K, dist, poses = init_calib()
    rv, tv = poses[cart]
    lines = []
    zs = {"top": 0, "A": -195, "B": -595, "C": -995, "D": -1300, "E": -1605, "floor": -1800}
    t = np.linspace(0, 1, n)
    for zn, z in zs.items():
        for y in (0, 450):
            P = np.column_stack([1600 * t, np.full(n, y), np.full(n, z)])
            lines.append((f"X_{zn}_Y{y}", project(P, K, dist, rv, tv)))
        for x in (0, 1600):
            P = np.column_stack([np.full(n, x), 450 * t, np.full(n, z)])
            lines.append((f"Y_{zn}_X{x}", project(P, K, dist, rv, tv)))
    for x in (0, 1600):
        for y in (0, 450):
            P = np.column_stack([np.full(n, x), np.full(n, y), -1800 * t])
            lines.append((f"Z_X{x}_Y{y}", project(P, K, dist, rv, tv)))
    return lines


def floor_plane_world(cart=310):
    """Return R, t of cart frame and the floor plane Z=-1800 of the given cart (camera frame)."""
    K, dist, poses = init_calib()
    rv, tv = poses[cart]
    R = cv2.Rodrigues(rv.reshape(3, 1))[0]
    return K, dist, R, tv


# ------------------------------------------------------------------ tracing
def smooth_curve(P, deg=3, n=None, ext=(0.0, 0.0), step=3.0):
    """Polynomial (in chord coordinate) fit through a polyline, optionally extended."""
    P = np.asarray(P, float)
    c, d, _, _ = edgelib.line_fit(P)
    if np.dot(P[-1] - P[0], d) < 0:
        d = -d
    nrm = np.array([-d[1], d[0]])
    t = (P - c) @ d
    r = (P - c) @ nrm
    dg = min(deg, len(P) - 1)
    q = np.polyfit(t, r, dg)
    q2 = np.polyfit(t, r, min(2, len(P) - 1))
    tt = np.arange(t.min() - ext[0], t.max() + ext[1] + 1e-9, step)
    inside = (tt >= t.min()) & (tt <= t.max())
    rr = np.where(inside, np.polyval(q, tt), np.polyval(q2, tt))
    return c[None] + tt[:, None] * d[None] + rr[:, None] * nrm[None]


def profile_contrast(P, N, off=3.0):
    """Intensity (left-normal side minus right side) at +-off px along the normal."""
    a = map_coordinates(IMG, [P[:, 1] + off * N[:, 1], P[:, 0] + off * N[:, 0]], order=1)
    b = map_coordinates(IMG, [P[:, 1] - off * N[:, 1], P[:, 0] - off * N[:, 0]], order=1)
    return a - b


def trace(poly, polarity=0, halfwidth=5.0, fit_deg=3, sigma=1.0, step=2.0, iters=4, min_strength=3.0,
          min_rel=0.4, max_res=None, cut_boxes=(), min_frag=12, trim=4.0, max_gap=10.0):
    """edgelib.trace_edge + cleaning.

    cleaning: strength < min_rel * median dropped; points off a smooth (deg<=3 in chord coordinate,
    fitted per 150 px window) curve by more than max(0.3, 4 robust sigma) dropped; points inside
    cut_boxes dropped; after that, fragments separated by gaps > max_gap are trimmed by `trim` px at
    interior ends and fragments < min_frag px are dropped. Returns dict points, strength, normal, s."""
    tr = edgelib.trace_edge(poly, halfwidth=halfwidth, step=step, polarity=polarity, sigma=sigma, iters=iters,
                            min_strength=min_strength, fit_deg=fit_deg)
    P, S, N, s = tr["points"], tr["strength"], tr["normal"], tr["s"]
    if len(P) < 5:
        return dict(points=P, strength=S, normal=N, s=s)
    keep = S >= min_rel * np.median(S)
    for (x0, y0, x1, y1) in cut_boxes:
        keep &= ~((P[:, 0] >= x0) & (P[:, 0] <= x1) & (P[:, 1] >= y0) & (P[:, 1] <= y1))
    # residual to a smooth local curve
    for _ in range(3):
        if keep.sum() < 8:
            break
        res = local_smooth_residual(P, keep)
        mad = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep])))
        thr = max(0.3, 4 * mad) if max_res is None else max_res
        newkeep = keep & (np.abs(res) <= thr)
        if newkeep.sum() == keep.sum():
            break
        keep = newkeep
    P, S, N, s = P[keep], S[keep], N[keep], s[keep]
    # fragments
    if len(P) < 3:
        return dict(points=P, strength=S, normal=N, s=s)
    gaps = np.where(np.diff(s) > max_gap)[0]
    starts = np.r_[0, gaps + 1]
    ends = np.r_[gaps + 1, len(P)]
    sel = np.zeros(len(P), bool)
    for a, b in zip(starts, ends):
        lo, hi = s[a], s[b - 1]
        if a > 0:
            lo += trim
        if b < len(P):
            hi -= trim
        if hi - lo >= min_frag:
            sel[a:b] |= (s[a:b] >= lo) & (s[a:b] <= hi)
    return dict(points=P[sel], strength=S[sel], normal=N[sel], s=s[sel])


def local_smooth_residual(P, keep, win=150.0, deg=2):
    """Residual of every point from a local polynomial fit (in the chord frame) of its window."""
    c, d, _, _ = edgelib.line_fit(P[keep])
    nrm = np.array([-d[1], d[0]])
    t = (P - c) @ d
    r = (P - c) @ nrm
    res = np.zeros(len(P))
    L = t[keep].max() - t[keep].min()
    if L <= win * 1.5:
        q = np.polyfit(t[keep], r[keep], min(3, keep.sum() - 1))
        return r - np.polyval(q, t)
    centres = np.arange(t.min(), t.max() + win / 2, win / 2)
    best = np.full(len(P), np.inf)
    for c0 in centres:
        m = keep & (np.abs(t - c0) <= win)
        if m.sum() < 8:
            continue
        q = np.polyfit(t[m], r[m], deg)
        inwin = np.abs(t - c0) <= win / 2
        rr = r - np.polyval(q, t)
        upd = inwin & (np.abs(t - c0) < best)
        res[upd] = rr[upd]
        best[upd] = np.abs(t - c0)[upd]
    return res


def chord_stats(P):
    c, d, rms, mx = edgelib.line_fit(P)
    bow, rms2 = edgelib.sagitta(P)
    L = float(np.ptp((P - c) @ d))
    return dict(rms=rms, max=mx, bow=bow, length=L)


def resid_noise(P, deg=3):
    """rms residual to a cubic in the chord coordinate (local noise estimate)."""
    c, d, _, _ = edgelib.line_fit(P)
    nrm = np.array([-d[1], d[0]])
    t = (P - c) @ d
    r = (P - c) @ nrm
    q = np.polyfit(t, r, min(deg, len(P) - 1))
    return float(np.sqrt(np.mean((r - np.polyval(q, t)) ** 2)))


# ------------------------------------------------------------------ crops
def render_crop(P, path, label="", scale=3, pad=24, maxside=1500, extra=(), guide=None, col=None):
    """Zoomed colour crop around points P with the points drawn (small dots)."""
    img = color_image()
    P = np.asarray(P, float)
    x0 = int(max(0, np.floor(P[:, 0].min()) - pad))
    x1 = int(min(W, np.ceil(P[:, 0].max()) + pad))
    y0 = int(max(0, np.floor(P[:, 1].min()) - pad))
    y1 = int(min(H, np.ceil(P[:, 1].max()) + pad))
    sc = scale
    while max(x1 - x0, y1 - y0) * sc > maxside and sc > 1:
        sc -= 1
    if max(x1 - x0, y1 - y0) * sc > maxside:
        sc = maxside / max(x1 - x0, y1 - y0)
    crop = cv2.resize(img[y0:y1, x0:x1], None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
    if guide is not None:
        g = (np.asarray(guide) - [x0, y0]) * sc
        cv2.polylines(crop, [np.round(g * 4).astype(np.int32)], False, (255, 200, 0), 1, cv2.LINE_AA, shift=2)
    for Q, c in extra:
        for p in (np.asarray(Q) - [x0, y0]) * sc:
            cv2.circle(crop, (int(round(p[0])), int(round(p[1]))), 1, c, -1)
    colr = (0, 0, 255) if col is None else col
    for p in (P - [x0, y0]) * sc:
        cv2.circle(crop, (int(round(p[0] * 4)), int(round(p[1] * 4))), 4, colr, -1, cv2.LINE_AA, shift=2)
    cv2.putText(crop, f"{label}  x{sc:.2g} @({x0},{y0})", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
    cv2.putText(crop, f"{label}  x{sc:.2g} @({x0},{y0})", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    cv2.imwrite(path, crop)
    return crop


def local_windows(P, path, label="", n=4, half=20, scale=6, col=(0, 0, 255)):
    """n high-zoom windows spread along the edge, side by side (to check the sub-pixel position)."""
    img = color_image()
    P = np.asarray(P, float)
    idx = np.linspace(0, len(P) - 1, n + 2)[1:-1].astype(int) if len(P) > n + 2 else np.arange(len(P))
    tiles = []
    for i in idx:
        cx, cy = P[i]
        x0 = int(np.clip(round(cx) - half, 0, W - 2 * half))
        y0 = int(np.clip(round(cy) - half, 0, H - 2 * half))
        tile = cv2.resize(img[y0:y0 + 2 * half, x0:x0 + 2 * half], None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_NEAREST)
        m = (P[:, 0] >= x0) & (P[:, 0] < x0 + 2 * half) & (P[:, 1] >= y0) & (P[:, 1] < y0 + 2 * half)
        for p in (P[m] - [x0, y0]) * scale + scale / 2 - 0.5:
            cv2.circle(tile, (int(round(p[0] * 4)), int(round(p[1] * 4))), 6, col, -1, cv2.LINE_AA, shift=2)
        cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1), (255, 255, 255), 1)
        tiles.append(tile)
    strip = np.hstack(tiles)
    cv2.putText(strip, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
    cv2.putText(strip, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    if path:
        cv2.imwrite(path, strip)
    return strip


def trace_line_centre(poly, dark=True, halfwidth=4.0, step=2.0, sigma=1.0, iters=4, fit_deg=3, min_strength=1.0):
    """Centre of a thin dark (or bright) line: extremum of the Gaussian-smoothed profile along the normal,
    parabolic sub-pixel refinement, iterated with a polynomial path. strength = depth of the line
    (mean of the profile ends minus the extremum value, sign-corrected)."""
    from edgelib import _resample_polyline
    path = np.asarray(poly, float)
    hw = halfwidth
    out = None
    for it in range(iters):
        Q, nrm, s = _resample_polyline(path, step)
        offs = np.arange(-hw - 3, hw + 3 + 1e-9, 0.25)
        X = Q[:, None, 0] + nrm[:, None, 0] * offs[None]
        Y = Q[:, None, 1] + nrm[:, None, 1] * offs[None]
        prof = map_coordinates(IMG, [Y.ravel(), X.ravel()], order=3, mode="nearest").reshape(X.shape)
        sm = gaussian_filter1d(prof, sigma / 0.25, axis=1)
        a = -sm if dark else sm
        inner = np.abs(offs) <= hw
        ai = np.where(inner[None], a, -np.inf)
        j = np.argmax(ai, axis=1)
        rows = np.arange(len(j))
        jj = np.clip(j, 1, len(offs) - 2)
        y0, y1, y2 = a[rows, jj - 1], a[rows, jj], a[rows, jj + 1]
        den = y0 - 2 * y1 + y2
        delta = np.where(np.abs(den) > 1e-12, 0.5 * (y0 - y2) / den, 0.0)
        delta = np.clip(delta, -0.5, 0.5)
        o = offs[jj] + delta * 0.25
        pts = Q + nrm * o[:, None]
        ends = 0.5 * (a[:, :8].mean(1) + a[:, -8:].mean(1))
        strength = y1 - ends
        ok = (strength >= min_strength) & (np.abs(o) < hw - 0.3)
        out = dict(points=pts[ok], strength=strength[ok], normal=nrm[ok], s=s[ok])
        if ok.sum() < 5:
            return out
        ss = out["s"]
        px = np.polyfit(ss, out["points"][:, 0], fit_deg)
        py = np.polyfit(ss, out["points"][:, 1], fit_deg)
        sfine = np.linspace(ss.min(), ss.max(), 50)
        path = np.column_stack([np.polyval(px, sfine), np.polyval(py, sfine)])
        hw = max(2.5, hw * 0.7)
    return out


def polarity_for(guide, bright):
    """edgelib polarity (+1: intensity increases along the left-hand normal) from the bright side."""
    if bright is None:
        return 0
    G = np.asarray(guide, float)
    d = G[-1] - G[0]
    d = d / np.linalg.norm(d)
    nrm = np.array([-d[1], d[0]])
    v = {"+x": (1, 0), "-x": (-1, 0), "+y": (0, 1), "-y": (0, -1)}[bright]
    return 1 if nrm @ np.array(v, float) > 0 else -1


def clean(tr, min_rel=0.4, cut_boxes=(), max_res=None, min_frag=12, trim=4.0, max_gap=10.0):
    """cleaning of a raw trace (see trace())."""
    P, S, N, s = tr["points"], tr["strength"], tr["normal"], tr["s"]
    if len(P) < 5:
        return dict(points=P, strength=S, normal=N, s=s)
    keep = S >= min_rel * np.median(S)
    for (x0, y0, x1, y1) in cut_boxes:
        keep &= ~((P[:, 0] >= x0) & (P[:, 0] <= x1) & (P[:, 1] >= y0) & (P[:, 1] <= y1))
    for _ in range(4):
        if keep.sum() < 8:
            break
        res = local_smooth_residual(P, keep)
        mad = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep])))
        thr = max(0.3, 4 * mad) if max_res is None else max_res
        newkeep = keep & (np.abs(res) <= thr)
        if newkeep.sum() == keep.sum():
            break
        keep = newkeep
    P, S, N, s = P[keep], S[keep], N[keep], s[keep]
    if len(P) < 3:
        return dict(points=P, strength=S, normal=N, s=s)
    gaps = np.where(np.diff(s) > max_gap)[0]
    starts = np.r_[0, gaps + 1]
    ends = np.r_[gaps + 1, len(P)]
    sel = np.zeros(len(P), bool)
    for a, b in zip(starts, ends):
        lo, hi = s[a], s[b - 1]
        if a > 0:
            lo += trim
        if b < len(P):
            hi -= trim
        if hi - lo >= min_frag:
            sel[a:b] |= (s[a:b] >= lo) & (s[a:b] <= hi)
    return dict(points=P[sel], strength=S[sel], normal=N[sel], s=s[sel])


_CC = None


def class_calib():
    """Lens used only to CLASSIFY edge directions (vanishing-direction test): the current plumb-line
    distortion estimate (work/cache/method_plumbline.json; monotonic up to the corners, unlike the
    initial joint fit) with cart poses re-fitted by solvePnP to all verified marker corners
    (work/cache/markers_final.json). Falls back to the initial calibration."""
    global _CC
    if _CC is not None:
        return _CC
    try:
        m = load_json(f"{CACHE}/method_plumbline.json")
        K = np.array(m["camera_matrix"], float)
        dist = np.array(m["dist_coeffs"], float)
        mk = load_json(f"{CACHE}/markers_final.json")
        poses = {}
        for cart in (80, 310):
            obj, img = [], []
            for r in mk:
                if r["cart"] != cart:
                    continue
                for c3, c2, ok in zip(r["corners_3d_mm"], r["corners_px"], r["corner_valid"]):
                    if ok:
                        obj.append(c3)
                        img.append(c2)
            obj = np.array(obj, float)
            img = np.array(img, float)
            K0, d0, p0 = init_calib()
            ok, rv, tv = cv2.solvePnP(obj, img, K, dist, p0[cart][0].reshape(3, 1).copy(),
                                      p0[cart][1].reshape(3, 1).copy(), useExtrinsicGuess=True,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
            res = project(obj, K, dist, rv.ravel(), tv.ravel()) - img
            poses[cart] = (rv.ravel(), tv.ravel(), float(np.sqrt(np.mean(np.sum(res ** 2, 1)))))
        _CC = (K, dist, poses, "method_plumbline.json + solvePnP")
    except Exception as ex:  # pragma: no cover
        K, dist, p0 = init_calib()
        _CC = (K, dist, {c: (p[0], p[1], np.nan) for c, p in p0.items()}, f"initial ({ex})")
    return _CC


def strip_view(P, hw=10, sperp=6, salong=1.5, rowlen=900, label="", others=()):
    """Straightened view of an edge: the colour image resampled in the frame of a smooth (quadratic)
    curve through the points: perpendicular magnified sperp x, along-edge salong x; long edges are
    split into several rows. Traced points = red dots; `others` = list of (points, colour)."""
    img = color_image().astype(np.float64)
    P = np.asarray(P, float)
    c, d, _, _ = edgelib.line_fit(P)
    if np.dot(P[-1] - P[0], d) < 0:
        d = -d
    nrm = np.array([-d[1], d[0]])
    t = (P - c) @ d
    r = (P - c) @ nrm
    q = np.polyfit(t, r, 2 if len(P) > 10 else 1)
    t0, t1 = t.min() - 8, t.max() + 8
    tt = np.arange(t0, t1, 1.0 / salong)
    rr = np.arange(-hw, hw + 1e-9, 1.0 / sperp)
    base = c[None] + tt[:, None] * d[None] + np.polyval(q, tt)[:, None] * nrm[None]
    X = base[None, :, 0] + rr[:, None] * nrm[0]
    Y = base[None, :, 1] + rr[:, None] * nrm[1]
    ch = [map_coordinates(img[:, :, k], [Y.ravel(), X.ravel()], order=1, mode="nearest").reshape(X.shape)
          for k in range(3)]
    S = np.clip(np.dstack(ch), 0, 255).astype(np.uint8)
    S = np.ascontiguousarray(S)

    def draw(Q, col, rad):
        Q = np.asarray(Q, float)
        tq = (Q - c) @ d
        rq = (Q - c) @ nrm - np.polyval(q, tq)
        for a, b in zip(tq, rq):
            u = (a - t0) * salong
            v = (b + hw) * sperp
            cv2.circle(S, (int(round(u * 4)), int(round(v * 4))), rad, col, -1, cv2.LINE_AA, shift=2)

    for Q, col in others:
        draw(Q, col, 5)
    draw(P, (0, 0, 255), 5)
    rows = []
    for a in range(0, S.shape[1], rowlen):
        piece = S[:, a:a + rowlen]
        if piece.shape[1] < rowlen:
            piece = np.hstack([piece, np.zeros((piece.shape[0], rowlen - piece.shape[1], 3), np.uint8)])
        rows.append(piece)
        rows.append(np.full((3, rowlen, 3), 255, np.uint8))
    out = np.vstack(rows)
    if label:
        cv2.putText(out, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(out, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    return out


def context_view(P, maxside=520, pad=40, guide=None):
    """Colour crop around the edge (with context), points drawn, scaled so that the longer side is maxside."""
    img = color_image()
    P = np.asarray(P, float)
    x0 = int(max(0, np.floor(P[:, 0].min()) - pad))
    x1 = int(min(W, np.ceil(P[:, 0].max()) + pad))
    y0 = int(max(0, np.floor(P[:, 1].min()) - pad))
    y1 = int(min(H, np.ceil(P[:, 1].max()) + pad))
    sc = maxside / max(x1 - x0, y1 - y0)
    crop = cv2.resize(img[y0:y1, x0:x1], None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC if sc > 1 else cv2.INTER_AREA)
    if guide is not None:
        g = (np.asarray(guide) - [x0, y0]) * sc
        cv2.polylines(crop, [np.round(g * 4).astype(np.int32)], False, (255, 200, 0), 1, cv2.LINE_AA, shift=2)
    for p in (P - [x0, y0]) * sc:
        cv2.circle(crop, (int(round(p[0] * 4)), int(round(p[1] * 4))), 5, (0, 0, 255), -1, cv2.LINE_AA, shift=2)
    cv2.putText(crop, f"({x0},{y0}) x{sc:.2f}", (4, crop.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3)
    cv2.putText(crop, f"({x0},{y0}) x{sc:.2f}", (4, crop.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    return crop


def compose(parts, bg=0):
    w = max(p.shape[1] for p in parts)
    out = []
    for p in parts:
        if p.shape[1] < w:
            p = np.hstack([p, np.full((p.shape[0], w - p.shape[1], 3), bg, np.uint8)])
        out.append(p)
    return np.vstack(out)
