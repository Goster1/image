"""Shared helpers for the 11_markers_* scripts (complete ArUco sticker detection).

Import with   M = importlib.import_module("11_markers_lib")

Contents
--------
* data loading: the 7 stills (float gray), the jitter-compensated mean image, per-frame jitter
  parameters, the seed calibration (K, dist, one pose per cart) from 02_initial_calib.py
* jitter mapping between a still and the common mean frame
* dictionary codes of DICT_4X4_1000 as 6x6 cell grids (1-cell black border, 1 = white)
* rectification of a marker neighbourhood with the homography of a quad
* bit-grid sampling / decoding and Hamming distances against the whole dictionary
* ArUco-rule sub-pixel corner refinement (cv2.cornerSubPix with the window used by the ArUco
  detector for CORNER_REFINE_SUBPIX)
* edge-based corners: sub-pixel tracing of the four sides of the black square, per-side validity
  checks (full / partial / invalid), total-least-squares line fit (straight in the image, or
  optionally straight in undistorted coordinates = slightly curved in the image) and intersection
  of adjacent sides
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates

from common import CACHE, H, W, load_gray, load_json, still_paths, undistort_points, distort_normalized

DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
NBITS = 4
NCELL = NBITS + 2

# ------------------------------------------------------------------------------------------
# data
# ------------------------------------------------------------------------------------------
_CACHE = {}


def seed():
    if "seed" not in _CACHE:
        d = load_json(f"{CACHE}/initial_calib.json")
        _CACHE["seed"] = dict(K=np.array(d["K"]), dist=np.array(d["dist"]),
                              poses={int(k): np.array(v) for k, v in d["poses"].items()},
                              jitter=np.array(d["frame_jitter"]), rot=d["rot"])
    return _CACHE["seed"]


def frames():
    """List of (file name, float64 gray image) of the 7 stills (same gray formula as the mean)."""
    if "frames" not in _CACHE:
        out = []
        for p in still_paths():
            im = cv2.imread(p).astype(np.float64)
            g = 0.114 * im[..., 0] + 0.587 * im[..., 1] + 0.299 * im[..., 2]
            out.append((p.split("/")[-1], g))
        _CACHE["frames"] = out
    return _CACHE["frames"]


def mean_gray():
    if "mean" not in _CACHE:
        _CACHE["mean"] = np.load(f"{CACHE}/mean_aligned_gray.npy").astype(np.float64)
    return _CACHE["mean"]


def mean_color():
    return cv2.imread(f"{CACHE}/mean_aligned_color.png")


def to_u8(img):
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


# ------------------------------------------------------------------------------------------
# jitter mapping  (frame f point (x, y) -> mean frame: x - (ax0 + ax1*(y-540)), y - (ay0 + ay1*(y-540)))
# ------------------------------------------------------------------------------------------

def frame_to_mean(P, f):
    P = np.array(P, float).reshape(-1, 2)
    ax0, ax1, ay0, ay1 = seed()["jitter"][f]
    yy = P[:, 1] - H / 2
    return np.column_stack([P[:, 0] - (ax0 + ax1 * yy), P[:, 1] - (ay0 + ay1 * yy)])


def mean_to_frame(P, f):
    """Inverse of frame_to_mean (fixed point, converges in 2-3 iterations)."""
    P = np.array(P, float).reshape(-1, 2)
    Q = P.copy()
    for _ in range(5):
        Q = Q + (P - frame_to_mean(Q, f))
    return Q


# ------------------------------------------------------------------------------------------
# dictionary
# ------------------------------------------------------------------------------------------

def code_bits(mid):
    """4x4 inner bits (1 = white) of dictionary id `mid` in its canonical orientation."""
    return cv2.aruco.Dictionary.getBitsFromByteList(DICT.bytesList[mid:mid + 1], NBITS).astype(int)


def code_grid(mid):
    g = np.zeros((NCELL, NCELL), int)
    g[1:-1, 1:-1] = code_bits(mid)
    return g


_ALL = None


def all_codes():
    """array (1000, 4, 16): inner bits of every id under the 4 rotations np.rot90(bits, r)."""
    global _ALL
    if _ALL is None:
        _ALL = np.array([[np.rot90(code_bits(i), r).ravel() for r in range(4)] for i in range(len(DICT.bytesList))])
    return _ALL


def hamming_table(bits, valid=None):
    """bits: 4x4 read grid (rows/cols in quad order q0->q1 = +x, q0->q3 = +y).
    Returns distances (1000, 4) over the valid cells: D[i, r] = #mismatches between the read
    grid and np.rot90(code_i, r)."""
    b = np.asarray(bits).ravel()
    v = np.ones(16, bool) if valid is None else np.asarray(valid, bool).ravel()
    A = all_codes()
    return ((A != b[None, None, :]) & v[None, None, :]).sum(-1)


# ------------------------------------------------------------------------------------------
# rectification / sampling
# ------------------------------------------------------------------------------------------

def rect_H(quad, C=10.0, M=2.0):
    """Homography rect -> image. The quad corners map to the rect square
    [M*C, (M+6)*C]^2 (q0 at top-left, q1 top-right, q2 bottom-right, q3 bottom-left).
    Returns (H, N) with N the rect patch size."""
    S = NCELL * C
    dst = np.array([[0, 0], [S, 0], [S, S], [0, S]], float) + M * C
    Hm = cv2.getPerspectiveTransform(dst.astype(np.float32), np.asarray(quad, np.float32))
    return Hm, int(round(S + 2 * M * C))


def homog4(src, dst):
    """Exact float64 homography from 4 point pairs (src -> dst)."""
    A = []
    b = []
    for (x, y), (u, v) in zip(np.asarray(src, float), np.asarray(dst, float)):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        b.append(u)
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        b.append(v)
    h = np.linalg.solve(np.array(A), np.array(b))
    return np.r_[h, 1.0].reshape(3, 3)


def apply_H(Hm, P):
    P = np.asarray(P, float).reshape(-1, 2)
    q = np.column_stack([P, np.ones(len(P))]) @ Hm.T
    return q[:, :2] / q[:, 2:3]


def warp(img, Hm, N, order=3):
    """Sample img on the rect grid (N x N) through Hm (rect -> image)."""
    yy, xx = np.mgrid[0:N, 0:N].astype(float)
    P = apply_H(Hm, np.column_stack([xx.ravel(), yy.ravel()]))
    v = map_coordinates(img, [P[:, 1], P[:, 0]], order=order, mode="nearest")
    return v.reshape(N, N)


def cell_samples(img, quad, frac=0.5, n=5):
    """Mean intensity of the central `frac` of each of the 6x6 cells (bilinear sampling of
    n x n points per cell through the quad homography, in marker cell coordinates)."""
    Hm = cv2.getPerspectiveTransform(np.array([[0, 0], [6, 0], [6, 6], [0, 6]], np.float32),
                                     np.asarray(quad, np.float32))
    t = (np.arange(n) + 0.5) / n
    t = 0.5 + (t - 0.5) * frac
    out = np.zeros((NCELL, NCELL))
    for r in range(NCELL):
        for c in range(NCELL):
            u, v = np.meshgrid(c + t, r + t)
            P = apply_H(Hm, np.column_stack([u.ravel(), v.ravel()]))
            out[r, c] = map_coordinates(img, [P[:, 1], P[:, 0]], order=1, mode="nearest").mean()
    return out


def ring_samples(img, quad, inner=0.15, outer=0.5, n=48):
    """Intensity samples of the white margin ring just outside the black square
    (between `inner` and `outer` cells outside the square)."""
    Hm = cv2.getPerspectiveTransform(np.array([[0, 0], [6, 0], [6, 6], [0, 6]], np.float32),
                                     np.asarray(quad, np.float32))
    s = np.linspace(0.3, 5.7, n)
    vals = []
    for off in np.linspace(inner, outer, 3):
        pts = np.concatenate([np.column_stack([s, np.full(n, -off)]), np.column_stack([np.full(n, 6 + off), s]),
                              np.column_stack([s, np.full(n, 6 + off)]), np.column_stack([np.full(n, -off), s])])
        P = apply_H(Hm, pts)
        vals.append(map_coordinates(img, [P[:, 1], P[:, 0]], order=1, mode="nearest").reshape(4, n))
    return np.array(vals)  # (3, 4 sides, n)


def decode(img, quad, cell_ok=None):
    """Read the 6x6 grid inside `quad` (quad order = read order).
    Returns dict with cell means, black/white references, bits (4x4), per-bit contrast margin,
    border check and dictionary distances."""
    cm = cell_samples(img, quad)
    ok = np.ones((NCELL, NCELL), bool) if cell_ok is None else np.asarray(cell_ok, bool)
    border = np.zeros((NCELL, NCELL), bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    ring = ring_samples(img, quad)
    black = float(np.median(cm[border & ok])) if (border & ok).any() else float(np.min(cm))
    white = float(np.median(ring))
    thr = 0.5 * (black + white)
    inner = cm[1:-1, 1:-1]
    bits = (inner > thr).astype(int)
    margin = (inner - thr) / max(white - black, 1e-6)  # +-0.5 = ideal white / black
    valid = ok[1:-1, 1:-1]
    D = hamming_table(bits, valid)
    border_bad = int(((cm > thr) & border & ok).sum())
    return dict(cells=cm, black=black, white=white, thr=thr, bits=bits, margin=margin, valid=valid,
                table=D, border_white_cells=border_bad)


def id_report(dec, expected):
    """Summarise the dictionary match of a decoded grid for the expected id."""
    D = dec["table"]
    nvalid = int(dec["valid"].sum())
    de = D[expected]
    r_exp = int(np.argmin(de))
    d_exp = int(de[r_exp])
    Dm = D.min(1)
    order = np.argsort(Dm, kind="stable")
    best = int(order[0])
    others = Dm.copy()
    others[expected] = 99
    second = int(np.argmin(others))
    n_zero = int((Dm == 0).sum())
    ambiguous = [int(i) for i in np.where(Dm == Dm.min())[0]][:10]
    # rotation of the read grid relative to canonical: read = rot90(code, r)
    return dict(expected=expected, hamming_expected=d_exp, rotation_read=r_exp, n_valid_bits=nvalid,
                best_id=best, best_hamming=int(Dm[best]), best_other_id=second,
                best_other_hamming=int(others[second]), n_ids_at_distance0=n_zero, ids_at_min=ambiguous,
                weakest_bit_margin=float(np.min(np.abs(dec["margin"][dec["valid"]]))) if nvalid else None,
                border_white_cells=dec["border_white_cells"])


def aruco_order(quad, rotation_read):
    """Reorder quad (read order) into ArUco corner order.

    If the grid read in quad order equals np.rot90(code, r) (np.rot90 = counter-clockwise), the
    canonical top-left cell of the code sits at read position [0,0] (r=0), [n-1,0] (r=1),
    [n-1,n-1] (r=2), [0,n-1] (r=3), i.e. ArUco corner 0 = quad[(-r) % 4] and the ArUco order is
    np.roll(quad, r). (Checked against plainly detected markers.)"""
    q = np.asarray(quad, float)
    return np.roll(q, rotation_read, axis=0)


# ------------------------------------------------------------------------------------------
# ArUco-rule sub-pixel corners
# ------------------------------------------------------------------------------------------

def aruco_winsize(quad, params=None):
    p = cv2.aruco.DetectorParameters() if params is None else params
    q = np.asarray(quad, float)
    per = np.sum(np.linalg.norm(np.roll(q, -1, 0) - q, axis=1))
    module = per / 4.0 / (NBITS + 2 * p.markerBorderBits)
    w = max(1, int(round(p.relativeCornerRefinmentWinSize * module)))
    return min(w, p.cornerRefinementWinSize)


def subpix(img, quad, win=None):
    """cv2.cornerSubPix exactly as the ArUco detector does for CORNER_REFINE_SUBPIX
    (window from the module size, 30 iterations, eps 0.1), on a float32 image."""
    q = np.asarray(quad, np.float32).reshape(-1, 1, 2).copy()
    w = aruco_winsize(quad) if win is None else win
    crit = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.1)
    cv2.cornerSubPix(img.astype(np.float32), q, (w, w), (-1, -1), crit)
    return q.reshape(-1, 2).astype(float), w


# ------------------------------------------------------------------------------------------
# edge-based corners
# ------------------------------------------------------------------------------------------

def _profiles(img, Q, nrm, offs):
    X = Q[:, None, 0] + nrm[:, None, 0] * offs[None]
    Y = Q[:, None, 1] + nrm[:, None, 1] * offs[None]
    return map_coordinates(img, [Y.ravel(), X.ravel()], order=3, mode="nearest").reshape(X.shape)


def trace_side(img, a, b, centroid, cell_px, end_frac=0.18, step=1.0, halfwidth=None, sigma=0.8, iters=3):
    """Sub-pixel edge samples of one side of the black square between approximate corners a, b.

    Samples every `step` px along the side (excluding `end_frac` of its length at both ends),
    profile along the outward normal, black -> white transition (positive derivative outward),
    gradient maximum with parabolic interpolation; iterated with the fitted line as new guide.
    Returns dict(points, strength, s, inside, outside, offset, normal_out)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    hw = 0.45 * cell_px if halfwidth is None else halfwidth
    hw = max(hw, 2.5)
    L = np.linalg.norm(b - a)
    d = (b - a) / L
    n = np.array([-d[1], d[0]])
    if np.dot(centroid - 0.5 * (a + b), n) > 0:  # make n point outward
        n = -n
    s = np.arange(end_frac * L, (1 - end_frac) * L + 1e-9, step)
    p0 = a
    res = None
    for it in range(iters):
        Q = p0[None] + s[:, None] * d[None]
        offs = np.arange(-hw, hw + 1e-9, 0.125)
        prof = _profiles(img, Q, np.repeat(n[None], len(s), 0), offs)
        der = gaussian_filter1d(prof, sigma / 0.125, axis=1, order=1) / 0.125  # outward derivative
        j = np.argmax(der[:, 2:-2], axis=1) + 2
        rows = np.arange(len(j))
        y0, y1, y2 = der[rows, j - 1], der[rows, j], der[rows, j + 1]
        den = y0 - 2 * y1 + y2
        delta = np.where(np.abs(den) > 1e-12, 0.5 * (y0 - y2) / den, 0.0)
        delta = np.clip(delta, -0.5, 0.5)
        o = offs[j] + delta * 0.125
        pts = Q + n[None] * o[:, None]
        # intensities inside (black border cell centre) and outside (white margin) of the found edge
        ins = map_coordinates(img, [(pts - n * 0.5 * cell_px)[:, 1], (pts - n * 0.5 * cell_px)[:, 0]], order=1)
        out = map_coordinates(img, [(pts + n * 0.33 * cell_px)[:, 1], (pts + n * 0.33 * cell_px)[:, 0]], order=1)
        at_limit = (j <= 3) | (j >= len(offs) - 4)
        res = dict(points=pts, strength=y1, s=s, offset=o, inside=ins, outside=out, at_limit=at_limit,
                   normal_out=n, dir=d, a=p0)
        # new guide: robust line through the samples
        good = ~at_limit & (y1 > 0.3 * np.median(y1))
        if good.sum() < 5:
            break
        c, dd, _, _ = _tls(pts[good])
        if np.dot(dd, d) < 0:
            dd = -dd
        # re-anchor: project old anchor onto the new line
        p0 = c + np.dot(p0 - c, dd) * dd
        d = dd
        n_new = np.array([-d[1], d[0]])
        n = n_new if np.dot(n_new, n) > 0 else -n_new
        hw = max(2.0, hw * 0.7)
    return res


def _tls(P):
    P = np.asarray(P, float)
    c = P.mean(0)
    U, S, Vt = np.linalg.svd(P - c, full_matrices=False)
    d = Vt[0]
    nn = np.array([-d[1], d[0]])
    r = (P - c) @ nn
    return c, d, float(np.sqrt(np.mean(r ** 2))), r


def _longest_run(b):
    best = cur = 0
    for v in b:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def side_quality(tr, black, white, min_good=0.8, min_run=0.45):
    """Per-sample goodness and side status.

    A sample is good if: the gradient maximum is not at the search limit, the inside (0.5 cell
    inward) is dark and the outside (0.33 cell outward, white margin) bright - polarity / occlusion
    check -, its strength is >= 40 % of the side's median strength (strength collapse) and it lies
    on the robust line (|r| < max(0.6 px, 3 rms)).
    status  "full"    : >= min_good of the samples good, both halves covered, line rms <= 0.35 px
            "partial" : otherwise, if a contiguous run of good samples covers >= min_run of the
                        traced length (the rest is cut by an occluder) and rms <= 0.35 px
            "invalid" : anything else (occluded / cut / not straight)
    near_a / near_b: fraction of good samples in the third of the traced range next to corner a / b
    (tells whether the corner neighbourhood itself is visible)."""
    step = white - black
    ins_ok = tr["inside"] < black + 0.35 * step
    out_ok = tr["outside"] > black + 0.65 * step
    base = ~tr["at_limit"] & ins_ok & out_ok
    smed = np.median(tr["strength"][base]) if base.sum() >= 5 else np.median(tr["strength"])
    st_ok = tr["strength"] > 0.4 * smed
    good = base & st_ok
    rms = np.inf
    if good.sum() >= 5:
        for _ in range(2):
            c, d, rms, _r = _tls(tr["points"][good])
            nn = np.array([-d[1], d[0]])
            r = (tr["points"] - c) @ nn
            good = base & st_ok & (np.abs(r) < max(0.6, 3 * rms))
            if good.sum() < 5:
                break
        if good.sum() >= 5:
            c, d, rms, _ = _tls(tr["points"][good])
    n = len(good)
    frac = float(good.mean()) if n else 0.0
    run = _longest_run(good) / n if n else 0.0
    s = tr["s"]
    mid = 0.5 * (s.min() + s.max()) if n else 0
    cover = bool(n > 3 and good[s < mid].mean() > 0.5 and good[s >= mid].mean() > 0.5)
    third = max(n // 3, 1)
    near_a = float(good[:third].mean()) if n else 0.0
    near_b = float(good[-third:].mean()) if n else 0.0
    if frac >= min_good and cover and rms <= 0.35:
        status = "full"
    elif run >= min_run and good.sum() >= 8 and rms <= 0.35:
        status = "partial"
    else:
        status = "invalid"
    return dict(good=good, frac_good=frac, run_frac=float(run), rms=float(rms), status=status,
                valid=status != "invalid", near_a=near_a, near_b=near_b, strength_median=float(smed),
                inside_ok=float(ins_ok.mean()), outside_ok=float(out_ok.mean()))


def fit_side_line(P, K=None, dist=None):
    """Line through edge points. If K/dist are given the fit is a straight line in undistorted
    normalised coordinates (= slightly curved in the image). Returns (c, d, rms_px) in the
    coordinates used (normalised if K given)."""
    if K is None:
        c, d, rms, _ = _tls(P)
        return c, d, rms
    u = undistort_points(P, K, dist)
    c, d, rms, _ = _tls(u)
    return c, d, rms * K[0, 0]


def intersect(c1, d1, c2, d2):
    A = np.column_stack([d1, -d2])
    t = np.linalg.solve(A, c2 - c1)
    return c1 + t[0] * d1


def to_pixels(pn, K, dist):
    xd, yd = distort_normalized(np.array([pn[0]]), np.array([pn[1]]), dist)
    return np.array([K[0, 0] * xd[0] + K[0, 2], K[1, 1] * yd[0] + K[1, 2]])


def edge_quad(img, quad, K=None, dist=None, end_frac=0.18, step=1.0, iters=2, min_good=0.8):
    """Edge-based refinement of a quad (any consistent order). Iterates trace + line fit +
    intersection. Returns dict(corners (4x2), side (list of 4 dicts), corner_valid (4 bools))."""
    q = np.asarray(quad, float).copy()
    out = None
    for it in range(iters):
        cen = q.mean(0)
        side = np.linalg.norm(np.roll(q, -1, 0) - q, axis=1).mean()
        cell = side / NCELL
        cells = cell_samples(img, q)
        border = np.zeros((NCELL, NCELL), bool)
        border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
        black = float(np.percentile(cells[border], 30))
        white = float(np.median(ring_samples(img, q)))
        sides = []
        lines = []
        for i in range(4):
            a, b = q[i], q[(i + 1) % 4]
            tr = trace_side(img, a, b, cen, cell, end_frac=end_frac, step=step)
            qual = side_quality(tr, black, white, min_good=min_good)
            g = qual["good"]
            if g.sum() >= 5:
                c, d, rms = fit_side_line(tr["points"][g], K, dist)
                cpx, dpx, rmspx, _ = _tls(tr["points"][g])
                sag = _sagitta(tr["points"][g])
            else:
                c = d = cpx = dpx = None
                rms = sag = np.inf
            sides.append(dict(trace=tr, quality=qual, line=(c, d), line_px=(cpx, dpx), rms=rms, sagitta=sag,
                              valid=qual["valid"], black=black, white=white))
            lines.append((c, d))
        newq = q.copy()
        cvalid = np.zeros(4, bool)
        for i in range(4):
            s_prev, s_next = sides[(i - 1) % 4], sides[i]  # corner i is between side i-1 and side i
            if s_prev["line"][0] is None or s_next["line"][0] is None:
                continue
            p = intersect(s_prev["line"][0], s_prev["line"][1], s_next["line"][0], s_next["line"][1])
            if K is not None:
                p = to_pixels(p, K, dist)
            if np.linalg.norm(p - q[i]) < 0.35 * side and s_prev["valid"] and s_next["valid"]:
                newq[i] = p
                cvalid[i] = True
        # corner neighbourhood visible on both adjacent sides (needed for corner-type detectors)
        cvis = np.array([sides[(i - 1) % 4]["quality"]["near_b"] >= 0.6 and sides[i]["quality"]["near_a"] >= 0.6
                         for i in range(4)])
        out = dict(corners=newq, sides=sides, corner_valid=cvalid, corner_visible=cvis, black=black, white=white)
        q = newq
    return out


def _sagitta(P):
    c, d, rms, r = _tls(P)
    t = (P - c) @ d
    if len(t) < 6:
        return 0.0
    qq = np.polyfit(t, r, 2)
    L = t.max() - t.min()
    return float(qq[0] * (L / 2) ** 2)
