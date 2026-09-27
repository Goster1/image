"""Helpers for the cart-80 edge extraction (10_edges_cart80_*.py): geometry guide from the initial
calibration (vanishing directions, back-projection to cart planes), LSD candidates, chaining,
sub-pixel tracing with cleaning (edgelib.trace_edge) and crop rendering."""
from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import map_coordinates

from common import CACHE, load_json, rodrigues, undistort_points, project, SHELF_Z, CART_W, CART_D, FLOOR_Z
import edgelib

IMG = edgelib.mean_image()
REGION = (1120, 0, 1720, 990)  # x0, y0, x1, y1

from scipy.ndimage import spline_filter, gaussian_filter1d

_COEF = spline_filter(IMG, order=3, mode="nearest")  # cubic-spline coefficients, computed once


def trace_edge(poly, halfwidth=6.0, step=2.0, polarity=0, sigma=1.0, iters=3, min_strength=4.0, fit_deg=None):
    """Same algorithm as edgelib.trace_edge (profile along the normal, derivative of Gaussian, parabolic
    sub-pixel peak, iterated), but sampling pre-filtered spline coefficients of the mean image once
    instead of re-filtering the whole image at every call (identical values, ~50x faster)."""
    path = np.asarray(poly, float)
    out = None
    hw = halfwidth
    for it in range(iters):
        Q, nrm, s = edgelib._resample_polyline(path, step)
        offs = np.arange(-hw, hw + 1e-9, 0.25)
        X = Q[:, None, 0] + nrm[:, None, 0] * offs[None]
        Y = Q[:, None, 1] + nrm[:, None, 1] * offs[None]
        prof = map_coordinates(_COEF, [Y.ravel(), X.ravel()], order=3, mode="nearest",
                               prefilter=False).reshape(X.shape)
        der = gaussian_filter1d(prof, sigma / 0.25, axis=1, order=1) / 0.25
        a = np.abs(der) if polarity == 0 else der * polarity
        j = np.argmax(a[:, 2:-2], axis=1) + 2
        rows = np.arange(len(j))
        y0, y1, y2 = a[rows, j - 1], a[rows, j], a[rows, j + 1]
        den = y0 - 2 * y1 + y2
        delta = np.where(np.abs(den) > 1e-12, 0.5 * (y0 - y2) / den, 0.0)
        delta = np.clip(delta, -0.5, 0.5)
        o = offs[j] + delta * 0.25
        pts = Q + nrm * o[:, None]
        strength = y1
        ok = (strength >= min_strength) & (j > 3) & (j < len(offs) - 4)
        out = dict(points=pts[ok], strength=strength[ok], normal=nrm[ok], s=s[ok], offset=o[ok])
        if ok.sum() < 5:
            return out
        if fit_deg is not None:
            ss = out["s"]
            px = np.polyfit(ss, out["points"][:, 0], fit_deg)
            py = np.polyfit(ss, out["points"][:, 1], fit_deg)
            sfine = np.linspace(ss.min(), ss.max(), 50)
            path = np.column_stack([np.polyval(px, sfine), np.polyval(py, sfine)])
        else:
            path = out["points"]
        hw = max(2.5, hw * 0.6)
    return out


def calib80():
    ic = load_json(f"{CACHE}/initial_calib.json")
    K = np.array(ic["K"], float)
    dist = np.array(ic["dist"], float)
    p = np.array(ic["poses"]["80"], float)
    return K, dist, p[:3], p[3:]


K0, D0, RV0, TV0 = calib80()
R0 = rodrigues(RV0)
C0 = -R0.T @ TV0  # camera centre in the cart-80 frame
AX = {"cartX": R0[:, 0], "cartY": R0[:, 1], "cartZ": R0[:, 2]}


def undist_px(uv):
    """distorted pixels -> ideal (undistorted) pixels with the initial calibration."""
    n = undistort_points(np.asarray(uv, float).reshape(-1, 2), K0, D0)
    return np.column_stack([K0[0, 0] * n[:, 0] + K0[0, 2], K0[1, 1] * n[:, 1] + K0[1, 2]])


def vp_dir_at(uv_ideal, axis):
    """unit image direction (ideal pixels) of a line of 3D direction `axis` through ideal point uv."""
    d = AX[axis] if isinstance(axis, str) else np.asarray(axis, float)
    dc = d  # direction already in camera frame (columns of R)
    x = np.asarray(uv_ideal, float)
    # point on the line a little further along the 3D direction: use the vanishing point
    if abs(dc[2]) > 1e-9:
        vp = np.array([K0[0, 0] * dc[0] / dc[2] + K0[0, 2], K0[1, 1] * dc[1] / dc[2] + K0[1, 2]])
        v = vp - x
    else:
        v = np.array([dc[0], dc[1]])
    return v / np.linalg.norm(v)


def classify_dir(p_a, p_b):
    """angle (deg) between the ideal-pixel segment a->b and the directions towards the 3 VPs."""
    a, b = undist_px([p_a, p_b])
    m = 0.5 * (a + b)
    s = (b - a) / np.linalg.norm(b - a)
    out = {}
    for ax in AX:
        v = vp_dir_at(m, ax)
        out[ax] = float(np.degrees(np.arccos(min(1.0, abs(s @ v)))))
    return out


def backproject(uv, Z):
    """distorted pixels -> 3D points (cart-80 frame) on the plane Z = const (initial calibration)."""
    n = undistort_points(np.asarray(uv, float).reshape(-1, 2), K0, D0)
    d = np.column_stack([n, np.ones(len(n))]) @ R0
    s = (Z - C0[2]) / d[:, 2]
    return C0[None] + s[:, None] * d


def backproject_plane(uv, normal_axis, value):
    """intersection with plane coordinate[normal_axis] == value (normal_axis 0=X,1=Y,2=Z)."""
    n = undistort_points(np.asarray(uv, float).reshape(-1, 2), K0, D0)
    d = np.column_stack([n, np.ones(len(n))]) @ R0
    s = (value - C0[normal_axis]) / d[:, normal_axis]
    return C0[None] + s[:, None] * d


def proj80(P):
    return project(np.asarray(P, float).reshape(-1, 3), K0, D0, RV0, TV0)


# ------------------------------------------------------------------------------------------
# LSD candidates
# ------------------------------------------------------------------------------------------

def lsd_segments(minlen=12, box=REGION):
    g = np.clip(IMG, 0, 255).astype(np.uint8)
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV)
    lines = lsd.detect(g)[0].reshape(-1, 4).astype(float)
    out = []
    for s in lines:
        a, b = s[:2], s[2:]
        L = np.hypot(*(b - a))
        m = (a + b) / 2
        if L < minlen or not (box[0] <= m[0] <= box[2] and box[1] <= m[1] <= box[3]):
            continue
        d = (b - a) / L
        n = np.array([-d[1], d[0]])
        ts = np.linspace(0.15, 0.85, 7)
        P = a[None] + (b - a)[None] * ts[:, None]
        ip = map_coordinates(IMG, [P[:, 1] + 3 * n[1], P[:, 0] + 3 * n[0]], order=1)
        im = map_coordinates(IMG, [P[:, 1] - 3 * n[1], P[:, 0] - 3 * n[0]], order=1)
        if np.mean(ip - im) < 0:  # orient: brighter on the left-normal side
            a, b = b, a
        out.append(np.r_[a, b, abs(np.mean(ip - im))])
    return np.array(out)


# ------------------------------------------------------------------------------------------
# tracing + cleaning
# ------------------------------------------------------------------------------------------

def trace_clean(poly, polarity=0, halfwidth=4.0, fit_deg=2, min_rel=0.4, max_res=0.5, step=2.0, sigma=1.0,
                min_strength=3.0, iters=4, img=None):
    """edgelib.trace_edge + removal of weak points (strength < min_rel * median) and of points off a
    smooth low-order curve (|res| > max(max_res, 3 sigma)).  Returns dict or None."""
    poly = np.asarray(poly, float)
    out = trace_edge(poly, halfwidth=halfwidth, step=step, polarity=polarity, sigma=sigma,
                             iters=iters, min_strength=min_strength, fit_deg=fit_deg)
    P, S = out["points"], out["strength"]
    if len(P) < 8:
        return None
    keep = np.ones(len(P), bool)
    for _ in range(5):
        med = np.median(S[keep])
        c, d, _, _ = edgelib.line_fit(P[keep])
        nrm = np.array([-d[1], d[0]])
        t = (P - c) @ d
        r = (P - c) @ nrm
        q = np.polyfit(t[keep], r[keep], min(fit_deg + 1, 3))
        res = r - np.polyval(q, t)
        sd = np.std(res[keep]) if keep.sum() > 10 else max_res
        new = (np.abs(res) < max(max_res, 3.0 * sd)) & (S > min_rel * med)
        if np.array_equal(new, keep):
            break
        keep = new
        if keep.sum() < 8:
            return None
    P, S = P[keep], S[keep]
    # order along the chord
    c, d, _, _ = edgelib.line_fit(P)
    o = np.argsort((P - c) @ d)
    P, S = P[o], S[o]
    # fit residual of a smooth curve (deg 3) = local noise estimate
    t = (P - c) @ d
    r = (P - c) @ np.array([-d[1], d[0]])
    q = np.polyfit(t, r, 3)
    noise = float(np.std(r - np.polyval(q, t)))
    return dict(points=P, strength=S, noise=noise)


def split_gaps(P, S, max_gap=12.0):
    """split an ordered point list at gaps > max_gap px; returns list of (P, S)."""
    d = np.hypot(*np.diff(P, axis=0).T)
    cut = np.where(d > max_gap)[0]
    parts = []
    i0 = 0
    for c in list(cut) + [len(P) - 1]:
        parts.append((P[i0:c + 1], S[i0:c + 1]))
        i0 = c + 1
    return parts


def draw_crop(pts_list, pad=18, scale=3, colours=None, labels=None, box=None, img_color=None, maxside=1400):
    """zoomed crop of the colour mean image around the given point sets, points drawn."""
    if img_color is None:
        img_color = cv2.imread(f"{CACHE}/mean_aligned_color.png")
    allp = np.vstack(pts_list)
    if box is None:
        x0 = int(max(0, np.floor(allp[:, 0].min()) - pad))
        y0 = int(max(0, np.floor(allp[:, 1].min()) - pad))
        x1 = int(min(img_color.shape[1], np.ceil(allp[:, 0].max()) + pad))
        y1 = int(min(img_color.shape[0], np.ceil(allp[:, 1].max()) + pad))
    else:
        x0, y0, x1, y1 = box
    sc = scale
    while max(x1 - x0, y1 - y0) * sc > maxside and sc > 1:
        sc -= 1
    crop = cv2.resize(img_color[y0:y1, x0:x1], None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
    for i, P in enumerate(pts_list):
        col = (0, 0, 255) if colours is None else colours[i]
        for x, y in P:
            u = int(round((x - x0 + 0.5) * sc - 0.5))
            v = int(round((y - y0 + 0.5) * sc - 0.5))
            if 0 <= u < crop.shape[1] and 0 <= v < crop.shape[0]:
                crop[v, u] = col
                if sc >= 3:
                    crop[max(v - 1, 0), u] = col
        if labels is not None:
            x, y = P[len(P) // 2]
            cv2.putText(crop, labels[i], (int((x - x0) * sc) + 4, int((y - y0) * sc) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    return crop, (x0, y0, sc)


# ------------------------------------------------------------------------------------------
# chaining of LSD segments (in ideal = undistorted pixel coordinates of the initial calibration)
# ------------------------------------------------------------------------------------------

def chain_segments(rows, ang_tol=2.0, dist_tol=1.5, gap_max=60.0, min_span=40.0, minlen=10.0):
    """rows: x0,y0,x1,y1,contrast,class,... (oriented).  Greedy collinear chaining of segments with
    the same orientation (polarity) in undistorted coordinates.  Returns list of dicts."""
    segs = rows[:, :4]
    lens = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    ok = lens >= minlen
    A = undist_px(segs[:, :2])
    B = undist_px(segs[:, 2:4])
    D = (B - A) / np.linalg.norm(B - A, axis=1, keepdims=True)
    order = [i for i in np.argsort(-lens) if ok[i]]
    used = np.zeros(len(segs), bool)
    chains = []
    for i0 in order:
        if used[i0]:
            continue
        mem = [i0]
        used[i0] = True
        changed = True
        while changed:
            changed = False
            P = np.vstack([A[mem], B[mem]])
            w = np.r_[lens[mem], lens[mem]]
            dsum = np.sum(D[mem] * lens[mem][:, None], axis=0)
            d = dsum / np.linalg.norm(dsum)
            nrm = np.array([-d[1], d[0]])
            c = np.average(P, axis=0, weights=w)
            t = (P - c) @ d
            for k in order:
                if used[k]:
                    continue
                if D[k] @ d < np.cos(np.radians(ang_tol)):
                    continue
                ra, rb = (A[k] - c) @ nrm, (B[k] - c) @ nrm
                ta, tb = (A[k] - c) @ d, (B[k] - c) @ d
                ext = max(0.0, max(t.min() - max(ta, tb), min(ta, tb) - t.max()))
                if ext > gap_max:
                    continue
                if max(abs(ra), abs(rb)) > dist_tol + 0.004 * ext:
                    continue
                mem.append(k)
                used[k] = True
                changed = True
                break
        P = np.vstack([A[mem], B[mem]])
        dsum = np.sum(D[mem] * lens[mem][:, None], axis=0)
        d = dsum / np.linalg.norm(dsum)
        t = (P - P.mean(0)) @ d
        span = t.max() - t.min()
        if span < min_span:
            continue
        Pd = np.vstack([segs[mem, :2], segs[mem, 2:4]])
        o = np.argsort(t)
        chains.append(dict(members=[int(m) for m in mem], poly=Pd[o], span=float(span),
                           cls=int(np.bincount(rows[mem, 5].astype(int), weights=lens[mem]).argmax()),
                           covered=float(lens[mem].sum())))
    return chains


def guide_from_points(P, deg=2, n=40, extend=0.0):
    """smooth guide polyline (deg-2 in the chord frame) through ordered points, optionally extended."""
    P = np.asarray(P, float)
    c, d, _, _ = edgelib.line_fit(P)
    nrm = np.array([-d[1], d[0]])
    t = (P - c) @ d
    r = (P - c) @ nrm
    deg = min(deg, len(P) - 1)
    q = np.polyfit(t, r, deg)
    tt = np.linspace(t.min() - extend, t.max() + extend, n)
    return c[None] + tt[:, None] * d[None] + np.polyval(q, tt)[:, None] * nrm[None]


def orient_like(poly, a, b):
    """return poly oriented from a towards b."""
    poly = np.asarray(poly, float)
    if np.dot(poly[-1] - poly[0], np.asarray(b) - np.asarray(a)) < 0:
        return poly[::-1]
    return poly


def _contiguous_run(good, core_lo, core_hi, max_bad=4):
    """indices [lo, hi] of the run of good points around the core, tolerating <= max_bad bad in a row."""
    n = len(good)
    hi = core_hi
    bad = 0
    j = core_hi + 1
    while j < n:
        if good[j]:
            hi = j
            bad = 0
        else:
            bad += 1
            if bad > max_bad:
                break
        j += 1
    lo = core_lo
    bad = 0
    j = core_lo - 1
    while j >= 0:
        if good[j]:
            lo = j
            bad = 0
        else:
            bad += 1
            if bad > max_bad:
                break
        j -= 1
    return lo, hi


def grow_trace(poly, polarity, step_ext=40.0, max_rounds=25, halfwidth=3.5, min_rel=0.45, fit_deg=2,
               bounds=None, max_res=0.5, res_tol=0.7, max_bad=4, no_grow=False):
    """trace along `poly`, then repeatedly extend the guide at both ends (quadratic extrapolation of the
    traced points) while the edge continues: new points must have strength >= min_rel * median of the
    core and lie within res_tol px of the smooth curve of the already accepted points."""
    poly = np.asarray(poly, float)
    res = trace_clean(poly, polarity=polarity, halfwidth=halfwidth, fit_deg=fit_deg, min_rel=min_rel,
                      max_res=max_res)
    if res is None:
        return None
    P = orient_like(res["points"], poly[0], poly[-1])
    core_med = float(np.median(res["strength"]))
    if not no_grow:
        grown = [True, True]
        for _ in range(max_rounds):
            if not any(grown):
                break
            c, d, _, _ = edgelib.line_fit(P)
            if d @ (P[-1] - P[0]) < 0:
                d = -d
            nrm = np.array([-d[1], d[0]])
            t = (P - c) @ d
            r = (P - c) @ nrm
            span = t.max() - t.min()
            q = np.polyfit(t, r, 2 if span > 120 else 1)
            ext = [step_ext if grown[0] else 0.0, step_ext if grown[1] else 0.0]
            t0, t1 = t.min() - ext[0], t.max() + ext[1]
            tt = np.arange(t0, t1 + 1e-9, 2.0)
            G = c[None] + tt[:, None] * d[None] + np.polyval(q, tt)[:, None] * nrm[None]
            if bounds is not None:
                inb = (G[:, 0] > bounds[0]) & (G[:, 0] < bounds[2]) & (G[:, 1] > bounds[1]) & (G[:, 1] < bounds[3])
                G = G[inb]
                tt = tt[inb]
            if len(G) < 10:
                break
            out = trace_edge(G, halfwidth=2.5, step=2.0, polarity=polarity, sigma=1.0, iters=1,
                                     min_strength=-1e9)
            Pn, Sn = out["points"], out["strength"]
            tn = (Pn - c) @ d
            rn = (Pn - c) @ nrm - np.polyval(q, tn)
            good = (Sn >= min_rel * core_med) & (np.abs(rn) < res_tol)
            core = np.where((tn >= t.min() - 1) & (tn <= t.max() + 1))[0]
            if len(core) == 0:
                break
            lo, hi = _contiguous_run(good, core.min(), core.max(), max_bad=max_bad)
            newlo = tn[lo] < t.min() - 3
            newhi = tn[hi] > t.max() + 3
            grown = [grown[0] and newlo and tn[lo] <= t.min() - 0.6 * ext[0] + 1e-6,
                     grown[1] and newhi and tn[hi] >= t.max() + 0.6 * ext[1] - 1e-6]
            sel = np.arange(lo, hi + 1)
            sel = sel[good[sel]]
            addlo = sel[tn[sel] < t.min() - 1]
            addhi = sel[tn[sel] > t.max() + 1]
            P = np.vstack([Pn[addlo], P, Pn[addhi]])
            if len(addlo) == 0:
                grown[0] = False
            if len(addhi) == 0:
                grown[1] = False
    g = orient_like(guide_from_points(P, deg=3 if len(P) > 80 else 2, n=max(40, len(P) // 3)), poly[0], poly[-1])
    res = trace_clean(g, polarity=polarity, halfwidth=2.5, fit_deg=3 if len(P) > 80 else 2, min_rel=min_rel,
                      max_res=max_res, iters=2)
    if res is not None and np.dot(res["points"][-1] - res["points"][0], poly[-1] - poly[0]) < 0:
        res["points"] = res["points"][::-1].copy()
        res["strength"] = res["strength"][::-1].copy()
    return res


_COLOR = None


def color_image():
    global _COLOR
    if _COLOR is None:
        _COLOR = cv2.imread(f"{CACHE}/mean_aligned_color.png")
    return _COLOR


def edge_strip(P, label="", win=64, scale=4, nwin=None, others=None):
    """review image for one edge: an overview panel + `nwin` zoomed windows (x`scale`) along the edge
    with the sub-pixel points drawn (red).  `others`: optional list of other point sets (drawn cyan)."""
    img = color_image()
    P = np.asarray(P, float)
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    if nwin is None:
        nwin = int(np.clip(round(s[-1] / 170) + 1, 2, 5))
    fr = np.linspace(0.04, 0.96, nwin)
    panels = []
    H = win * scale
    # overview
    x0, y0 = np.floor(P.min(0)).astype(int) - 25
    x1, y1 = np.ceil(P.max(0)).astype(int) + 25
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, img.shape[1]), min(y1, img.shape[0])
    x0, y0 = max(x0 - 40, 0), max(y0 - 40, 0)
    x1, y1 = min(x1 + 40, img.shape[1]), min(y1 + 40, img.shape[0])
    ov = img[y0:y1, x0:x1].copy()
    f = min(H / ov.shape[0], 1.2 * H / ov.shape[1])
    ov = cv2.resize(ov, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)
    Q = (P - [x0, y0]) * f
    cv2.polylines(ov, [np.round(Q * 4).astype(np.int32)], False, (0, 0, 255), 1, cv2.LINE_AA, shift=2)
    for k in fr:
        c = P[np.searchsorted(s, k * s[-1]).clip(0, len(P) - 1)]
        cv2.rectangle(ov, tuple(np.round((c - win / 2 - [x0, y0]) * f).astype(int)),
                      tuple(np.round((c + win / 2 - [x0, y0]) * f).astype(int)), (0, 255, 255), 1)
    pad = np.zeros((H, ov.shape[1], 3), np.uint8)
    pad[:ov.shape[0]] = ov[:H]
    cv2.putText(pad, label, (4, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    panels.append(pad)
    for k in fr:
        c = P[np.searchsorted(s, k * s[-1]).clip(0, len(P) - 1)]
        wx0, wy0 = int(round(c[0] - win / 2)), int(round(c[1] - win / 2))
        wx0 = int(np.clip(wx0, 0, img.shape[1] - win))
        wy0 = int(np.clip(wy0, 0, img.shape[0] - win))
        crop = cv2.resize(img[wy0:wy0 + win, wx0:wx0 + win], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        sets = [(P, (0, 0, 255))] + [(o, (255, 255, 0)) for o in (others or [])]
        for S_, col in sets:
            for x, y in np.asarray(S_, float):
                u = int(round((x - wx0 + 0.5) * scale - 0.5))
                v = int(round((y - wy0 + 0.5) * scale - 0.5))
                if 1 <= u < crop.shape[1] - 1 and 1 <= v < crop.shape[0] - 1:
                    crop[v - 1:v + 2, u - 1:u + 2] = col
        cv2.rectangle(crop, (0, 0), (crop.shape[1] - 1, crop.shape[0] - 1), (80, 80, 80), 1)
        panels.append(crop)
    return np.hstack(panels)


def montage(strips, width=None):
    width = width or max(s.shape[1] for s in strips)
    rows = []
    for s in strips:
        r = np.zeros((s.shape[0], width, 3), np.uint8)
        r[:, :min(width, s.shape[1])] = s[:, :width]
        rows.append(r)
        rows.append(np.full((4, width, 3), 255, np.uint8))
    return np.vstack(rows)
