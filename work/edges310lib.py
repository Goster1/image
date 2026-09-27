"""Helpers for the cart-310 edge extraction (10_edges_cart310_*.py): LSD chaining, sub-pixel
tracing with cleaning, crop rendering.  Uses edgelib.trace_edge for the sub-pixel localisation.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import map_coordinates

from common import CACHE, load_json, rodrigues, undistort_points, project, SHELF_Z, CART_W, CART_D, FLOOR_Z
import edgelib

IMG = edgelib.mean_image()


def calib310():
    ic = load_json(f"{CACHE}/initial_calib.json")
    K = np.array(ic["K"], float)
    dist = np.array(ic["dist"], float)
    p = np.array(ic["poses"]["310"], float)
    return K, dist, p[:3], p[3:]


# ------------------------------------------------------------------------------------------
# LSD + polarity + chaining
# ------------------------------------------------------------------------------------------

def lsd_segments(minlen=12, box=(150, 0, 900, 1010)):
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
        if np.mean(ip - im) < 0:  # orient so that the image is brighter on the left normal side
            a, b = b, a
        out.append(np.r_[a, b])
    return np.array(out)


def chain_segments(segs, ang_tol=2.5, dist_tol=1.8, gap_max=70.0, min_span=60.0):
    """Greedy collinear chaining of oriented segments (curvature tolerated via quadratic fit)."""
    L = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    order = np.argsort(-L)
    used = np.zeros(len(segs), bool)
    chains = []
    for i0 in order:
        if used[i0]:
            continue
        members = [i0]
        used[i0] = True
        changed = True
        while changed:
            changed = False
            P = np.vstack([segs[members, :2], segs[members, 2:]])
            # principal axis from the member segments (weighted by length)
            dsum = np.sum([(segs[k, 2:] - segs[k, :2]) for k in members], axis=0)
            d = dsum / np.linalg.norm(dsum)
            nrm = np.array([-d[1], d[0]])
            c = P.mean(0)
            t = (P - c) @ d
            r = (P - c) @ nrm
            span = t.max() - t.min()
            deg = 2 if span > 150 and len(members) >= 3 else 1
            q = np.polyfit(t, r, deg)
            dq = np.polyder(q)
            for k in order:
                if used[k]:
                    continue
                a, b = segs[k, :2], segs[k, 2:]
                sd = (b - a) / np.linalg.norm(b - a)
                ta, tb = (a - c) @ d, (b - c) @ d
                # local chain direction at the segment position
                tm = 0.5 * (ta + tb)
                slope = np.polyval(dq, np.clip(tm, t.min() - 100, t.max() + 100)) if deg > 1 else q[0]
                ld = d + slope * nrm
                ld /= np.linalg.norm(ld)
                if sd @ ld < np.cos(np.radians(ang_tol)):
                    continue
                ra, rb = (a - c) @ nrm, (b - c) @ nrm
                ea = abs(ra - np.polyval(q, ta))
                eb = abs(rb - np.polyval(q, tb))
                ext = max(0.0, max(t.min() - max(ta, tb), min(ta, tb) - t.max()))
                tol = dist_tol + 0.01 * ext
                if ea > tol or eb > tol:
                    continue
                if ext > gap_max:
                    continue
                members.append(k)
                used[k] = True
                changed = True
        P = np.vstack([segs[members, :2], segs[members, 2:]])
        dsum = np.sum([(segs[k, 2:] - segs[k, :2]) for k in members], axis=0)
        d = dsum / np.linalg.norm(dsum)
        t = (P - P.mean(0)) @ d
        if t.max() - t.min() < min_span:
            continue
        o = np.argsort(t)
        chains.append(dict(members=[int(m) for m in members], poly=P[o], span=float(t.max() - t.min()),
                           dir=d))
    return chains


# ------------------------------------------------------------------------------------------
# tracing + cleaning
# ------------------------------------------------------------------------------------------

def trace_clean(poly, polarity=1, halfwidth=4.0, fit_deg=2, min_rel=0.35, max_res=0.6, step=2.0, sigma=1.0,
                min_strength=3.0, clip=None):
    """trace_edge + outlier / weak-point removal.  Returns dict or None.

    clip: optional (s0, s1) arc-length window (fractions of the guide length) to keep.
    """
    poly = np.asarray(poly, float)
    out = edgelib.trace_edge(poly, halfwidth=halfwidth, step=step, polarity=polarity, sigma=sigma, iters=4,
                             min_strength=min_strength, fit_deg=fit_deg)
    P, S = out["points"], out["strength"]
    if len(P) < 8:
        return None
    # iterate: smooth fit (deg fit_deg+1 in chord coordinate), reject residual outliers and weak points
    keep = np.ones(len(P), bool)
    for _ in range(4):
        med = np.median(S[keep])
        c, d, _, _ = edgelib.line_fit(P[keep])
        nrm = np.array([-d[1], d[0]])
        t = (P - c) @ d
        r = (P - c) @ nrm
        q = np.polyfit(t[keep], r[keep], min(fit_deg + 1, 3))
        res = r - np.polyval(q, t)
        new = (np.abs(res) < max(max_res, 3.0 * np.std(res[keep]) if keep.sum() > 10 else max_res)) & (S > min_rel * med)
        new &= np.abs(res) < 1.5
        if (new == keep).all():
            break
        keep = new
    P, S = P[keep], S[keep]
    if len(P) < 8:
        return None
    # order along the chord, keeping the direction of the guide (polarity convention)
    c, d, _, _ = edgelib.line_fit(P)
    if d @ (poly[-1] - poly[0]) < 0:
        d = -d
    t = (P - c) @ d
    o = np.argsort(t)
    return dict(points=P[o], strength=S[o], t=t[o])


def split_gaps(tr, max_gap=8.0, min_pts=10):
    """split a traced edge where consecutive points are further apart than max_gap (occlusions)."""
    P, S = tr["points"], tr["strength"]
    g = np.hypot(*np.diff(P, axis=0).T)
    cuts = np.nonzero(g > max_gap)[0]
    parts, s0 = [], 0
    for cidx in list(cuts) + [len(P) - 1]:
        seg = slice(s0, cidx + 1)
        if cidx + 1 - s0 >= min_pts:
            parts.append(dict(points=P[seg], strength=S[seg]))
        s0 = cidx + 1
    return parts


def chord_stats(P):
    c, d, rms, mx = edgelib.line_fit(P)
    bow, _ = edgelib.sagitta(P)
    return rms, bow, float(np.hypot(*(P[-1] - P[0])))


# ------------------------------------------------------------------------------------------
# model helpers
# ------------------------------------------------------------------------------------------

def project_line(P0, P1, n=60):
    K, dist, rv, tv = calib310()
    t = np.linspace(0, 1, n)[:, None]
    X = np.asarray(P0, float)[None] * (1 - t) + np.asarray(P1, float)[None] * t
    return project(X, K, dist, rv, tv)


def direction_angles(P):
    """angle (deg) between the undistorted edge (initial calib) and the directions to the
    vanishing points of cart X, Y, Z, averaged over the edge (depth independent)."""
    K, dist, rv, tv = calib310()
    R = rodrigues(rv)
    n = undistort_points(P, K, dist)
    c, d, _, _ = edgelib.line_fit(n)
    out = {}
    for k, v in zip("XYZ", np.eye(3)):
        w3 = R @ v
        if abs(w3[2]) > 1e-9:
            w = w3[:2] / w3[2] - c
        else:
            w = w3[:2]
        w /= np.linalg.norm(w)
        out[k] = float(np.degrees(np.arccos(min(1.0, abs(d @ w)))))
    # nadir = world vertical ~ cart -Z
    return out


# ------------------------------------------------------------------------------------------
# rendering
# ------------------------------------------------------------------------------------------
COLOR_IMG = None


def color_image():
    global COLOR_IMG
    if COLOR_IMG is None:
        COLOR_IMG = cv2.imread(f"{CACHE}/mean_aligned_color.png")
    return COLOR_IMG


def render_crop(P, path, label="", scale=4, pad=18, maxside=900, extra=None):
    """zoomed crop around the traced points; draws the points as small dots (sub-pixel)."""
    img = color_image()
    P = np.asarray(P, float)
    x0 = int(max(0, np.floor(P[:, 0].min() - pad)))
    x1 = int(min(img.shape[1], np.ceil(P[:, 0].max() + pad)))
    y0 = int(max(0, np.floor(P[:, 1].min() - pad)))
    y1 = int(min(img.shape[0], np.ceil(P[:, 1].max() + pad)))
    crop = img[y0:y1, x0:x1]
    s = min(scale, maxside / max(crop.shape[:2]))
    s = max(s, 1.0)
    big = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    for (x, y) in P:
        cx, cy = (x - x0 + 0.5) * s - 0.5, (y - y0 + 0.5) * s - 0.5
        cv2.circle(big, (int(round(cx * 4)), int(round(cy * 4))), int(max(4, s * 2)), (0, 0, 255), -1,
                   cv2.LINE_AA, shift=2)
    if extra is not None:
        for Q, col in extra:
            Q = np.asarray(Q, float)
            pts = np.round(((Q - [x0, y0] + 0.5) * s - 0.5) * 4).astype(np.int32)
            cv2.polylines(big, [pts], False, col, 1, cv2.LINE_AA, shift=2)
    cv2.putText(big, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
    cv2.putText(big, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    cv2.imwrite(path, big)
    return (x0, y0, s)


def local_crops(P, path, label="", n=3, half=22, scale=6):
    """montage of n small high-zoom windows along the edge (start / middle / end) to check the
    sub-pixel position on the actual edge."""
    img = color_image()
    P = np.asarray(P, float)
    idx = np.linspace(0, len(P) - 1, n).round().astype(int)
    tiles = []
    for i in idx:
        cx, cy = P[i]
        x0, y0 = int(round(cx)) - half, int(round(cy)) - half
        x0 = min(max(0, x0), img.shape[1] - 2 * half)
        y0 = min(max(0, y0), img.shape[0] - 2 * half)
        crop = img[y0:y0 + 2 * half, x0:x0 + 2 * half]
        big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        sel = (P[:, 0] >= x0) & (P[:, 0] < x0 + 2 * half) & (P[:, 1] >= y0) & (P[:, 1] < y0 + 2 * half)
        for (x, y) in P[sel]:
            px, py = (x - x0 + 0.5) * scale - 0.5, (y - y0 + 0.5) * scale - 0.5
            cv2.circle(big, (int(round(px * 4)), int(round(py * 4))), 8, (0, 0, 255), -1, cv2.LINE_AA, shift=2)
        tiles.append(big)
        tiles.append(np.full((big.shape[0], 4, 3), 255, np.uint8))
    m = np.hstack(tiles[:-1])
    cv2.putText(m, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
    cv2.putText(m, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    cv2.imwrite(path, m)


# ------------------------------------------------------------------------------------------
# edge growing (follow an edge along its full visible length)
# ------------------------------------------------------------------------------------------

def _profile_points(Q, nrm, polarity, hw=2.5, sigma=1.0):
    """one-shot sub-pixel localisation along the normals of guide points Q (no iteration)."""
    from scipy.ndimage import gaussian_filter1d
    offs = np.arange(-hw - 2, hw + 2 + 1e-9, 0.25)
    X = Q[:, None, 0] + nrm[:, None, 0] * offs[None]
    Y = Q[:, None, 1] + nrm[:, None, 1] * offs[None]
    prof = map_coordinates(IMG, [Y.ravel(), X.ravel()], order=3, mode="nearest").reshape(X.shape)
    der = gaussian_filter1d(prof, sigma / 0.25, axis=1, order=1) / 0.25
    a = der * polarity
    inner = np.abs(offs) <= hw
    a_in = np.where(inner[None], a, -np.inf)
    j = np.argmax(a_in, axis=1)
    rows = np.arange(len(j))
    y0, y1, y2 = a[rows, j - 1], a[rows, j], a[rows, j + 1]
    den = y0 - 2 * y1 + y2
    delta = np.where(np.abs(den) > 1e-12, 0.5 * (y0 - y2) / den, 0.0)
    delta = np.clip(delta, -0.5, 0.5)
    o = offs[j] + delta * 0.25
    # a true local maximum inside the window (not at the window border)
    ok = (np.abs(offs[j]) < hw - 0.2)
    return Q + nrm * o[:, None], y1, ok


def grow(P, polarity=1, rel=0.45, max_gap=14.0, step=2.0, hw=2.0, local=60.0, max_len=2000, stop_box=None,
         abs_min=4.0, max_dev=0.9):
    """Extend an ordered sub-pixel edge (Nx2) at both ends while the edge stays strong and on the
    locally extrapolated path.  Gaps up to max_gap px (occluders) are bridged if the edge resumes.
    Returns the extended ordered point list and strengths."""
    P = np.asarray(P, float)
    S = map_strength(P, polarity)
    med = np.median(S)
    for end in (0, 1):
        for _ in range(400):
            Pe = P if end == 1 else P[::-1]
            Se = S if end == 1 else S[::-1]
            # local line fit near this end
            tail = Pe[-1]
            dist_from_end = np.hypot(*(Pe - tail).T)
            sel = dist_from_end < local
            if sel.sum() < 5:
                sel = np.arange(len(Pe)) >= len(Pe) - 8
            c, d, _, _ = edgelib.line_fit(Pe[sel])
            if (Pe[-1] - Pe[sel][0]) @ d < 0:
                d = -d
            # local quadratic in the local frame for curvature
            nrm_l = np.array([-d[1], d[0]])
            tt = (Pe[sel] - c) @ d
            rr = (Pe[sel] - c) @ nrm_l
            q = np.polyfit(tt, rr, 2 if np.ptp(tt) > 40 else 1)
            t_end = (tail - c) @ d
            ts = t_end + step * np.arange(1, int(max_gap / step) + 2)
            Q = c[None] + ts[:, None] * d[None] + np.polyval(q, ts)[:, None] * nrm_l[None]
            # normal of the guide (consistent with the polarity convention of trace_edge)
            gd = d if end == 1 else -d
            nrm = np.array([-gd[1], gd[0]])
            nrm = np.repeat(nrm[None], len(Q), 0)
            pts, st, ok = _profile_points(Q, nrm, polarity, hw=hw)
            dev = np.abs((pts - Q) @ nrm[0])
            good = ok & (st > rel * med) & (st > abs_min) & (dev < max_dev)
            inside = (pts[:, 0] > 2) & (pts[:, 0] < IMG.shape[1] - 3) & (pts[:, 1] > 2) & (pts[:, 1] < IMG.shape[0] - 3)
            good &= inside
            if stop_box is not None:
                good &= (pts[:, 0] >= stop_box[0]) & (pts[:, 0] <= stop_box[2]) & (pts[:, 1] >= stop_box[1]) & (pts[:, 1] <= stop_box[3])
            if not good.any():
                break
            k = np.nonzero(good)[0][0]
            # accept the first good point (gap bridged if k > 0) and the contiguous good run after it
            k2 = k
            while k2 + 1 < len(good) and good[k2 + 1]:
                k2 += 1
            k2 = min(k2, k + 4)
            newp = pts[k:k2 + 1]
            news = st[k:k2 + 1]
            if end == 0:
                newp, news = newp[::-1], news[::-1]
            if end == 1:
                P = np.vstack([P, newp])
                S = np.r_[S, news]
            else:
                P = np.vstack([newp, P])
                S = np.r_[news, S]
            if np.hypot(*(P[-1] - P[0])) > max_len:
                break
    return P, S


def map_strength(P, polarity, sigma=1.0):
    """gradient strength (derivative of Gaussian along the local normal) at the given points."""
    P = np.asarray(P, float)
    d = np.gradient(P, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
    nrm = np.column_stack([-d[:, 1], d[:, 0]])
    _, st, _ = _profile_points(P, nrm, polarity, hw=1.0, sigma=sigma)
    return st


def resample_path(P, step=2.0):
    P = np.asarray(P, float)
    seg = np.hypot(*np.diff(P, axis=0).T)
    s = np.r_[0, np.cumsum(seg)]
    n = max(int(s[-1] // step) + 1, 2)
    t = np.linspace(0, s[-1], n)
    return np.column_stack([np.interp(t, s, P[:, 0]), np.interp(t, s, P[:, 1])])


# ------------------------------------------------------------------------------------------
# straightened strips for visual verification
# ------------------------------------------------------------------------------------------

def curve_from_points(P, deg=2):
    """smooth guide y(t) in the chord frame of P; returns (c, d, nrm, coef, tmin, tmax)."""
    P = np.asarray(P, float)
    c, d, _, _ = edgelib.line_fit(P)
    if d @ (P[-1] - P[0]) < 0:
        d = -d
    nrm = np.array([-d[1], d[0]])
    t = (P - c) @ d
    r = (P - c) @ nrm
    q = np.polyfit(t, r, min(deg, max(1, len(P) // 10)))
    return c, d, nrm, q, t.min(), t.max()


def eval_curve(cv, t):
    c, d, nrm, q, _, _ = cv
    t = np.asarray(t, float)
    return c[None] + t[:, None] * d[None] + np.polyval(q, t)[:, None] * nrm[None]


def strip_image(cv, t0, t1, hw=10, sperp=5, salong=1.5, pts=None, rowlen=420, marks=None):
    """straightened colour strip along the guide curve cv between t0..t1 (chord coords).
    Vertical axis: offset along the chord normal (+ = left normal, drawn upwards), scaled sperp.
    pts: sub-pixel edge points to draw (red)."""
    img = color_image().astype(np.float32)
    c, d, nrm, q, _, _ = cv
    ts = np.arange(t0, t1, 1.0 / salong)
    offs = np.arange(hw, -hw - 1e-9, -1.0 / sperp)
    base = eval_curve(cv, ts)
    X = base[None, :, 0] + offs[:, None] * nrm[0]
    Y = base[None, :, 1] + offs[:, None] * nrm[1]
    chans = [map_coordinates(img[..., k], [Y.ravel(), X.ravel()], order=1, mode="nearest").reshape(X.shape)
             for k in range(3)]
    S = np.clip(np.dstack(chans), 0, 255).astype(np.uint8)
    S = np.ascontiguousarray(S)
    if pts is not None and len(pts):
        pts = np.asarray(pts, float)
        tp = (pts - c) @ d
        rp = (pts - c) @ nrm - np.polyval(q, tp)
        for a, b in zip(tp, rp):
            u = (a - t0) * salong
            v = (hw - b) * sperp
            if 0 <= u < S.shape[1]:
                cv2.circle(S, (int(round(u * 4)), int(round(v * 4))), 5, (0, 0, 255), -1, cv2.LINE_AA, shift=2)
    # centre line ticks every 50 px
    for tt in np.arange(np.ceil(t0 / 50) * 50, t1, 50):
        u = int((tt - t0) * salong)
        cv2.line(S, (u, 0), (u, 4), (0, 255, 255), 1)
    rows = []
    for u0 in range(0, S.shape[1], rowlen):
        r = S[:, u0:u0 + rowlen]
        if r.shape[1] < rowlen:
            r = np.hstack([r, np.zeros((r.shape[0], rowlen - r.shape[1], 3), np.uint8)])
        rows.append(r)
        rows.append(np.full((3, rowlen, 3), 255, np.uint8))
    return np.vstack(rows[:-1])


# ------------------------------------------------------------------------------------------
# thin-line (valley / ridge) centre tracer: same scheme as edgelib.trace_edge but locating the
# extremum of the smoothed intensity profile instead of the extremum of its derivative
# ------------------------------------------------------------------------------------------

def trace_line_centre(poly, dark=True, halfwidth=3.0, step=2.0, sigma=0.8, iters=4, fit_deg=3, min_contrast=3.0,
                      side=2.5):
    from scipy.ndimage import gaussian_filter1d
    path = np.asarray(poly, float)
    hw = halfwidth
    out = None
    for it in range(iters):
        Q, nrm, s = edgelib._resample_polyline(path, step)
        offs = np.arange(-hw - side - 1, hw + side + 1 + 1e-9, 0.25)
        X = Q[:, None, 0] + nrm[:, None, 0] * offs[None]
        Y = Q[:, None, 1] + nrm[:, None, 1] * offs[None]
        prof = map_coordinates(IMG, [Y.ravel(), X.ravel()], order=3, mode="nearest").reshape(X.shape)
        sm = gaussian_filter1d(prof, sigma / 0.25, axis=1)
        a = -sm if dark else sm
        inner = np.abs(offs) <= hw
        a_in = np.where(inner[None], a, -np.inf)
        j = np.argmax(a_in, axis=1)
        rows = np.arange(len(j))
        y0, y1, y2 = a[rows, j - 1], a[rows, j], a[rows, j + 1]
        den = y0 - 2 * y1 + y2
        delta = np.where(np.abs(den) > 1e-12, 0.5 * (y0 - y2) / den, 0.0)
        delta = np.clip(delta, -0.5, 0.5)
        o = offs[j] + delta * 0.25
        # contrast: centre vs. mean of the profile at +-side px
        k = int(round(side / 0.25))
        jl = np.clip(j - k, 0, len(offs) - 1)
        jr = np.clip(j + k, 0, len(offs) - 1)
        contrast = 0.5 * (a[rows, jl] + a[rows, jr])
        contrast = y1 - contrast
        pts = Q + nrm * o[:, None]
        ok = (contrast >= min_contrast) & (np.abs(offs[j]) < hw - 0.2)
        out = dict(points=pts[ok], strength=contrast[ok], normal=nrm[ok], s=s[ok], offset=o[ok])
        if ok.sum() < 5:
            return out
        ss = out["s"]
        px = np.polyfit(ss, out["points"][:, 0], fit_deg)
        py = np.polyfit(ss, out["points"][:, 1], fit_deg)
        sfine = np.linspace(ss.min(), ss.max(), 60)
        path = np.column_stack([np.polyval(px, sfine), np.polyval(py, sfine)])
        hw = max(1.5, hw * 0.6)
    return out
