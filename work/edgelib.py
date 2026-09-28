"""Sub-pixel edge tools shared by all edge-based methods.

trace_edge(): given an approximate polyline along an intensity edge, sample profiles along the
normal every `step` px, locate the gradient extremum with sub-pixel (parabolic) precision, and
iterate with a smooth fitted path.  Works on the jitter-compensated mean image
(work/cache/mean_aligned_gray.npy) by default.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from common import CACHE

_IMG = None


def mean_image(smooth=0.0):
    global _IMG
    if _IMG is None:
        _IMG = np.load(f"{CACHE}/mean_aligned_gray.npy").astype(np.float64)
    return gaussian_filter(_IMG, smooth) if smooth > 0 else _IMG


def _resample_polyline(P, step):
    P = np.asarray(P, float)
    seg = np.hypot(*np.diff(P, axis=0).T)
    s = np.r_[0, np.cumsum(seg)]
    n = max(int(s[-1] // step) + 1, 2)
    t = np.linspace(0, s[-1], n)
    x = np.interp(t, s, P[:, 0])
    y = np.interp(t, s, P[:, 1])
    Q = np.column_stack([x, y])
    d = np.gradient(Q, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
    nrm = np.column_stack([-d[:, 1], d[:, 0]])
    return Q, nrm, t


def trace_edge(poly, img=None, halfwidth=6.0, step=2.0, polarity=0, sigma=1.0, iters=3,
               min_strength=4.0, fit_deg=None, end_margin=0.0):
    """Return dict(points Nx2, strength N, normal N x2, s N) of sub-pixel edge points.

    poly     : approximate polyline (list of [x, y]) along the edge
    polarity : +1 dark->bright along the normal, -1 bright->dark, 0 = strongest either way
               (normal = left-hand normal of the polyline direction, image coords)
    sigma    : Gaussian smoothing of the profile derivative (px)
    fit_deg  : if given, after each iteration the path is replaced by a polynomial fit of this
               degree (in arc length) to the found points (keeps the guide smooth)
    """
    img = mean_image() if img is None else img
    path = np.asarray(poly, float)
    out = None
    hw = halfwidth
    for it in range(iters):
        Q, nrm, s = _resample_polyline(path, step)
        if end_margin > 0:
            keep = (s > end_margin) & (s < s[-1] - end_margin)
            Q, nrm, s = Q[keep], nrm[keep], s[keep]
        offs = np.arange(-hw, hw + 1e-9, 0.25)
        X = Q[:, None, 0] + nrm[:, None, 0] * offs[None]
        Y = Q[:, None, 1] + nrm[:, None, 1] * offs[None]
        prof = map_coordinates(img, [Y.ravel(), X.ravel()], order=3, mode="nearest").reshape(X.shape)
        # derivative of Gaussian along the profile (sampling 0.25 px)
        from scipy.ndimage import gaussian_filter1d

        der = gaussian_filter1d(prof, sigma / 0.25, axis=1, order=1) / 0.25
        if polarity == 0:
            a = np.abs(der)
        else:
            a = der * polarity
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


def line_fit(P):
    """Total-least-squares line through points: returns (centre, direction, rms, max)."""
    P = np.asarray(P, float)
    c = P.mean(0)
    U, S, Vt = np.linalg.svd(P - c, full_matrices=False)
    d = Vt[0]
    n = np.array([-d[1], d[0]])
    r = (P - c) @ n
    return c, d, float(np.sqrt(np.mean(r ** 2))), float(np.max(np.abs(r)))


def sagitta(P):
    """Max deviation of a curve from its chord and signed quadratic bow (px)."""
    P = np.asarray(P, float)
    c, d, rms, mx = line_fit(P)
    n = np.array([-d[1], d[0]])
    t = (P - c) @ d
    r = (P - c) @ n
    q = np.polyfit(t, r, 2)
    L = t.max() - t.min()
    return float(q[0] * (L / 2) ** 2), rms
