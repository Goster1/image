"""Synthetic validation of 12_review_markers_bias.py (and of the stage-11 edge corners).

A code with the real geometry of a sticker (quad of 80:5 or 310:1) is rendered at 16x supersampling in
LINEAR intensity (black 0.08, white 0.9, white margin 10 mm, then a mid-gray card), blurred with a
Gaussian PSF (sigma in px), integrated over the pixel, then optionally tone-mapped (gamma encoding
v = lin^(1/g)) and scaled to 0..255 with noise. Then: stage-11 edge corners (11_markers_lib.edge_quad)
and the interior-transition bias fit. Reports the corner error (along the diagonal, inward < 0) and the
fitted b_inner / b_outer.
"""
import importlib

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from common import CACHE, load_json

L = importlib.import_module("11_markers_lib")
B = importlib.import_module("12_review_markers_bias")


def render(quad, mid, sigma, gamma, ss=16, noise=1.0, seed=0, flare=0.0):
    quad = np.asarray(quad, float)
    x0, y0 = np.floor(quad.min(0) - 20).astype(int)
    x1, y1 = np.ceil(quad.max(0) + 20).astype(int)
    Wd, Hd = x1 - x0, y1 - y0
    # cell coords -> image (pixel centres at integers)
    Hm = cv2.getPerspectiveTransform(np.array([[0, 0], [6, 0], [6, 6], [0, 6]], np.float32), quad.astype(np.float32)).astype(float)
    Hinv = np.linalg.inv(Hm)
    ys, xs = np.mgrid[0:Hd * ss, 0:Wd * ss].astype(float)
    X = x0 - 0.5 + (xs + 0.5) / ss
    Y = y0 - 0.5 + (ys + 0.5) / ss
    q = np.stack([X, Y, np.ones_like(X)], -1) @ Hinv.T
    u, v = q[..., 0] / q[..., 2], q[..., 1] / q[..., 2]
    grid = L.code_grid(mid)
    img = np.full(u.shape, 0.45)  # card / background
    marg = (u > -0.667) & (u < 6.667) & (v > -0.667) & (v < 6.667)
    img[marg] = 0.9
    inside = (u >= 0) & (u < 6) & (v >= 0) & (v < 6)
    ci = np.clip(np.floor(u).astype(int), 0, 5)
    ri = np.clip(np.floor(v).astype(int), 0, 5)
    img[inside] = np.where(grid[ri[inside], ci[inside]] == 1, 0.9, 0.08)
    img = gaussian_filter(img, sigma * ss)
    if flare > 0:
        img = img + flare * gaussian_filter(img, 3.0 * ss)
    img = img.reshape(Hd, ss, Wd, ss).mean((1, 3))
    img = np.clip(img, 0, None) ** (1.0 / gamma)
    img = img / img.max() * 230
    rng = np.random.default_rng(seed)
    img = img + rng.normal(0, noise, img.shape)
    full = np.zeros((1080, 1920))
    full[:] = img.mean()
    full[y0:y1, x0:x1] = img
    return full


def run(quad, mid, sigma, gamma, flare=0.0):
    img = render(quad, mid, sigma, gamma, flare=flare)
    # stage-11 edge corners, started from a perturbed quad
    q0 = np.asarray(quad, float) + np.random.default_rng(1).normal(0, 0.7, (4, 2))
    e = L.edge_quad(img, q0, iters=3)
    qe = e["corners"]
    c = np.asarray(quad).mean(0)
    rad = [float(np.dot(qe[i] - quad[i], (quad[i] - c) / np.linalg.norm(quad[i] - c))) for i in range(4)]
    rows = B.transitions(img, qe, L.code_grid(mid))
    cell = rows[0][3]
    x2, se2, _ = B.fit_two(rows)
    beta, _, _ = B.fit_beta(rows)
    return np.mean(rad), beta * cell, x2[0] * cell, x2[1] * cell


def main():
    M = {(m["cart"], m["id"]): m for m in load_json(f"{CACHE}/markers_final.json")}
    for key in [(80, 5), (310, 1)]:
        quad = np.array(M[key]["corners_px"], float)
        for sigma in (0.5, 0.9):
            for gamma in (1.0, 2.2):
                for flare in (0.0, 0.15):
                    rad, b, bi, bo = run(quad, key[1], sigma, gamma, flare)
                    print(f"{key} sigma={sigma} gamma={gamma} flare={flare}: corner radial error {rad:+.3f} px "
                          f"(side normal ~ {rad / np.sqrt(2):+.3f}); fitted b(common) {b:+.3f}  b_inner {bi:+.3f}  b_outer {bo:+.3f} px")


if __name__ == "__main__":
    main()
