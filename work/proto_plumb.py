"""Prototype plumb-line (automatic Canny chains) -- quick independent check of the distortion."""
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, label, convolve
from scipy.optimize import least_squares

from common import CACHE, H, W, K_from, undistort_points, save_json

img = np.load(f"{CACHE}/mean_aligned_gray.npy").astype(np.float64)
gs = gaussian_filter(img, 1.2)
gx = cv2.Sobel(gs, cv2.CV_64F, 1, 0, ksize=3) / 8
gy = cv2.Sobel(gs, cv2.CV_64F, 0, 1, ksize=3) / 8
mag = np.hypot(gx, gy)
e = cv2.Canny(np.clip(gs, 0, 255).astype(np.uint8), 20, 50, L2gradient=True) > 0
# remove junctions
nb = convolve(e.astype(int), np.ones((3, 3), int), mode="constant") - 1
e2 = e & (nb <= 2)
lab, n = label(e2, structure=np.ones((3, 3)))
chains = []
for i in range(1, n + 1):
    ys, xs = np.nonzero(lab == i) if False else (None, None)
objs = __import__("scipy.ndimage", fromlist=["find_objects"]).find_objects(lab)
for i, sl in enumerate(objs, start=1):
    if sl is None:
        continue
    sub = lab[sl] == i
    if sub.sum() < 60:
        continue
    ys, xs = np.nonzero(sub)
    ys = ys + sl[0].start
    xs = xs + sl[1].start
    # order along principal direction (chains are nearly straight pieces)
    P = np.column_stack([xs, ys]).astype(float)
    c = P.mean(0)
    u, s, vt = np.linalg.svd(P - c, full_matrices=False)
    t = (P - c) @ vt[0]
    o = np.argsort(t)
    P = P[o]
    # sub-pixel along gradient
    gxx = gx[P[:, 1].astype(int), P[:, 0].astype(int)]
    gyy = gy[P[:, 1].astype(int), P[:, 0].astype(int)]
    gn = np.hypot(gxx, gyy) + 1e-9
    dx, dy = gxx / gn, gyy / gn
    from scipy.ndimage import map_coordinates

    m0 = map_coordinates(mag, [P[:, 1] - dy, P[:, 0] - dx], order=1)
    m1 = map_coordinates(mag, [P[:, 1], P[:, 0]], order=1)
    m2 = map_coordinates(mag, [P[:, 1] + dy, P[:, 0] + dx], order=1)
    den = m0 - 2 * m1 + m2
    dlt = np.where(np.abs(den) > 1e-9, 0.5 * (m0 - m2) / den, 0)
    dlt = np.clip(dlt, -0.7, 0.7)
    P = P + np.column_stack([dx * dlt, dy * dlt])
    if np.ptp(t) < 80:
        continue
    chains.append(P)
print("chains", len(chains))


def split(P, tol=4.0, minlen=80):
    """recursive split into pieces whose chord deviation < tol"""
    out = []
    stack = [P]
    while stack:
        Q = stack.pop()
        if len(Q) < 20:
            continue
        a, b = Q[0], Q[-1]
        d = b - a
        L = np.hypot(*d)
        if L < minlen:
            continue
        n = np.array([-d[1], d[0]]) / L
        r = np.abs((Q - a) @ n)
        k = np.argmax(r)
        if r[k] > tol:
            stack += [Q[: k + 1], Q[k:]]
        else:
            out.append(Q)
    return out


pieces = [q for P in chains for q in split(P)]
print("pieces", len(pieces), "points", sum(len(p) for p in pieces))
F0 = 1300.0


def resid(x, pcs, ret_per=False):
    k1, k2, cx, cy = x
    K = K_from(F0, F0, cx, cy)
    out = []
    per = []
    for Q in pcs:
        n = undistort_points(Q[::2], K, [k1, k2, 0, 0, 0], iters=20) * F0
        c = n.mean(0)
        u, s, vt = np.linalg.svd(n - c, full_matrices=False)
        r = (n - c) @ vt[1]
        t = (n - c) @ vt[0]
        Lu = np.ptp(t); Ld = np.hypot(*(Q[-1] - Q[0]))
        r = r * (Ld / max(Lu, 1e-9))
        out.append(r / np.sqrt(len(r)))
        per.append(np.sqrt(np.mean(r ** 2)))
    return per if ret_per else np.concatenate(out)


x = np.array([-0.2, 0.0, W / 2, H / 2])
pcs = pieces
for rnd in range(4):
    r = least_squares(resid, x, args=(pcs,), loss="soft_l1", f_scale=0.3, x_scale=[0.01, 0.01, 10, 10],
                      bounds=([-1, -2, W / 2 - 400, H / 2 - 400], [0.5, 2, W / 2 + 400, H / 2 + 400]))
    x = r.x
    per = np.array(resid(x, pieces, True))
    thr = max(3 * np.median(per), 0.4)
    pcs = [p for p, e in zip(pieces, per) if e < thr]
    print(f"round {rnd}: k1={x[0]:.4f} k2={x[1]:.4f} cx={x[2]:.1f} cy={x[3]:.1f}  pieces kept {len(pcs)}/{len(pieces)}  median rms {np.median(per):.3f}")
save_json(dict(x=x, f0=F0, pieces=[p.tolist() for p in pcs]), f"{CACHE}/proto_plumb.json")
