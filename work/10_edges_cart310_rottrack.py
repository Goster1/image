"""Exploration step 4: find the long cart-X edges of cart 310 by tracking gradient extrema row by
row in an image rotated so that cart-X lines are near-vertical (the rotation is only used to FIND
guides; all measurements are later redone on the unrotated mean image with edgelib.trace_edge).
Output: work/cache/edgecrops_cart310/_rottracks.json (guides in full-image coordinates).
"""
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from common import CACHE, save_json
import edges310lib as E

OUT = f"{CACHE}/edgecrops_cart310"
ANG = np.degrees(np.arctan2(0.35, 0.93))
CEN = (600, 500)


def rot_matrix():
    return cv2.getRotationMatrix2D(CEN, -ANG, 1.0)


def main():
    M = rot_matrix()
    Mi = cv2.invertAffineTransform(M)
    R = cv2.warpAffine(E.IMG.astype(np.float32), M, (1920, 1080), flags=cv2.INTER_CUBIC)
    R = gaussian_filter1d(R, 1.0, axis=0)
    D = gaussian_filter1d(R, 1.0, axis=1, order=1)
    xs0, xs1 = 360, 900
    tracks = []
    taken = np.zeros(D.shape, bool)
    seeds = []
    for y in range(60, 1000, 40):
        d = D[y, xs0:xs1]
        for i in range(2, len(d) - 2):
            if abs(d[i]) > 10 and abs(d[i]) >= abs(d[i - 1]) and abs(d[i]) >= abs(d[i + 1]):
                seeds.append((abs(d[i]), xs0 + i, y, np.sign(d[i])))
    seeds.sort(reverse=True)
    for s, x0, y0, pol in seeds:
        if taken[y0, x0]:
            continue
        pts = {y0: float(x0)}
        med = s
        for dirn in (-1, 1):
            xs = [float(x0)]
            ys = [y0]
            y = y0
            gap = 0
            while 0 < y + dirn < 1079 and gap <= 10:
                y += dirn
                # predict from the last ~40 rows
                yy, xx = np.array(ys[-40:]), np.array(xs[-40:])
                if len(yy) >= 6:
                    p = np.polyfit(yy, xx, 1)
                    xp = np.polyval(p, y)
                else:
                    xp = xs[-1]
                xi = int(round(xp))
                if xi < 3 or xi > 1915:
                    break
                win = D[y, xi - 2:xi + 3] * pol
                j = int(np.argmax(win))
                val = win[j]
                xc = xi - 2 + j
                if 0 < j < 4:
                    a, b, c = win[j - 1], win[j], win[j + 1]
                    den = a - 2 * b + c
                    off = 0.5 * (a - c) / den if abs(den) > 1e-9 else 0.0
                else:
                    off = 0.0
                xn = xc + np.clip(off, -0.5, 0.5)
                if val > max(7.0, 0.35 * med) and abs(xn - xp) < 1.2 and 0 < j < 4:
                    xs.append(xn)
                    ys.append(y)
                    gap = 0
                else:
                    gap += 1
            for a, b in zip(ys, xs):
                pts[a] = b
        yy = np.array(sorted(pts))
        xx = np.array([pts[k] for k in yy])
        if yy.max() - yy.min() < 60:
            continue
        for a, b in zip(yy, xx):
            taken[a, max(0, int(round(b)) - 1):int(round(b)) + 2] = True
        tracks.append(dict(y=yy, x=xx, pol=int(pol), s=float(s)))
    # back to full-image coordinates
    out = []
    for t in tracks:
        Pr = np.column_stack([t["x"], t["y"]])
        P = Pr @ Mi[:, :2].T + Mi[:, 2]
        out.append(dict(points=P, rot=Pr, pol=t["pol"], span=float(t["y"].max() - t["y"].min()), s=t["s"]))
    out.sort(key=lambda t: -t["span"])
    for k, t in enumerate(out):
        t["k"] = k
    save_json(out, f"{OUT}/_rottracks.json")
    col = E.color_image().copy()
    for t in out:
        colr = (0, 0, 255) if t["pol"] > 0 else (255, 128, 0)
        cv2.polylines(col, [np.round(t["points"] * 4).astype(np.int32)], False, colr, 1, cv2.LINE_AA, shift=2)
    Rc = cv2.warpAffine(col, M, (1920, 1080), flags=cv2.INTER_CUBIC)
    for t in out:
        if t["span"] < 100:
            continue
        m = t["rot"][len(t["rot"]) // 2]
        cv2.putText(Rc, str(t["k"]), (int(m[0]) + 2, int(m[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
    for i, (y0, y1) in enumerate([(40, 380), (360, 700), (680, 1020)]):
        cv2.imwrite(f"{OUT}/_rottracks_{i}.png", cv2.resize(Rc[y0:y1, 360:900], None, fx=2, fy=2,
                                                              interpolation=cv2.INTER_CUBIC))
    for t in out:
        if t["span"] >= 100:
            print(t["k"], t["pol"], round(t["span"]), round(t["s"]), np.round(t["rot"][0]), np.round(t["rot"][-1]))


if __name__ == "__main__":
    main()
