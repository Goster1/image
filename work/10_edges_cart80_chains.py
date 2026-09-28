"""Cart 80: chain the classified LSD segments (undistorted coords of the initial calibration), trace
each chain sub-pixel with growth along the real edge, and render review tiles.
Input: cache/edges80_lsd.npy (from 10_edges_cart80_candidates.py).
Output: cache/edges80_auto.json, cache/edgecrops_cart80/auto_tile*.png"""
import cv2
import numpy as np

from common import CACHE, save_json
import edgelib
import edges80lib as L

OUT = f"{CACHE}/edgecrops_cart80"
COL = {0: (0, 0, 255), 1: (0, 200, 0), 2: (255, 80, 0), 3: (0, 200, 255)}
CLS = ["cartX", "cartY", "cartZ", "other"]


def dist_to_curve(P, D):
    P = np.asarray(P)
    out = np.empty(len(P))
    for i0 in range(0, len(P), 200):
        B = P[i0:i0 + 200]
        out[i0:i0 + 200] = np.min(np.hypot(B[:, None, 0] - D[None, :, 0], B[:, None, 1] - D[None, :, 1]), axis=1)
    return out


def main():
    rows = np.load(f"{CACHE}/edges80_lsd.npy")
    chains = L.chain_segments(rows, ang_tol=2.0, dist_tol=1.5, gap_max=60.0, min_span=40.0, minlen=10.0)
    print(len(chains), "chains with span >= 40")
    auto = []
    for ci, ch in enumerate(sorted(chains, key=lambda c: -c["span"])):
        poly = L.guide_from_points(ch["poly"], deg=2 if ch["span"] > 120 else 1, n=40)
        res = L.grow_trace(poly, polarity=1, bounds=(1100, 1, 1760, 1000), halfwidth=3.0)
        if res is None or len(res["points"]) < 15:
            continue
        P = res["points"]
        length = float(np.sum(np.hypot(*np.diff(P, axis=0).T)))
        bow, rms = edgelib.sagitta(P)
        ang = L.classify_dir(P[0], P[-1])
        a = np.array([ang[c] for c in CLS[:3]])
        auto.append(dict(aid=len(auto), cls=ch["cls"], span_lsd=ch["span"], length=length,
                         ang=[float(v) for v in a], points=P, strength=res["strength"], noise=res["noise"],
                         bow=bow, rms=rms, guide=poly))
    # de-duplicate: drop traces whose points coincide (within 0.7 px) with a longer trace for > 60 %
    for e in auto:
        P = np.asarray(e["points"])
        seg = np.hypot(*np.diff(P, axis=0).T)
        s = np.r_[0, np.cumsum(seg)]
        t = np.arange(0, s[-1], 0.25)
        e["dense"] = np.column_stack([np.interp(t, s, P[:, 0]), np.interp(t, s, P[:, 1])])
    keep = []
    for e in sorted(auto, key=lambda e: -e["length"]):
        dup = False
        for k in keep:
            d = dist_to_curve(e["points"], k["dense"])
            if np.mean(d < 0.8) > 0.6:
                dup = True
                break
        if not dup:
            keep.append(e)
    for i, e in enumerate(keep):
        e["aid"] = i
        del e["dense"]
    print(len(keep), "unique traced candidates;", sum(e["length"] > 100 for e in keep), "longer than 100 px")
    save_json(dict(edges=keep), f"{CACHE}/edges80_auto.json")
    img = cv2.imread(f"{CACHE}/mean_aligned_color.png")
    tiles = [(1120, 0, 1720, 260), (1120, 230, 1720, 490), (1120, 460, 1720, 720), (1120, 700, 1720, 990)]
    sc = 2
    for ti, (x0, y0, x1, y1) in enumerate(tiles):
        crop = cv2.resize(img[y0:y1, x0:x1], None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
        crop = (crop * 0.7).astype(np.uint8)
        for e in keep:
            if e["length"] < 30:
                continue
            P = (np.asarray(e["points"]) - [x0, y0]) * sc
            ins = (P[:, 0] >= 0) & (P[:, 0] < crop.shape[1]) & (P[:, 1] >= 0) & (P[:, 1] < crop.shape[0])
            if ins.sum() < 3:
                continue
            k = int(np.argmin(e["ang"])) if min(e["ang"]) < 3 else 3
            cv2.polylines(crop, [np.round(P).astype(np.int32)], False, COL[k], 1, cv2.LINE_AA)
            m = P[ins][len(P[ins]) // 2]
            cv2.putText(crop, str(e["aid"]), (int(m[0]) + 3, int(m[1]) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(f"{OUT}/auto_tile{ti}.png", crop)


if __name__ == "__main__":
    main()
