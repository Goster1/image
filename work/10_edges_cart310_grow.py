"""Exploration step 2: grow every LSD chain of region cart310 along its full visible length
(edges310lib.grow), remove duplicates, and write an id-labelled overview for curation:
work/cache/edgecrops_cart310/_grown*.png and _grown.json.
"""
import cv2
import numpy as np

from common import CACHE, save_json
import edges310lib as E
import edgelib

OUT = f"{CACHE}/edgecrops_cart310"
BOX = (150, 0, 905, 1012)


def dup_frac(A, B, tol=0.8):
    """fraction of points of A lying within tol px of polyline B"""
    from scipy.spatial import cKDTree
    Bd = E.resample_path(B, 0.5)
    d, _ = cKDTree(Bd).query(A)
    return float(np.mean(d < tol))


def main():
    segs = E.lsd_segments(minlen=10)
    chains = E.chain_segments(segs, min_span=40)
    grown = []
    for i, ch in enumerate(chains):
        tr = E.trace_clean(ch["poly"], polarity=1, halfwidth=3.0, fit_deg=2)
        if tr is None:
            continue
        for part in E.split_gaps(tr, max_gap=10):
            P = part["points"]
            if np.hypot(*(P[-1] - P[0])) < 40:
                continue
            # skip if already covered by a grown edge
            if any(dup_frac(P, g["points"]) > 0.6 for g in grown):
                continue
            G, S = E.grow(P, polarity=1, stop_box=BOX)
            G = E.resample_path(G, 2.0)
            grown.append(dict(points=G, seed=i))
    # final re-trace + stats
    res = []
    for g in grown:
        G = g["points"]
        L = np.hypot(*(G[-1] - G[0]))
        if L < 60:
            continue
        tr = E.trace_clean(G, polarity=1, halfwidth=2.0, fit_deg=3, max_res=0.5)
        if tr is None:
            continue
        P = tr["points"]
        if any(dup_frac(P, r["points"]) > 0.6 for r in res):
            continue
        rms, bow, L = E.chord_stats(P)
        ang = E.direction_angles(P)
        cls = min(ang, key=ang.get)
        gaps = np.hypot(*np.diff(P, axis=0).T)
        res.append(dict(points=P, strength=tr["strength"], L=L, rms=rms, bow=bow, ang=ang,
                        cls=cls if ang[cls] < 4 else "?", smed=float(np.median(tr["strength"])),
                        maxgap=float(gaps.max()), n=len(P)))
    res.sort(key=lambda r: -r["L"])
    for k, r in enumerate(res):
        r["gid"] = k
    print(len(res), "grown edges; by class", {c: sum(1 for r in res if r["cls"] == c) for c in "XYZ?"})
    col = E.color_image().copy()
    colors = {"X": (0, 0, 255), "Y": (0, 200, 0), "Z": (255, 0, 0), "?": (0, 255, 255)}
    for r in res:
        cv2.polylines(col, [np.round(r["points"] * 4).astype(np.int32)], False, colors[r["cls"]], 1, cv2.LINE_AA, shift=2)
    for ti, (x0, y0) in enumerate([(150, 0), (450, 0), (150, 330), (450, 330), (300, 660), (550, 660)]):
        tile = cv2.resize(col[y0:y0 + 350, x0:x0 + 350], None, fx=2.6, fy=2.6, interpolation=cv2.INTER_CUBIC)
        for r in res:
            P = r["points"]
            for m in (P[len(P) // 2], P[len(P) // 5], P[4 * len(P) // 5]) if r["L"] > 300 else (P[len(P) // 2],):
                if x0 <= m[0] < x0 + 350 and y0 <= m[1] < y0 + 350:
                    p = ((m - [x0, y0]) * 2.6).astype(int)
                    cv2.putText(tile, str(r["gid"]), tuple(p), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
                    cv2.putText(tile, str(r["gid"]), tuple(p), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors[r["cls"]], 1)
        cv2.imwrite(f"{OUT}/_grown_tile{ti}.png", tile)
    cv2.imwrite(f"{OUT}/_grown_all.png", col[0:1012, 150:905])
    save_json(res, f"{OUT}/_grown.json")
    for r in res:
        P = r["points"]
        print(f"{r['gid']:3d} {r['cls']} L={r['L']:6.1f} n={r['n']:4d} gap={r['maxgap']:5.1f} rms={r['rms']:.2f} "
              f"bow={r['bow']:+.2f} s={r['smed']:5.1f} ang=({r['ang']['X']:.1f},{r['ang']['Y']:.1f},{r['ang']['Z']:.1f}) "
              f"{np.round(P[0], 1)}->{np.round(P[-1], 1)}")


if __name__ == "__main__":
    main()
