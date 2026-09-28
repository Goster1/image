"""Exploration: chain LSD segments into long candidate edges in region cart310, trace them
sub-pixel and write an id-labelled overview (work/cache/edgecrops_cart310/_chains*.png) plus
work/cache/edgecrops_cart310/_chains.json.  The curated edge list in 10_edges_cart310_trace.py
was built from this (by visual inspection of the overview and of per-edge crops).
"""
import cv2
import numpy as np

from common import CACHE, save_json
import edges310lib as E

OUT = f"{CACHE}/edgecrops_cart310"


def main():
    segs = E.lsd_segments(minlen=10)
    chains = E.chain_segments(segs, min_span=50)
    print(len(segs), "segments ->", len(chains), "chains")
    res = []
    for i, ch in enumerate(chains):
        tr = E.trace_clean(ch["poly"], polarity=1, halfwidth=3.0, fit_deg=2)
        if tr is None:
            continue
        for part in E.split_gaps(tr, max_gap=10):
            P = part["points"]
            rms, bow, L = E.chord_stats(P)
            if L < 50:
                continue
            ang = E.direction_angles(P)
            cls = min(ang, key=ang.get)
            res.append(dict(cid=len(res), chain=i, points=P, strength=part["strength"], L=L, rms=rms, bow=bow,
                            ang=ang, cls=cls if ang[cls] < 4 else "?", smed=float(np.median(part["strength"]))))
    res.sort(key=lambda r: -r["L"])
    for k, r in enumerate(res):
        r["cid"] = k
    print("by class", {c: sum(1 for r in res if r["cls"] == c) for c in "XYZ?"})
    col = E.color_image().copy()
    colors = {"X": (0, 0, 255), "Y": (0, 200, 0), "Z": (255, 0, 0), "?": (0, 255, 255)}
    for r in res:
        P = r["points"]
        cv2.polylines(col, [np.round(P * 4).astype(np.int32)], False, colors[r["cls"]], 1, cv2.LINE_AA, shift=2)
    for ti, (x0, y0) in enumerate([(150, 0), (450, 0), (150, 330), (450, 330), (300, 660), (550, 660)]):
        tile = cv2.resize(col[y0:y0 + 350, x0:x0 + 350], None, fx=2.6, fy=2.6, interpolation=cv2.INTER_CUBIC)
        for r in res:
            P = r["points"]
            m = P[len(P) // 2]
            if x0 <= m[0] < x0 + 350 and y0 <= m[1] < y0 + 350:
                p = ((m - [x0, y0]) * 2.6).astype(int)
                cv2.putText(tile, str(r["cid"]), tuple(p), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
                cv2.putText(tile, str(r["cid"]), tuple(p), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors[r["cls"]], 1)
        cv2.imwrite(f"{OUT}/_chains_tile{ti}.png", tile)
    cv2.imwrite(f"{OUT}/_chains_all.png", col[0:1010, 150:900])
    save_json([{k: v for k, v in r.items()} for r in res], f"{OUT}/_chains.json")
    for r in res[:80]:
        P = r["points"]
        print(f"{r['cid']:3d} {r['cls']} L={r['L']:6.1f} n={len(P):4d} rms={r['rms']:.2f} bow={r['bow']:+.2f} "
              f"s={r['smed']:.1f} ang=({r['ang']['X']:.1f},{r['ang']['Y']:.1f},{r['ang']['Z']:.1f}) "
              f"{np.round(P[0], 1)}->{np.round(P[-1], 1)}")


if __name__ == "__main__":
    main()
