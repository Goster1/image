"""Exploration step 3: cluster the traced LSD chain pieces of region cart310 that lie on the same
physical line (collinear, same polarity), to build long guide polylines for the curated list.
Prints clusters with their extent; writes _clusters.json and an overview image.
"""
import cv2
import numpy as np

from common import CACHE, load_json, save_json
import edges310lib as E
import edgelib

OUT = f"{CACHE}/edgecrops_cart310"


def main():
    ch = load_json(f"{OUT}/_chains.json")
    items = []
    for r in ch:
        P = np.array(r["points"])
        if r["L"] < 25:
            continue
        c, d, rms, _ = edgelib.line_fit(P)
        o = P[-1] - P[0]
        if d @ o < 0:
            d = -d
        items.append(dict(cid=r["cid"], P=P, c=c, d=d, cls=r["cls"], L=r["L"]))
    n = len(items)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        A = items[i]
        for j in range(i + 1, n):
            B = items[j]
            if A["d"] @ B["d"] < np.cos(np.radians(3.0)):
                continue
            # distance of B's points to A's line and vice versa, with gap-dependent tolerance
            nrmA = np.array([-A["d"][1], A["d"][0]])
            tA = (A["P"] - A["c"]) @ A["d"]
            tB = (B["P"] - A["c"]) @ A["d"]
            gap = max(0.0, max(tB.min() - tA.max(), tA.min() - tB.max()))
            if gap > 250:
                continue
            rB = (B["P"] - A["c"]) @ nrmA
            # quadratic-free: use local linear fit of A's end nearest to B
            if tB.mean() > tA.mean():
                sel = tA > tA.max() - 60
            else:
                sel = tA < tA.min() + 60
            cl, dl, _, _ = edgelib.line_fit(A["P"][sel]) if sel.sum() > 5 else (A["c"], A["d"], 0, 0)
            nl = np.array([-dl[1], dl[0]])
            rB = (B["P"] - cl) @ nl
            tol = 1.2 + 0.006 * gap
            if np.max(np.abs(rB)) < tol + 0.8 * 0 and np.median(np.abs(rB)) < tol:
                parent[find(j)] = find(i)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    clusters = []
    for g, idx in groups.items():
        P = np.vstack([items[i]["P"] for i in idx])
        c, d, rms, mx = edgelib.line_fit(P)
        d0 = items[idx[0]]["d"]
        if d @ d0 < 0:
            d = -d
        t = (P - c) @ d
        o = np.argsort(t)
        P = P[o]
        span = t.max() - t.min()
        cover = sum(items[i]["L"] for i in idx)
        clusters.append(dict(members=[items[i]["cid"] for i in idx], P=P, span=float(span), cover=float(cover),
                             rms=rms, cls=items[idx[0]]["cls"]))
    clusters.sort(key=lambda c: -c["span"])
    col = E.color_image().copy()
    rng = np.random.default_rng(1)
    for k, c in enumerate(clusters):
        c["k"] = k
        colr = tuple(int(v) for v in rng.integers(60, 255, 3))
        c["color"] = colr
        for i in c["members"]:
            pass
        cv2.polylines(col, [np.round(c["P"] * 4).astype(np.int32)], False, colr, 1, cv2.LINE_AA, shift=2)
    for ti, (x0, y0) in enumerate([(150, 0), (450, 0), (150, 330), (450, 330), (300, 660), (550, 660)]):
        tile = cv2.resize(col[y0:y0 + 350, x0:x0 + 350], None, fx=2.6, fy=2.6, interpolation=cv2.INTER_CUBIC)
        for c in clusters:
            if c["span"] < 60:
                continue
            P = c["P"]
            for m in (P[len(P) // 2], P[len(P) // 6], P[5 * len(P) // 6]):
                if x0 <= m[0] < x0 + 350 and y0 <= m[1] < y0 + 350:
                    p = ((m - [x0, y0]) * 2.6).astype(int)
                    cv2.putText(tile, str(c["k"]), tuple(p), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
                    cv2.putText(tile, str(c["k"]), tuple(p), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c["color"], 1)
        cv2.imwrite(f"{OUT}/_clusters_tile{ti}.png", tile)
    save_json([{k: v for k, v in c.items()} for c in clusters], f"{OUT}/_clusters.json")
    for c in clusters:
        if c["span"] < 60:
            continue
        print(f"{c['k']:3d} {c['cls']} span={c['span']:6.1f} cover={c['cover']:6.1f} rms={c['rms']:.2f} "
              f"members={c['members']} {np.round(c['P'][0], 1)}->{np.round(c['P'][-1], 1)}")


if __name__ == "__main__":
    main()
