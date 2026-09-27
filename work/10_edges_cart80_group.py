"""Cart 80: group automatic candidates that lie on the same image curve (same physical edge seen in
pieces).  Two candidates are joined when, in undistorted coordinates of the initial calibration, the
points of one lie within `tol` px of the quadratic fitted to the other (and vice versa), with the same
contrast polarity.  Prints the groups (used to write the curated guide table edges80_defs.py)."""
import numpy as np
from scipy.ndimage import map_coordinates

from common import CACHE, load_json
import edgelib
import edges80lib as L

CLS = ["cartX", "cartY", "cartZ"]


def polarity_sig(P):
    """mean intensity difference (right-hand side minus left-hand side) along the ordered points,
    with the points ordered top->bottom (or left->right for flat edges)."""
    P = np.asarray(P, float)
    if abs(P[-1, 1] - P[0, 1]) > abs(P[-1, 0] - P[0, 0]):
        if P[-1, 1] < P[0, 1]:
            P = P[::-1]
    elif P[-1, 0] < P[0, 0]:
        P = P[::-1]
    d = np.gradient(P, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    n = np.column_stack([-d[:, 1], d[:, 0]])
    a = map_coordinates(L.IMG, [P[:, 1] + 2.5 * n[:, 1], P[:, 0] + 2.5 * n[:, 0]], order=1)
    b = map_coordinates(L.IMG, [P[:, 1] - 2.5 * n[:, 1], P[:, 0] - 2.5 * n[:, 0]], order=1)
    return float(np.median(a - b))


def main(tol=1.0, minlen=40):
    auto = [e for e in load_json(f"{CACHE}/edges80_auto.json")["edges"] if e["length"] >= minlen]
    info = []
    for e in auto:
        P = np.asarray(e["points"])
        U = L.undist_px(P)
        c, d, _, _ = edgelib.line_fit(U)
        n = np.array([-d[1], d[0]])
        t = (U - c) @ d
        q = np.polyfit(t, (U - c) @ n, 2 if np.ptp(t) > 150 else 1)
        k = int(np.argmin(e["ang"]))
        cls = CLS[k] if e["ang"][k] < 3 else "other"
        info.append(dict(aid=e["aid"], U=U, c=c, d=d, n=n, q=q, cls=cls, pol=np.sign(polarity_sig(P)),
                         L=e["length"], tr=(t.min(), t.max())))
    N = len(info)
    parent = list(range(N))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def off(a, b):
        t = (b["U"] - a["c"]) @ a["d"]
        r = (b["U"] - a["c"]) @ a["n"] - np.polyval(a["q"], t)
        return np.median(r), np.percentile(np.abs(r), 90)

    for i in range(N):
        for j in range(i + 1, N):
            a, b = info[i], info[j]
            if a["cls"] != b["cls"] or a["pol"] != b["pol"]:
                continue
            if abs(a["d"] @ b["d"]) < np.cos(np.radians(3)):
                continue
            m1, p1 = off(a, b)
            m2, p2 = off(b, a)
            if abs(m1) < tol and abs(m2) < tol and p1 < 2 * tol and p2 < 2 * tol:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(N):
        groups.setdefault(find(i), []).append(i)
    out = []
    for g in groups.values():
        tot = sum(info[i]["L"] for i in g)
        out.append((tot, g))
    out.sort(key=lambda x: -x[0])
    for tot, g in out:
        if tot < 60:
            continue
        P = np.vstack([np.asarray(auto[i]["points"]) for i in g])
        print(f"{info[g[0]]['cls']:6s} total {tot:5.0f}  aids {[info[i]['aid'] for i in g]}  pol {int(info[g[0]]['pol'])}"
              f"  y {P[:, 1].min():.0f}..{P[:, 1].max():.0f}  x {P[:, 0].min():.0f}..{P[:, 0].max():.0f}")


if __name__ == "__main__":
    main()
