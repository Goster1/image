"""Cart 80: print compact guide polylines (literal numbers for edges80_defs.py) from automatic
candidates.  usage: python3 10_edges_cart80_mkguide.py "aid[+aid...][:axis:lo:hi]" ...
axis = x or y restricts the union of the candidate points to lo..hi before fitting.
The printed polyline is a deg-3 fit in the chord frame, sampled at ~8 points, oriented top->bottom
(or left->right for flat edges)."""
import sys

import numpy as np

from common import CACHE, load_json
import edgelib


def guide(P, npts=8):
    c, d, _, _ = edgelib.line_fit(P)
    if abs(d[1]) > abs(d[0]):
        d = d if d[1] > 0 else -d
    else:
        d = d if d[0] > 0 else -d
    n = np.array([-d[1], d[0]])
    t = (P - c) @ d
    r = (P - c) @ n
    q = np.polyfit(t, r, min(3, max(1, len(P) // 30)))
    tt = np.linspace(t.min(), t.max(), npts)
    G = c[None] + tt[:, None] * d[None] + np.polyval(q, tt)[:, None] * n[None]
    return G


def main():
    auto = {e["aid"]: e for e in load_json(f"{CACHE}/edges80_auto.json")["edges"]}
    for spec in sys.argv[1:]:
        parts = spec.split(":")
        aids = [int(a) for a in parts[0].split("+")]
        P = np.vstack([np.asarray(auto[a]["points"]) for a in aids])
        if len(parts) == 4:
            ax = 0 if parts[1] == "x" else 1
            lo, hi = float(parts[2]), float(parts[3])
            P = P[(P[:, ax] >= lo) & (P[:, ax] <= hi)]
        G = guide(P)
        print(spec, "guide=" + str([[round(float(x), 1), round(float(y), 1)] for x, y in G]).replace(" ", ""))


if __name__ == "__main__":
    main()
