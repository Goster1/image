"""Diagnostics for curated cart310 edges: residual of the traced points from a smooth cubic, in
20 px bins along the edge (to spot bends at the ends / jumps), and gradient strength."""
import sys
import numpy as np

from common import CACHE, load_json
import edgelib


def main(ids):
    E = load_json(f"{CACHE}/edges_cart310_partial.json") if False else None
    import importlib
    tr = importlib.import_module("10_edges_cart310_trace")
    edges = tr.main(ids)
    for e in edges:
        P = np.array(e["points"])
        S = np.array(e["strength"])
        c, d, _, _ = edgelib.line_fit(P)
        if d @ (P[-1] - P[0]) < 0:
            d = -d
        n = np.array([-d[1], d[0]])
        t = (P - c) @ d
        r = (P - c) @ n
        q = np.polyfit(t, r, 3)
        res = r - np.polyval(q, t)
        q2 = np.polyfit(t, r, 2)
        res2 = r - np.polyval(q2, t)
        # straightness after undistortion with a rough reference lens (diagnostic only)
        from common import K_from, undistort_to_pixels
        Kr = K_from(1290, 1290, 959.5, 539.5)
        U = undistort_to_pixels(P, Kr, [-0.29, 0.08, 0, 0, 0])
        cu, du, rmsu, mxu = edgelib.line_fit(U)
        nu = np.array([-du[1], du[0]])
        ru = (U - cu) @ nu
        print(e["id"], "cubic rms %.3f  quad rms %.3f | undistorted(ref k1=-0.29,k2=0.08) line rms %.3f max %.3f"
              % (res.std(), res2.std(), rmsu, mxu))
        res2 = ru
        bins = np.arange(t.min(), t.max() + 20, 20)
        line = []
        for a, b in zip(bins[:-1], bins[1:]):
            s = (t >= a) & (t < b)
            if s.sum():
                i = np.nonzero(s)[0][len(np.nonzero(s)[0]) // 2]
                line.append(f"[{P[i,0]:.0f},{P[i,1]:.0f}] r={res2[s].mean():+.2f} s={S[s].mean():.0f}")
        print("   " + " | ".join(line))


if __name__ == "__main__":
    main(sys.argv[1:])
