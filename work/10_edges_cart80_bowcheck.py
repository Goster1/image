"""Cart 80: plausibility check of the raw bow of every traced edge against the bow that a straight 3D
line with the same end points would show under the initial lens model (information only; the initial
distortion is poorly determined, so only gross mismatches - bent structures - are flagged).
Also compared with the prototype plumb-line model of CONTEXT.md (k1=-0.29, k2=0.08, f=1300, centre).
Output: printed table, cache/edges80_bowcheck.json"""
import numpy as np

from common import CACHE, load_json, save_json, K_from, project
import edgelib
import edges80lib as L


def predicted_bow(P, K, dist):
    """straight 3D line through the (undistorted) end points of P, re-distorted -> sagitta (px)."""
    from common import undistort_points
    n = undistort_points(np.array([P[0], P[-1]]), K, dist)
    t = np.linspace(0, 1, 60)[:, None]
    N = n[0][None] * (1 - t) + n[1][None] * t
    X = np.column_stack([N, np.ones(len(N))])
    Q = project(X, K, dist)
    return edgelib.sagitta(Q)[0]


def main():
    edges = load_json(f"{CACHE}/edges_cart80.json")["edges"]
    models = {"initial": (L.K0, L.D0), "proto_plumb": (K_from(1300, 1300, 959.5, 539.5), np.array([-0.29, 0.08, 0, 0, 0]))}
    out = []
    for e in edges:
        P = np.array(e["points"])
        row = dict(id=e["id"], observed_bow=e["chord_bow_px"])
        for nm, (K, dist) in models.items():
            row[f"pred_bow_{nm}"] = round(float(predicted_bow(P, K, dist)), 2)
        out.append(row)
        print(f"{e['id']:30s} obs {row['observed_bow']:6.2f}  initial {row['pred_bow_initial']:6.2f}  "
              f"proto {row['pred_bow_proto_plumb']:6.2f}")
    save_json(out, f"{CACHE}/edges80_bowcheck.json")


if __name__ == "__main__":
    main()
