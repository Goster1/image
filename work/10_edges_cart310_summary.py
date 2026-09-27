"""Summary numbers for the cart310 edge set (counts, lengths, exact lines) and an observation:
height of the visible shelf-front outlines, obtained by intersecting the viewing rays of the traced
points with the plane Y=0 using the initial calibration (only indicative: that calibration is
~5 px accurate)."""
import numpy as np

from common import CACHE, load_json, rodrigues, undistort_points, save_json
import edges310lib as E


def main():
    D = load_json(f"{CACHE}/edges_cart310.json")
    K, dist, rv, tv = E.calib310()
    R = rodrigues(rv)
    C = -R.T @ tv
    by = {}
    tot = 0.0
    for e in D["edges"]:
        P = np.array(e["points"])
        L = float(np.sum(np.hypot(*np.diff(P, axis=0).T)[np.hypot(*np.diff(P, axis=0).T) < 8]))
        span = float(np.hypot(*(P[-1] - P[0])))
        by.setdefault(e["direction"], []).append((e["id"], span, L, e["verified"]))
        tot += L
    for k, v in by.items():
        print(f"{k:14s} n={len(v):2d} span_sum={sum(s for _, s, _, _ in v):7.1f} traced_len={sum(l for _, _, l, _ in v):7.1f}  "
              + ", ".join(f"{i.replace('cart310_', '')}({s:.0f},{q})" for i, s, _, q in v))
    print("total traced length (gaps excluded) %.0f px, edges %d" % (tot, len(D["edges"])))
    print("exact model lines:", [e["id"] for e in D["edges"] if e["model_line"] and e["model_line"].get("exact")])
    print("\nvisible shelf-front outline height (ray x plane Y=0, initial calibration):")
    obs = {}
    for e in D["edges"]:
        if "_front_" not in e["id"] or not e["id"].startswith("cart310_x_"):
            continue
        P = np.array(e["points"])
        n = undistort_points(P, K, dist)
        d = (R.T @ np.column_stack([n, np.ones(len(n))]).T).T
        t = (0.0 - C[1]) / d[:, 1]
        X = C[None] + t[:, None] * d
        z_nom = e["model_line"]["fixed"]["Z"]
        obs[e["id"]] = dict(Z_median=float(np.median(X[:, 2])), Z_nominal=z_nom,
                            X_range=[float(X[:, 0].min()), float(X[:, 0].max())])
        print(f"  {e['id']:26s} Z_vis={np.median(X[:, 2]):7.0f} mm (nominal surface {z_nom}), X {X[:, 0].min():.0f}..{X[:, 0].max():.0f}")
    save_json(obs, f"{CACHE}/edgecrops_cart310/_front_heights.json")


if __name__ == "__main__":
    main()


def model_offsets():
    """signed normal distance (px) of the traced points from the projected model line (initial calib)."""
    from scipy.spatial import cKDTree
    D = load_json(f"{CACHE}/edges_cart310.json")
    out = {}
    for e in D["edges"]:
        ml = e["model_line"]
        if not ml:
            continue
        P = np.array(e["points"])
        fx = ml["fixed"]
        free = ml["free"]
        a, b = ml["range_mm"]
        if free == "Z" and len(fx) == 2:
            pass
        elif len(fx) < 2:
            continue
        p0 = {k: 0.0 for k in "XYZ"}
        p1 = {k: 0.0 for k in "XYZ"}
        p0.update(fx)
        p1.update(fx)
        p0[free], p1[free] = a - 300, b + 300
        Q = E.project_line([p0["X"], p0["Y"], p0["Z"]], [p1["X"], p1["Y"], p1["Z"]], n=4000)
        tr = cKDTree(Q)
        dd, idx = tr.query(P)
        # sign: along the local normal of the model curve
        i = np.clip(idx, 1, len(Q) - 2)
        tang = Q[i + 1] - Q[i - 1]
        tang /= np.linalg.norm(tang, axis=1, keepdims=True)
        nrm = np.column_stack([-tang[:, 1], tang[:, 0]])
        sd = np.sum((P - Q[idx]) * nrm, axis=1)
        out[e["id"]] = float(np.median(sd))
        print(f"  {e['id']:28s} exact={ml.get('exact')}  median offset from projected model line {np.median(sd):+6.1f} px "
              f"(min {sd.min():+.1f}, max {sd.max():+.1f})")
    return out


if __name__ == "__main__":
    print("\noffsets of traced edges from the projected model lines (initial calibration):")
    model_offsets()
