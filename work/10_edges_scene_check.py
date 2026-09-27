"""Scene region: quick self-consistency check of the accepted straight scene edges (NOT a calibration result).

Straightens all straight_3d scene edges with a radial k1,k2 model (f fixed at the classification lens value,
distortion centre fixed at the image centre, then free) and reports the per-edge rms residual. Residuals are
measured in the undistorted image (simple, slightly biased towards weaker distortion - see CONTEXT.md);
the purpose is only to flag edges that do not behave like straight lines.
Writes work/cache/edges_scene_check.json."""
import json

import numpy as np
from scipy.optimize import least_squares

import edgelib
import edgesscene_lib as L
from common import K_from, save_json, undistort_points

d = json.load(open(f"{L.CACHE}/edges_scene.json"))
E = [e for e in d["edges"] if e["straight_3d"]]
K0, _, _, src = L.class_calib()
f = float(K0[0, 0])


def residuals(p, edges, free=False, per=False):
    k1, k2, cx, cy = (p if free else (p[0], p[1], 959.5, 539.5))
    K = K_from(f, f, cx, cy)
    out, pe = [], {}
    for e in edges:
        n = undistort_points(np.array(e["points"]), K, [k1, k2, 0, 0, 0]) * f
        c, dd, _, _ = edgelib.line_fit(n)
        r = (n - c) @ np.array([-dd[1], dd[0]])
        out.append(r / np.sqrt(len(r)))
        pe[e["id"]] = float(np.sqrt(np.mean(r ** 2)))
    return pe if per else np.concatenate(out)


fix = least_squares(residuals, [-0.3, 0.1], args=(E,))
free = least_squares(residuals, [-0.3, 0.1, 959.5, 539.5], args=(E, True))
pe = residuals(fix.x, E, per=True)
res = {"note": "quick consistency check only (f fixed = %.1f from %s; undistorted-image residuals)" % (f, src),
       "n_edges": len(E),
       "centre_fixed": {"k1": fix.x[0], "k2": fix.x[1], "rms_px": float(np.sqrt(np.sum(fix.fun ** 2) / len(E)))},
       "centre_free": {"k1": free.x[0], "k2": free.x[1], "cx": free.x[2], "cy": free.x[3],
                       "rms_px": float(np.sqrt(np.sum(free.fun ** 2) / len(E)))},
       "per_edge_rms_px_centre_fixed": dict(sorted(pe.items(), key=lambda kv: -kv[1]))}
save_json(res, f"{L.CACHE}/edges_scene_check.json")
print(json.dumps({k: v for k, v in res.items() if k != "per_edge_rms_px_centre_fixed"}, indent=1, default=float))
for k, v in list(res["per_edge_rms_px_centre_fixed"].items())[:8]:
    print(f"  {k:26s} {v:.3f}")
